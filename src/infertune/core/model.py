"""Model architecture description.

The design commitment here is that there is no single KV-cache formula. The familiar
``2 * layers * kv_heads * head_dim * bytes`` expression describes MHA and GQA and is
wrong — often by an order of magnitude — for MLA, and structurally wrong for recurrent
hybrids. So `AttentionSpec` dispatches on architecture kind, and refuses to answer for
kinds it does not yet model rather than returning a confidently wrong number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .dtypes import DType, bytes_for


class AttentionKind(Enum):
    """How a model stores attention state, which determines how memory scales."""

    MHA = "mha"
    """Multi-head attention: one KV head per query head."""

    GQA = "gqa"
    """Grouped-query attention: KV heads shared across query heads."""

    MLA = "mla"
    """Multi-head latent attention (DeepSeek-family): compressed latent KV."""

    SLIDING_WINDOW = "sliding_window"
    """Every layer attends over a bounded window, capping KV per sequence."""

    HYBRID = "hybrid"
    """Interleaved local/global layers; only some layers hold full-length KV."""

    RECURRENT_HYBRID = "recurrent_hybrid"
    """Mamba/SSM blocks: fixed state per sequence, independent of context length."""


class UnsupportedArchitectureError(NotImplementedError):
    """Raised when an architecture's memory behaviour is not yet modelled.

    Deliberately an error rather than a fallback. A wrong KV estimate produces a
    config that OOMs or silently thrashes, which is worse than a refusal.
    """


@dataclass(frozen=True, slots=True)
class AttentionSpec:
    """Attention configuration, sufficient to size the KV cache."""

    kind: AttentionKind
    n_kv_heads: int
    head_dim: int
    n_kv_layers: int
    """Layers that hold a KV cache. Differs from total layers for hybrid models."""

    sliding_window_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.n_kv_heads < 1:
            raise ValueError(f"n_kv_heads must be >= 1, got {self.n_kv_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.n_kv_layers < 1:
            raise ValueError(f"n_kv_layers must be >= 1, got {self.n_kv_layers}")
        if self.sliding_window_tokens is not None and self.sliding_window_tokens < 1:
            raise ValueError(
                f"sliding_window_tokens must be >= 1 or None, got {self.sliding_window_tokens}"
            )
        if self.kind is AttentionKind.SLIDING_WINDOW and self.sliding_window_tokens is None:
            raise ValueError("SLIDING_WINDOW attention requires sliding_window_tokens")

    def kv_heads_per_gpu(self, tensor_parallel_size: int) -> int:
        """KV heads materialised on each rank under tensor parallelism.

        Note the ceiling. When ``n_kv_heads < tp`` the KV heads are *replicated*
        across ranks rather than split, so tensor parallelism stops reducing per-GPU
        KV cache. Dividing naively here understates memory and is a direct route to
        recommending a config that OOMs.
        """
        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")
        return max(1, math.ceil(self.n_kv_heads / tensor_parallel_size))

    def kv_replication_factor(self, tensor_parallel_size: int) -> float:
        """Aggregate KV storage multiplier caused by head replication.

        ``1.0`` means TP splits KV cleanly. ``2.0`` means the cluster stores each KV
        head twice, so half the aggregate KV memory is wasted.
        """
        per_gpu = self.kv_heads_per_gpu(tensor_parallel_size)
        return (per_gpu * tensor_parallel_size) / self.n_kv_heads

    def kv_bytes_per_token(self, kv_dtype: DType, tensor_parallel_size: int = 1) -> int:
        """Per-GPU KV cache bytes for one cached token.

        This is the load-bearing number in the whole system: the KV budget divided by
        this value gives the token capacity that the scheduler actually has to work
        with.

        Raises:
            UnsupportedArchitectureError: for architectures not yet modelled.
        """
        if self.kind in (AttentionKind.MHA, AttentionKind.GQA, AttentionKind.SLIDING_WINDOW):
            # K and V, per head, per layer. For SLIDING_WINDOW the per-token cost is
            # identical; the window caps how many tokens are retained per sequence,
            # which is the estimator's concern, not this function's.
            elements = (
                2 * self.kv_heads_per_gpu(tensor_parallel_size) * self.head_dim * self.n_kv_layers
            )
            return bytes_for(elements, kv_dtype)

        raise UnsupportedArchitectureError(
            f"KV sizing for {self.kind.value!r} attention is not implemented yet. "
            "MLA stores a compressed latent cache (the GQA formula overestimates it by "
            "roughly 10x), HYBRID retains full-length KV on only a subset of layers, and "
            "RECURRENT_HYBRID holds fixed state per sequence rather than per token. "
            "Modelling these is scheduled for M1; refusing rather than guessing."
        )


@dataclass(frozen=True, slots=True)
class MoESpec:
    """Mixture-of-experts configuration.

    Total parameters drive memory; active parameters drive compute. Conflating them
    is why MoE models are so often mis-sized.
    """

    n_experts: int
    n_experts_per_token: int
    n_shared_experts: int = 0

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            raise ValueError(f"n_experts must be >= 1, got {self.n_experts}")
        if not 1 <= self.n_experts_per_token <= self.n_experts:
            raise ValueError(
                f"n_experts_per_token must be in [1, {self.n_experts}], "
                f"got {self.n_experts_per_token}"
            )
        if self.n_shared_experts < 0:
            raise ValueError(f"n_shared_experts must be >= 0, got {self.n_shared_experts}")


@dataclass(frozen=True, slots=True)
class QuantSpec:
    """Weight quantization as found in a checkpoint.

    ``excluded_patterns`` matters more than it looks: quantized checkpoints routinely
    leave embeddings, norms, and ``lm_head`` at full precision, so
    ``params * bits / 8`` overstates compression. The estimator measures actual tensor
    bytes instead of trusting this, and uses this only to explain the result.
    """

    method: str
    """e.g. "awq", "gptq", "fp8", "compressed-tensors"."""

    weight_dtype: DType
    group_size: int | None = None
    excluded_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("quantization method must be non-empty")
        if self.group_size is not None and self.group_size < 1:
            raise ValueError(f"group_size must be >= 1 or None, got {self.group_size}")

    def effective_bits_per_weight(self) -> float:
        """Bits per quantized weight including scale/zero-point overhead.

        AWQ 4-bit at group size 128 costs about 4.15 bits per weight, not 4: each
        group of 128 weights also stores an fp16 scale and a 4-bit zero point.
        """
        if self.group_size is None:
            return float(self.weight_dtype.bits)
        scale_bits = 16 / self.group_size
        zero_point_bits = self.weight_dtype.bits / self.group_size
        return self.weight_dtype.bits + scale_bits + zero_point_bits


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Everything about a model that affects its deployment configuration."""

    model_id: str
    n_params_total: int
    n_params_active: int
    weight_bytes: int
    """Measured from checkpoint tensor metadata, not derived from parameter count."""

    n_layers: int
    hidden_size: int
    n_heads: int
    vocab_size: int
    max_position_embeddings: int
    attn: AttentionSpec
    weight_dtype: DType
    moe: MoESpec | None = None
    quant: QuantSpec | None = None
    tied_embeddings: bool = False
    weight_bytes_source: str = "unknown"
    """Provenance, e.g. "safetensors-header" or "estimated". Reported to the user."""

    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        for name in (
            "n_params_total",
            "n_params_active",
            "weight_bytes",
            "n_layers",
            "hidden_size",
            "n_heads",
            "vocab_size",
            "max_position_embeddings",
        ):
            value: int = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.n_params_active > self.n_params_total:
            raise ValueError(
                f"n_params_active ({self.n_params_active}) cannot exceed "
                f"n_params_total ({self.n_params_total})"
            )
        if self.attn.n_kv_layers > self.n_layers:
            raise ValueError(
                f"attn.n_kv_layers ({self.attn.n_kv_layers}) cannot exceed "
                f"n_layers ({self.n_layers})"
            )

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    def weight_bytes_per_gpu(self, tensor_parallel_size: int = 1) -> int:
        """Weight bytes resident on each rank under tensor parallelism.

        A deliberate first-order approximation: TP shards the bulk of the weights but
        replicates embeddings, norms, and small tensors, so real per-rank usage is
        slightly higher. M1 refines this from measured per-tensor metadata; this floor
        is honest about being a floor.
        """
        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")
        return -(-self.weight_bytes // tensor_parallel_size)
