"""Model description: architecture, quantization, and measured weight size.

Cache cost models live in :mod:`infertune.core.cache` and are re-exported here for
convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cache import (
    AttentionKind,
    AttentionSpec,
    CacheSpec,
    LayeredCacheSpec,
    MLASpec,
    RecurrentSpec,
    UnsupportedArchitectureError,
)
from .dtypes import DType

__all__ = [
    "AttentionKind",
    "AttentionSpec",
    "CacheSpec",
    "LayeredCacheSpec",
    "MLASpec",
    "MoESpec",
    "ModelProfile",
    "QuantSpec",
    "RecurrentSpec",
    "UnsupportedArchitectureError",
]


@dataclass(frozen=True, slots=True)
class MoESpec:
    """Mixture-of-experts configuration.

    Total parameters drive **memory**; active parameters drive **compute**. Conflating the
    two is why MoE models are so often mis-sized: DeepSeek-V3 needs 1275 GiB of weights but
    only activates ~37B parameters per token.
    """

    n_experts: int
    n_experts_per_token: int
    n_shared_experts: int = 0
    n_dense_layers: int = 0
    """Leading layers using a normal MLP instead of MoE (DeepSeek ``first_k_dense_replace``)."""

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
        if self.n_dense_layers < 0:
            raise ValueError(f"n_dense_layers must be >= 0, got {self.n_dense_layers}")


@dataclass(frozen=True, slots=True)
class QuantSpec:
    """Weight quantization as declared by a checkpoint.

    ``excluded_patterns`` matters more than it looks. Quantized checkpoints routinely leave
    embeddings, norms, routers, and ``lm_head`` at full precision — gpt-oss-20b's mxfp4
    config explicitly excludes ``model.embed_tokens``, ``lm_head``, every ``self_attn``, and
    the MoE routers. So ``params * bits / 8`` badly overstates the compression achieved.

    The estimator therefore **measures** actual tensor bytes and uses this only to explain
    the result.
    """

    method: str
    """e.g. "awq", "gptq", "fp8", "compressed-tensors", "mxfp4"."""

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

        AWQ 4-bit at group size 128 costs about 4.15 bits per weight, not 4: each group of
        128 weights also stores an fp16 scale and a 4-bit zero point.
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
    """Resident weight bytes. Measured from checkpoint tensor metadata, with tied
    embeddings deduplicated — not derived from a parameter count."""

    n_layers: int
    hidden_size: int
    n_heads: int
    vocab_size: int
    max_position_embeddings: int
    cache: CacheSpec
    weight_dtype: DType
    moe: MoESpec | None = None
    quant: QuantSpec | None = None
    tied_embeddings: bool = False
    intermediate_size: int | None = None
    """MLP inner width, per token, after MoE routing.

    Drives prefill activation memory, which is dominated by the MLP intermediate rather
    than the residual stream: Qwen2.5-7B's intermediate is 5.3x its hidden size, and
    ignoring it under-estimated activations badly enough to miss the M2 accuracy bar.
    Falls back to 4x hidden_size when unknown.
    """

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
        if self.cache.n_cache_layers > self.n_layers:
            raise ValueError(
                f"cache.n_cache_layers ({self.cache.n_cache_layers}) cannot exceed "
                f"n_layers ({self.n_layers})"
            )

    @property
    def attn(self) -> CacheSpec:
        """Backwards-compatible alias for :attr:`cache`."""
        return self.cache

    @property
    def effective_intermediate_size(self) -> int:
        """MLP inner width per token, defaulting to the common 4x ratio."""
        if self.intermediate_size and self.intermediate_size > 0:
            return self.intermediate_size
        return 4 * self.hidden_size

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    @property
    def active_fraction(self) -> float:
        """Share of parameters used per token. 1.0 for dense models."""
        return self.n_params_active / self.n_params_total

    def weight_bytes_per_gpu(self, tensor_parallel_size: int = 1) -> int:
        """Weight bytes resident on each rank under tensor parallelism.

        A deliberate first-order approximation: TP shards the bulk of the weights but
        replicates embeddings, norms, and small tensors, so real per-rank usage is slightly
        higher. Reported as a floor, and the ledger says so.
        """
        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")
        return -(-self.weight_bytes // tensor_parallel_size)
