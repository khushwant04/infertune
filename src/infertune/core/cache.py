"""Per-architecture attention/state cache models.

The design commitment: **there is no single KV cache formula**. The familiar
``2 * layers * kv_heads * head_dim * bytes`` expression describes MHA and GQA only. Every
figure below was verified against the real ``config.json`` of a released checkpoint.

How wrong the naive formula gets, measured:

===================== ================================================================
architecture          error if the GQA formula is used
===================== ================================================================
MLA (DeepSeek-V3)     **~57x overestimate** — the cache is one 576-element latent per
                      token per layer, not 2 x 128 heads x 128 dim
Hybrid (Nemotron-H)   **~13x overestimate** — only 4 of 52 layers hold a KV cache
Hybrid (gpt-oss-20b)  ~2x at long context — 12 of 24 layers are capped at a 128-token
                      sliding window
Recurrent (Mamba2)    **structurally wrong** — state is fixed per *sequence*, so it
                      does not scale with context at all
===================== ================================================================

Consequently the primary abstraction is :meth:`CacheSpec.sequence_bytes`, the cache cost
of one sequence of a given length, rather than a per-token constant. A per-token constant
cannot express a sliding-window cap or a fixed recurrent state.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    """Mixed layer types; only some layers hold full-length KV."""

    RECURRENT = "recurrent"
    """Mamba/SSM blocks: fixed state per sequence, independent of context length."""


class UnsupportedArchitectureError(NotImplementedError):
    """Raised when an architecture's memory behaviour is not yet modelled.

    Deliberately an error rather than a fallback. A wrong cache estimate produces a
    config that OOMs or silently thrashes, which is worse than a refusal.
    """


class CacheSpec(ABC):
    """Cache cost model for one attention/state architecture."""

    @abstractmethod
    def sequence_bytes(self, seq_len: int, dtype: DType, tensor_parallel_size: int = 1) -> int:
        """Per-GPU cache bytes held by a single sequence of ``seq_len`` tokens.

        The primary abstraction, because it is the only one that survives sliding
        windows (sub-linear in ``seq_len``) and recurrent state (constant in ``seq_len``).
        """

    @abstractmethod
    def marginal_bytes_per_token(self, dtype: DType, tensor_parallel_size: int = 1) -> int:
        """Cache bytes added by one more token, once past any window boundary.

        Zero for purely recurrent models. For sliding-window layers this excludes the
        capped layers, since past the window they stop growing.
        """

    def fixed_bytes_per_sequence(self, dtype: DType, tensor_parallel_size: int = 1) -> int:
        """Cache bytes a sequence occupies regardless of its length."""
        return 0

    @property
    @abstractmethod
    def n_cache_layers(self) -> int:
        """Layers that hold any cache at all."""

    @property
    def scales_with_context(self) -> bool:
        """Whether cache grows with context length.

        ``False`` for purely recurrent models, where ``max_num_seqs`` becomes a true
        reservation and the §2.1 working-set argument inverts.
        """
        return True


def _check_tp(tensor_parallel_size: int) -> None:
    if tensor_parallel_size < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")


@dataclass(frozen=True, slots=True)
class AttentionSpec(CacheSpec):
    """MHA, GQA, or uniform sliding-window attention."""

    kind: AttentionKind
    n_kv_heads: int
    head_dim: int
    n_kv_layers: int
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
        if self.kind in (AttentionKind.MLA, AttentionKind.RECURRENT, AttentionKind.HYBRID):
            raise UnsupportedArchitectureError(
                f"{self.kind.value!r} cannot be modelled by AttentionSpec. Use MLASpec for "
                "MLA (DeepSeek-family), RecurrentSpec for Mamba/SSM state, or LayeredCacheSpec "
                "for mixed layer types. Using the GQA formula for MLA overestimates the cache "
                "by roughly 57x."
            )

    @property
    def n_cache_layers(self) -> int:
        return self.n_kv_layers

    def kv_heads_per_gpu(self, tensor_parallel_size: int) -> int:
        """KV heads materialised on each rank under tensor parallelism.

        Note the ceiling. When ``n_kv_heads < tp`` the KV heads are *replicated* across
        ranks rather than split, so tensor parallelism stops reducing per-GPU KV. Dividing
        naively understates memory and is a direct route to recommending a config that OOMs.
        """
        _check_tp(tensor_parallel_size)
        return max(1, math.ceil(self.n_kv_heads / tensor_parallel_size))

    def kv_replication_factor(self, tensor_parallel_size: int) -> float:
        """Aggregate KV storage multiplier caused by head replication.

        ``1.0`` means TP splits KV cleanly; ``2.0`` means the cluster stores each KV head
        twice, so half the aggregate KV memory is wasted.
        """
        per_gpu = self.kv_heads_per_gpu(tensor_parallel_size)
        return (per_gpu * tensor_parallel_size) / self.n_kv_heads

    def kv_bytes_per_token(self, kv_dtype: DType, tensor_parallel_size: int = 1) -> int:
        """Per-GPU KV bytes for one cached token, across all cache layers."""
        elements = (
            2  # K and V
            * self.kv_heads_per_gpu(tensor_parallel_size)
            * self.head_dim
            * self.n_kv_layers
        )
        return bytes_for(elements, kv_dtype)

    def marginal_bytes_per_token(self, dtype: DType, tensor_parallel_size: int = 1) -> int:
        if self.sliding_window_tokens is not None:
            # Past the window every layer stops growing.
            return 0
        return self.kv_bytes_per_token(dtype, tensor_parallel_size)

    def sequence_bytes(self, seq_len: int, dtype: DType, tensor_parallel_size: int = 1) -> int:
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")
        retained = seq_len
        if self.sliding_window_tokens is not None:
            retained = min(seq_len, self.sliding_window_tokens)
        return retained * self.kv_bytes_per_token(dtype, tensor_parallel_size)


@dataclass(frozen=True, slots=True)
class MLASpec(CacheSpec):
    """Multi-head latent attention (DeepSeek-family).

    MLA caches a single compressed latent vector per token per layer — ``kv_lora_rank``
    elements — plus a small shared RoPE vector of ``qk_rope_head_dim`` elements. Crucially
    there is **no factor of two** (no separate K and V) and **no multiplication by head
    count**. For DeepSeek-V3 that is 512 + 64 = 576 elements per token per layer, against
    the 2 x 128 x 128 = 32,768 the GQA formula would assume: a 57x difference.

    Two further subtleties:

    * The latent is **not sharded by tensor parallelism**, because it is shared across
      heads rather than partitioned between them. Dividing by TP here would understate
      per-GPU memory.
    * fp8 MLA is **not** simply half of bf16 MLA. vLLM's ``fp8_ds_mla`` layout stores the
      latent in fp8 but keeps the RoPE part in bf16 and adds per-token scales, giving 656
      bytes per token per layer rather than the naive 576 — a 1.76x saving, not 2x.
    """

    kv_lora_rank: int
    qk_rope_head_dim: int
    n_kv_layers: int

    def __post_init__(self) -> None:
        if self.kv_lora_rank < 1:
            raise ValueError(f"kv_lora_rank must be >= 1, got {self.kv_lora_rank}")
        if self.qk_rope_head_dim < 1:
            raise ValueError(f"qk_rope_head_dim must be >= 1, got {self.qk_rope_head_dim}")
        if self.n_kv_layers < 1:
            raise ValueError(f"n_kv_layers must be >= 1, got {self.n_kv_layers}")

    @property
    def n_cache_layers(self) -> int:
        return self.n_kv_layers

    def kv_bytes_per_token(self, kv_dtype: DType, tensor_parallel_size: int = 1) -> int:
        """Per-GPU MLA cache bytes for one token, across all layers.

        ``tensor_parallel_size`` is accepted and deliberately ignored: the latent cache is
        replicated on every rank.
        """
        _check_tp(tensor_parallel_size)
        if kv_dtype.is_float8:
            # vLLM fp8_ds_mla: fp8 latent + bf16 RoPE + 16 bytes of scales per token.
            per_layer = self.kv_lora_rank + 2 * self.qk_rope_head_dim + 16
        else:
            per_layer = bytes_for(self.kv_lora_rank + self.qk_rope_head_dim, kv_dtype)
        return per_layer * self.n_kv_layers

    def marginal_bytes_per_token(self, dtype: DType, tensor_parallel_size: int = 1) -> int:
        return self.kv_bytes_per_token(dtype, tensor_parallel_size)

    def sequence_bytes(self, seq_len: int, dtype: DType, tensor_parallel_size: int = 1) -> int:
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")
        return seq_len * self.kv_bytes_per_token(dtype, tensor_parallel_size)


@dataclass(frozen=True, slots=True)
class RecurrentSpec(CacheSpec):
    """Mamba2/SSM recurrent state.

    Structurally different from attention: state is **fixed per sequence** and does not
    grow with context. So for these models ``max_num_seqs`` genuinely *is* a reservation,
    and the working-set argument that applies to paged KV inverts.

    Per Mamba2 layer, per sequence:

    * convolution state: ``(d_inner + 2 * n_groups * ssm_state_size) * conv_kernel``
    * SSM state: ``d_inner * ssm_state_size``

    For Nemotron-H-8B (d_inner 8192, n_groups 8, ssm_state_size 128, conv_kernel 4) that
    is about 2.08 MiB per sequence per layer at bf16 — roughly 50 MiB per sequence across
    its 24 Mamba layers, which dwarfs the KV cache of its 4 attention layers.
    """

    d_inner: int
    n_groups: int
    ssm_state_size: int
    conv_kernel: int
    n_recurrent_layers: int

    def __post_init__(self) -> None:
        for name in ("d_inner", "n_groups", "ssm_state_size", "conv_kernel", "n_recurrent_layers"):
            value: int = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")

    @property
    def n_cache_layers(self) -> int:
        return self.n_recurrent_layers

    @property
    def scales_with_context(self) -> bool:
        return False

    def _elements_per_sequence_per_layer(self, tensor_parallel_size: int) -> int:
        # Mamba inner dimension is sharded by TP, unlike the MLA latent.
        d_inner = math.ceil(self.d_inner / tensor_parallel_size)
        groups = max(1, math.ceil(self.n_groups / tensor_parallel_size))
        conv_channels = d_inner + 2 * groups * self.ssm_state_size
        conv_state = conv_channels * self.conv_kernel
        ssm_state = d_inner * self.ssm_state_size
        return conv_state + ssm_state

    def fixed_bytes_per_sequence(self, dtype: DType, tensor_parallel_size: int = 1) -> int:
        _check_tp(tensor_parallel_size)
        per_layer = self._elements_per_sequence_per_layer(tensor_parallel_size)
        return bytes_for(per_layer * self.n_recurrent_layers, dtype)

    def marginal_bytes_per_token(self, dtype: DType, tensor_parallel_size: int = 1) -> int:
        return 0

    def sequence_bytes(self, seq_len: int, dtype: DType, tensor_parallel_size: int = 1) -> int:
        if seq_len < 0:
            raise ValueError(f"seq_len must be >= 0, got {seq_len}")
        return self.fixed_bytes_per_sequence(dtype, tensor_parallel_size)


@dataclass(frozen=True, slots=True)
class LayeredCacheSpec(CacheSpec):
    """A model whose layers do not all cache the same way.

    Covers two distinct real patterns:

    * **Interleaved local/global attention** — gpt-oss-20b alternates 12 sliding-window
      layers (window 128) with 12 full-attention layers. At 8K context the sliding layers
      hold 128 tokens each instead of 8192, so the naive uniform estimate is ~2x high.
    * **Attention/Mamba hybrids** — Nemotron-H-8B is 52 layers of which only 4 are
      attention (``hybrid_override_pattern`` ``M-M-M-M*-...``). Assuming all 52 layers hold
      KV overestimates by ~13x.

    Composition rather than special-casing, so a new pattern is a new list of parts, not a
    new branch.
    """

    parts: tuple[CacheSpec, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("LayeredCacheSpec requires at least one part")

    @property
    def n_cache_layers(self) -> int:
        return sum(p.n_cache_layers for p in self.parts)

    @property
    def scales_with_context(self) -> bool:
        return any(p.scales_with_context for p in self.parts)

    def fixed_bytes_per_sequence(self, dtype: DType, tensor_parallel_size: int = 1) -> int:
        return sum(p.fixed_bytes_per_sequence(dtype, tensor_parallel_size) for p in self.parts)

    def marginal_bytes_per_token(self, dtype: DType, tensor_parallel_size: int = 1) -> int:
        return sum(p.marginal_bytes_per_token(dtype, tensor_parallel_size) for p in self.parts)

    def sequence_bytes(self, seq_len: int, dtype: DType, tensor_parallel_size: int = 1) -> int:
        return sum(p.sequence_bytes(seq_len, dtype, tensor_parallel_size) for p in self.parts)
