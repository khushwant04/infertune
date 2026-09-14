"""KV cache sizing — the load-bearing arithmetic.

Anchored on Llama-3.1-8B, whose 128 KiB/token is a widely reproduced figure and
therefore a good golden value.
"""

from __future__ import annotations

import pytest

from infertune.core.dtypes import DType
from infertune.core.model import (
    AttentionKind,
    AttentionSpec,
    UnsupportedArchitectureError,
)
from infertune.core.units import GIB, KIB, gib

# Llama-3.1-8B: 32 layers, 8 KV heads (GQA), head_dim 128.
LLAMA_31_8B = AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=8, head_dim=128, n_kv_layers=32)


def test_llama_31_8b_is_128_kib_per_token() -> None:
    """2 (K,V) x 8 heads x 128 dim x 2 bytes x 32 layers = 131,072 B."""
    assert LLAMA_31_8B.kv_bytes_per_token(DType.BF16) == 131_072
    assert LLAMA_31_8B.kv_bytes_per_token(DType.BF16) == 128 * KIB


def test_fp8_kv_halves_the_cost() -> None:
    bf16 = LLAMA_31_8B.kv_bytes_per_token(DType.BF16)
    fp8 = LLAMA_31_8B.kv_bytes_per_token(DType.FP8_E4M3)
    assert fp8 == bf16 // 2 == 64 * KIB


def test_tensor_parallelism_splits_kv_heads() -> None:
    assert LLAMA_31_8B.kv_bytes_per_token(DType.BF16, tensor_parallel_size=2) == 64 * KIB
    assert LLAMA_31_8B.kv_bytes_per_token(DType.BF16, tensor_parallel_size=4) == 32 * KIB
    assert LLAMA_31_8B.kv_bytes_per_token(DType.BF16, tensor_parallel_size=8) == 16 * KIB


def test_kv_heads_replicate_when_tp_exceeds_head_count() -> None:
    """The subtlety that breaks naive division.

    With 8 KV heads and tp=16, heads are replicated rather than split: each rank still
    holds one head. Dividing 8/16 = 0.5 would understate per-GPU KV by 2x and produce a
    config that OOMs.
    """
    assert LLAMA_31_8B.kv_heads_per_gpu(16) == 1
    assert LLAMA_31_8B.kv_bytes_per_token(DType.BF16, tensor_parallel_size=16) == 16 * KIB
    # tp=8 and tp=16 cost the same per GPU; tp=16 buys nothing on KV.
    assert LLAMA_31_8B.kv_bytes_per_token(
        DType.BF16, tensor_parallel_size=16
    ) == LLAMA_31_8B.kv_bytes_per_token(DType.BF16, tensor_parallel_size=8)


def test_replication_factor_quantifies_the_waste() -> None:
    assert LLAMA_31_8B.kv_replication_factor(1) == pytest.approx(1.0)
    assert LLAMA_31_8B.kv_replication_factor(8) == pytest.approx(1.0)
    assert LLAMA_31_8B.kv_replication_factor(16) == pytest.approx(2.0)
    assert LLAMA_31_8B.kv_replication_factor(32) == pytest.approx(4.0)


def test_kv_heads_per_gpu_is_never_fractional_or_zero() -> None:
    spec = AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=6, head_dim=128, n_kv_layers=32)
    for tp in (1, 2, 3, 4, 5, 6, 7, 8, 16):
        heads = spec.kv_heads_per_gpu(tp)
        assert isinstance(heads, int)
        assert heads >= 1


def test_mha_costs_more_than_gqa_at_equal_size() -> None:
    """Two "7B models" can differ 4x in KV cost. This is why parameter count is not enough."""
    mha = AttentionSpec(kind=AttentionKind.MHA, n_kv_heads=32, head_dim=128, n_kv_layers=32)
    gqa = AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=8, head_dim=128, n_kv_layers=32)
    assert mha.kv_bytes_per_token(DType.BF16) == 4 * gqa.kv_bytes_per_token(DType.BF16)


def test_token_capacity_from_a_budget() -> None:
    """3.77 GiB at 128 KiB/token is ~30,900 tokens — the README's worked example."""
    per_token = LLAMA_31_8B.kv_bytes_per_token(DType.BF16)
    assert gib(3.77) // per_token == pytest.approx(30_900, abs=50)


def test_five_point_four_gib_is_only_five_full_8k_sequences() -> None:
    """The inconsistency that motivated the design: 5.4 GiB cannot hold 24 x 8K contexts."""
    per_token = LLAMA_31_8B.kv_bytes_per_token(DType.BF16)
    capacity = gib(5.4) // per_token
    assert capacity == pytest.approx(44_236, abs=10)
    assert capacity / 8192 == pytest.approx(5.4, abs=0.1)
    # 24 sequences at full 8K context would need 24 GiB, not 5.4.
    assert 24 * 8192 * per_token == 24 * GIB


def test_sliding_window_requires_a_window() -> None:
    with pytest.raises(ValueError, match="requires sliding_window_tokens"):
        AttentionSpec(kind=AttentionKind.SLIDING_WINDOW, n_kv_heads=8, head_dim=128, n_kv_layers=32)


def test_sliding_window_per_token_cost_matches_gqa() -> None:
    """The window caps retained tokens, not the per-token cost."""
    swa = AttentionSpec(
        kind=AttentionKind.SLIDING_WINDOW,
        n_kv_heads=8,
        head_dim=128,
        n_kv_layers=32,
        sliding_window_tokens=4096,
    )
    assert swa.kv_bytes_per_token(DType.BF16) == LLAMA_31_8B.kv_bytes_per_token(DType.BF16)


@pytest.mark.parametrize(
    "kind", [AttentionKind.MLA, AttentionKind.HYBRID, AttentionKind.RECURRENT_HYBRID]
)
def test_unmodelled_architectures_refuse_rather_than_guess(kind: AttentionKind) -> None:
    """Refusing beats a confidently wrong number that OOMs in production."""
    spec = AttentionSpec(kind=kind, n_kv_heads=8, head_dim=128, n_kv_layers=32)
    with pytest.raises(UnsupportedArchitectureError, match="not implemented yet"):
        spec.kv_bytes_per_token(DType.BF16)


def test_unsupported_architecture_error_is_a_notimplementederror() -> None:
    assert issubclass(UnsupportedArchitectureError, NotImplementedError)


@pytest.mark.parametrize("tp", [0, -1])
def test_invalid_tp_rejected(tp: int) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        LLAMA_31_8B.kv_bytes_per_token(DType.BF16, tensor_parallel_size=tp)


@pytest.mark.parametrize(
    ("heads", "dim", "layers"),
    [(0, 128, 32), (8, 0, 32), (8, 128, 0), (-1, 128, 32)],
)
def test_attention_spec_validates_dimensions(heads: int, dim: int, layers: int) -> None:
    with pytest.raises(ValueError):
        AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=heads, head_dim=dim, n_kv_layers=layers)


def test_sub_byte_kv_dtype_rounds_up_not_down() -> None:
    """Sub-byte types must not silently truncate to zero bytes."""
    tiny = AttentionSpec(kind=AttentionKind.MHA, n_kv_heads=1, head_dim=1, n_kv_layers=1)
    assert tiny.kv_bytes_per_token(DType.INT4) == 1  # 2 elements x 4 bits = 1 byte
