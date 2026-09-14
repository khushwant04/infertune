"""Per-architecture cache models.

Every expected value is derived from a released checkpoint's ``config.json``, and the
headline correction factors are asserted so a regression to the naive formula fails loudly.
"""

from __future__ import annotations

import pytest

from infertune.core.cache import (
    AttentionKind,
    AttentionSpec,
    LayeredCacheSpec,
    MLASpec,
    RecurrentSpec,
)
from infertune.core.dtypes import DType
from infertune.core.units import KIB, MIB

# DeepSeek-V3: kv_lora_rank 512, qk_rope_head_dim 64, 61 layers, 128 "kv heads".
DEEPSEEK_V3_MLA = MLASpec(kv_lora_rank=512, qk_rope_head_dim=64, n_kv_layers=61)

# Nemotron-H-8B: 52 layers = 4 attention + 24 Mamba + 24 MLP.
NEMOTRON_H_ATTENTION = AttentionSpec(
    kind=AttentionKind.GQA, n_kv_heads=8, head_dim=128, n_kv_layers=4
)
NEMOTRON_H_MAMBA = RecurrentSpec(
    d_inner=128 * 64, n_groups=8, ssm_state_size=128, conv_kernel=4, n_recurrent_layers=24
)

# gpt-oss-20b: 24 layers alternating sliding(128)/full, 8 kv heads, head_dim 64.
GPT_OSS_FULL = AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=8, head_dim=64, n_kv_layers=12)
GPT_OSS_SLIDING = AttentionSpec(
    kind=AttentionKind.SLIDING_WINDOW,
    n_kv_heads=8,
    head_dim=64,
    n_kv_layers=12,
    sliding_window_tokens=128,
)


class TestMLA:
    def test_latent_layout_is_576_elements_per_layer(self) -> None:
        """512 (compressed KV) + 64 (shared RoPE), with no 2x and no head multiplier."""
        per_layer = DEEPSEEK_V3_MLA.kv_bytes_per_token(DType.BF16) // 61
        assert per_layer == (512 + 64) * 2

    def test_deepseek_v3_is_68_kib_per_token(self) -> None:
        assert DEEPSEEK_V3_MLA.kv_bytes_per_token(DType.BF16) == 70_272
        assert DEEPSEEK_V3_MLA.kv_bytes_per_token(DType.BF16) / KIB == pytest.approx(68.6, abs=0.1)

    def test_gqa_formula_would_overestimate_by_57x(self) -> None:
        """The headline MLA correction, matching an independently published ~56x measurement.

        DeepSeek-V3 sets num_key_value_heads to 128, so a profiler that reaches for the GQA
        formula produces 3.81 MiB/token instead of 68.6 KiB/token.
        """
        naive = AttentionSpec(
            kind=AttentionKind.GQA, n_kv_heads=128, head_dim=128, n_kv_layers=61
        ).kv_bytes_per_token(DType.BF16)
        ratio = naive / DEEPSEEK_V3_MLA.kv_bytes_per_token(DType.BF16)
        assert ratio == pytest.approx(56.9, abs=0.5)

    def test_tensor_parallelism_does_not_shard_the_latent(self) -> None:
        """The latent is shared across heads, so TP cannot split it.

        Dividing by TP here would understate per-GPU memory — the direction that OOMs.
        """
        base = DEEPSEEK_V3_MLA.kv_bytes_per_token(DType.BF16)
        for tp in (1, 2, 4, 8, 16):
            assert DEEPSEEK_V3_MLA.kv_bytes_per_token(DType.BF16, tp) == base

    def test_fp8_mla_is_176x_not_2x(self) -> None:
        """vLLM's fp8_ds_mla keeps RoPE in bf16 and adds scales: 656 B/layer, not 576."""
        bf16 = DEEPSEEK_V3_MLA.kv_bytes_per_token(DType.BF16)
        fp8 = DEEPSEEK_V3_MLA.kv_bytes_per_token(DType.FP8_E4M3)
        assert fp8 // 61 == 512 + 2 * 64 + 16 == 656
        assert bf16 / fp8 == pytest.approx(1.76, abs=0.01)
        assert fp8 > bf16 // 2, "fp8 MLA must not be modelled as a naive halving"

    def test_scales_with_context(self) -> None:
        assert DEEPSEEK_V3_MLA.scales_with_context
        assert DEEPSEEK_V3_MLA.sequence_bytes(1000, DType.BF16) == 1000 * 70_272
        assert DEEPSEEK_V3_MLA.fixed_bytes_per_sequence(DType.BF16) == 0

    @pytest.mark.parametrize(("rank", "rope", "layers"), [(0, 64, 61), (512, 0, 61), (512, 64, 0)])
    def test_validation(self, rank: int, rope: int, layers: int) -> None:
        with pytest.raises(ValueError):
            MLASpec(kv_lora_rank=rank, qk_rope_head_dim=rope, n_kv_layers=layers)


class TestRecurrent:
    def test_nemotron_h_state_is_about_50_mib_per_sequence(self) -> None:
        """Verified against the published config: d_inner 8192, state 128, conv kernel 4."""
        state = NEMOTRON_H_MAMBA.fixed_bytes_per_sequence(DType.BF16)
        assert state / MIB == pytest.approx(49.88, abs=0.1)

    def test_state_does_not_grow_with_context(self) -> None:
        """The structural difference: max_num_seqs becomes a genuine reservation."""
        assert not NEMOTRON_H_MAMBA.scales_with_context
        assert NEMOTRON_H_MAMBA.marginal_bytes_per_token(DType.BF16) == 0
        short = NEMOTRON_H_MAMBA.sequence_bytes(128, DType.BF16)
        long = NEMOTRON_H_MAMBA.sequence_bytes(1_000_000, DType.BF16)
        assert short == long

    def test_state_is_sharded_by_tensor_parallelism(self) -> None:
        """Unlike the MLA latent, Mamba's inner dimension does split across ranks."""
        single = NEMOTRON_H_MAMBA.fixed_bytes_per_sequence(DType.BF16, 1)
        dual = NEMOTRON_H_MAMBA.fixed_bytes_per_sequence(DType.BF16, 2)
        assert dual < single
        assert dual == pytest.approx(single / 2, rel=0.02)

    def test_conv_and_ssm_terms_are_both_counted(self) -> None:
        d_inner, groups, state, conv = 8192, 8, 128, 4
        expected_elements = (d_inner + 2 * groups * state) * conv + d_inner * state
        spec = RecurrentSpec(
            d_inner=d_inner,
            n_groups=groups,
            ssm_state_size=state,
            conv_kernel=conv,
            n_recurrent_layers=1,
        )
        assert spec.fixed_bytes_per_sequence(DType.BF16) == expected_elements * 2


class TestLayered:
    def test_nemotron_h_only_4_of_52_layers_hold_kv(self) -> None:
        """13x correction: assuming all 52 layers cache KV is the naive failure."""
        hybrid = LayeredCacheSpec((NEMOTRON_H_ATTENTION, NEMOTRON_H_MAMBA))
        naive = AttentionSpec(
            kind=AttentionKind.GQA, n_kv_heads=8, head_dim=128, n_kv_layers=52
        ).kv_bytes_per_token(DType.BF16)
        assert naive / hybrid.marginal_bytes_per_token(DType.BF16) == pytest.approx(13.0, abs=0.1)

    def test_hybrid_combines_growing_and_fixed_terms(self) -> None:
        hybrid = LayeredCacheSpec((NEMOTRON_H_ATTENTION, NEMOTRON_H_MAMBA))
        assert hybrid.marginal_bytes_per_token(DType.BF16) == 16 * KIB
        assert hybrid.fixed_bytes_per_sequence(DType.BF16) / MIB == pytest.approx(49.88, abs=0.1)
        at_8k = hybrid.sequence_bytes(8192, DType.BF16)
        assert at_8k == hybrid.fixed_bytes_per_sequence(DType.BF16) + 8192 * 16 * KIB

    def test_hybrid_scales_with_context_if_any_part_does(self) -> None:
        assert LayeredCacheSpec((NEMOTRON_H_ATTENTION, NEMOTRON_H_MAMBA)).scales_with_context
        assert not LayeredCacheSpec((NEMOTRON_H_MAMBA,)).scales_with_context

    def test_sliding_layers_stop_growing_past_the_window(self) -> None:
        """gpt-oss caps 12 of 24 layers at 128 tokens, so growth is sub-linear."""
        interleaved = LayeredCacheSpec((GPT_OSS_FULL, GPT_OSS_SLIDING))
        uniform = AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=8, head_dim=64, n_kv_layers=24)
        at_8k_interleaved = interleaved.sequence_bytes(8192, DType.BF16)
        at_8k_uniform = uniform.sequence_bytes(8192, DType.BF16)
        assert at_8k_interleaved < at_8k_uniform
        assert at_8k_uniform / at_8k_interleaved == pytest.approx(1.97, abs=0.05)

    def test_marginal_cost_excludes_capped_layers(self) -> None:
        interleaved = LayeredCacheSpec((GPT_OSS_FULL, GPT_OSS_SLIDING))
        assert interleaved.marginal_bytes_per_token(DType.BF16) == GPT_OSS_FULL.kv_bytes_per_token(
            DType.BF16
        )

    def test_below_the_window_interleaved_matches_uniform(self) -> None:
        interleaved = LayeredCacheSpec((GPT_OSS_FULL, GPT_OSS_SLIDING))
        uniform = AttentionSpec(kind=AttentionKind.GQA, n_kv_heads=8, head_dim=64, n_kv_layers=24)
        assert interleaved.sequence_bytes(128, DType.BF16) == uniform.sequence_bytes(
            128, DType.BF16
        )

    def test_layer_counts_sum(self) -> None:
        assert LayeredCacheSpec((GPT_OSS_FULL, GPT_OSS_SLIDING)).n_cache_layers == 24

    def test_requires_at_least_one_part(self) -> None:
        with pytest.raises(ValueError, match="at least one part"):
            LayeredCacheSpec(())


def test_sliding_window_spec_caps_retained_tokens() -> None:
    spec = AttentionSpec(
        kind=AttentionKind.SLIDING_WINDOW,
        n_kv_heads=8,
        head_dim=128,
        n_kv_layers=32,
        sliding_window_tokens=4096,
    )
    per_token = spec.kv_bytes_per_token(DType.BF16)
    assert spec.sequence_bytes(1024, DType.BF16) == 1024 * per_token
    assert spec.sequence_bytes(100_000, DType.BF16) == 4096 * per_token
    assert spec.marginal_bytes_per_token(DType.BF16) == 0


def test_negative_sequence_length_rejected() -> None:
    for spec in (DEEPSEEK_V3_MLA, NEMOTRON_H_ATTENTION, NEMOTRON_H_MAMBA):
        with pytest.raises(ValueError, match="seq_len"):
            spec.sequence_bytes(-1, DType.BF16)
