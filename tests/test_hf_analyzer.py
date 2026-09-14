"""Architecture detection from real ``config.json`` files.

Hermetic: every config is vendored under ``tests/fixtures/configs``, so these tests need no
network and no model downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from infertune.core.cache import (
    AttentionKind,
    AttentionSpec,
    LayeredCacheSpec,
    MLASpec,
    RecurrentSpec,
)
from infertune.core.dtypes import DType
from infertune.models.hf import (
    UnknownArchitectureError,
    build_cache_spec,
    build_moe_spec,
    build_quant_spec,
    text_config,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "configs"


def load(name: str) -> dict[str, Any]:
    path = FIXTURES / f"{name}.json"
    if not path.is_file():
        pytest.skip(f"fixture {name} not vendored")
    config: dict[str, Any] = json.loads(path.read_text())
    return text_config(config)


def all_fixtures() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


class TestArchitectureDetection:
    def test_deepseek_v3_detected_as_mla(self) -> None:
        """Must win over num_key_value_heads, which DeepSeek also sets (to 128)."""
        cfg = load("deepseek-v3")
        assert cfg["num_key_value_heads"] == 128
        spec, warnings = build_cache_spec(cfg)
        assert isinstance(spec, MLASpec)
        assert spec.kv_lora_rank == 512
        assert spec.qk_rope_head_dim == 64
        assert spec.n_kv_layers == 61
        assert any("MLA detected" in w for w in warnings)

    def test_nemotron_h_detected_as_attention_mamba_hybrid(self) -> None:
        cfg = load("nemotron-h-8b")
        spec, warnings = build_cache_spec(cfg)
        assert isinstance(spec, LayeredCacheSpec)
        attention = [p for p in spec.parts if isinstance(p, AttentionSpec)]
        recurrent = [p for p in spec.parts if isinstance(p, RecurrentSpec)]
        assert len(attention) == 1
        assert attention[0].n_kv_layers == 4
        assert len(recurrent) == 1
        assert recurrent[0].n_recurrent_layers == 24
        assert any("only 4 of 52" in w for w in warnings)

    def test_gpt_oss_detected_as_interleaved_attention(self) -> None:
        cfg = load("gpt-oss-20b")
        spec, warnings = build_cache_spec(cfg)
        assert isinstance(spec, LayeredCacheSpec)
        windows = {p.sliding_window_tokens for p in spec.parts if isinstance(p, AttentionSpec)}
        assert windows == {None, 128}
        assert spec.n_cache_layers == 24
        assert any("sub-linear" in w for w in warnings)

    def test_qwen25_awq_honours_use_sliding_window_false(self) -> None:
        """The trap: sliding_window is 131072 but the flag disabling it is set.

        Honouring the window alone would model a cache cap the engine never applies.
        """
        cfg = load("qwen25-7b-awq")
        assert cfg["sliding_window"] == 131072
        assert cfg["use_sliding_window"] is False
        spec, warnings = build_cache_spec(cfg)
        assert isinstance(spec, AttentionSpec)
        assert spec.sliding_window_tokens is None
        assert spec.kind is AttentionKind.GQA
        assert any("use_sliding_window is false" in w for w in warnings)

    def test_plain_gqa_models(self) -> None:
        for name in ("qwen3-8b", "mistral-7b-v03"):
            spec, _ = build_cache_spec(load(name))
            assert isinstance(spec, AttentionSpec)
            assert spec.kind in (AttentionKind.GQA, AttentionKind.MHA)

    def test_head_dim_inferred_when_absent(self) -> None:
        cfg = load("mistral-7b-v03")
        assert "head_dim" not in cfg or cfg.get("head_dim") is None
        spec, _ = build_cache_spec(cfg)
        assert isinstance(spec, AttentionSpec)
        assert spec.head_dim == cfg["hidden_size"] // cfg["num_attention_heads"]

    @pytest.mark.parametrize("name", all_fixtures())
    def test_every_fixture_yields_a_usable_cache_spec(self, name: str) -> None:
        """No fixture may silently produce a nonsense cache model."""
        spec, _ = build_cache_spec(load(name))
        assert spec.n_cache_layers >= 1
        per_token = spec.marginal_bytes_per_token(DType.BF16)
        per_seq = spec.fixed_bytes_per_sequence(DType.BF16)
        assert per_token >= 0
        assert per_seq >= 0
        assert per_token + per_seq > 0, "a model must cache something"
        assert spec.sequence_bytes(4096, DType.BF16) > 0


class TestRefusals:
    def test_unknown_layer_types_refuse(self) -> None:
        cfg = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 8,
            "layer_types": ["full_attention", "quantum_attention"],
        }
        with pytest.raises(UnknownArchitectureError, match="unrecognised layer_types"):
            build_cache_spec(cfg)

    def test_sliding_layers_without_a_window_refuse(self) -> None:
        cfg = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 8,
            "layer_types": ["sliding_attention", "full_attention"],
        }
        with pytest.raises(UnknownArchitectureError, match="sliding_window is unset"):
            build_cache_spec(cfg)

    def test_mla_without_rope_dim_refuses(self) -> None:
        cfg = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "num_attention_heads": 8,
            "kv_lora_rank": 512,
        }
        with pytest.raises(UnknownArchitectureError, match="qk_rope_head_dim"):
            build_cache_spec(cfg)

    def test_mamba_without_state_size_refuses(self) -> None:
        """Refuse rather than guess a recurrent state size."""
        cfg = {
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "num_attention_heads": 8,
            "hybrid_override_pattern": "M-M*",
            "mamba_num_heads": 8,
            "mamba_head_dim": 8,
        }
        with pytest.raises(UnknownArchitectureError, match="refusing to guess"):
            build_cache_spec(cfg)

    def test_indivisible_hidden_size_refuses(self) -> None:
        cfg = {"num_hidden_layers": 1, "hidden_size": 100, "num_attention_heads": 7}
        with pytest.raises(UnknownArchitectureError, match="not divisible"):
            build_cache_spec(cfg)


class TestMoE:
    def test_deepseek_v3_moe(self) -> None:
        moe = build_moe_spec(load("deepseek-v3"))
        assert moe is not None
        assert moe.n_experts == 256
        assert moe.n_experts_per_token == 8
        assert moe.n_shared_experts == 1
        assert moe.n_dense_layers == 3

    def test_qwen3_moe_uses_num_experts(self) -> None:
        moe = build_moe_spec(load("qwen3-30b-a3b"))
        assert moe is not None
        assert (moe.n_experts, moe.n_experts_per_token) == (128, 8)

    def test_gpt_oss_uses_num_local_experts(self) -> None:
        moe = build_moe_spec(load("gpt-oss-20b"))
        assert moe is not None
        assert (moe.n_experts, moe.n_experts_per_token) == (32, 4)

    def test_dense_models_have_no_moe_spec(self) -> None:
        for name in ("qwen3-8b", "mistral-7b-v03"):
            assert build_moe_spec(load(name)) is None


class TestQuantisation:
    def test_awq_bits_and_group_size(self) -> None:
        quant = build_quant_spec(load("qwen25-7b-awq"))
        assert quant is not None
        assert quant.method == "awq"
        assert quant.weight_dtype is DType.INT4
        assert quant.group_size == 128
        # AWQ 4-bit at group 128 costs ~4.15 bits/weight, not 4.
        assert quant.effective_bits_per_weight() == pytest.approx(4.15, abs=0.01)

    def test_compressed_tensors_bits_read_from_config_groups(self) -> None:
        quant = build_quant_spec(load("llama31-8b-fp8"))
        assert quant is not None
        assert quant.weight_dtype is DType.INT8

    def test_mxfp4_excludes_embeddings_and_attention(self) -> None:
        """Confirms quantized checkpoints keep significant tensors at full precision."""
        quant = build_quant_spec(load("gpt-oss-20b"))
        assert quant is not None
        assert quant.method == "mxfp4"
        assert any("embed_tokens" in p for p in quant.excluded_patterns)
        assert any("lm_head" in p for p in quant.excluded_patterns)

    def test_unquantised_models_have_no_quant_spec(self) -> None:
        assert build_quant_spec(load("qwen3-8b")) is None
