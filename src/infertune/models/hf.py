"""Build a :class:`~infertune.core.model.ModelProfile` from a Hugging Face config.

Architecture detection is driven by fields verified against released checkpoints, in
priority order, because several of these fields co-occur and the wrong precedence produces
a badly wrong cache model:

1. ``kv_lora_rank`` -> MLA (DeepSeek-family). Must be checked *before* ``num_key_value_heads``,
   which DeepSeek-V3 also sets (to 128) even though MLA ignores it entirely.
2. ``hybrid_override_pattern`` -> attention/Mamba hybrid (Nemotron-H).
3. ``layer_types`` -> interleaved local/global attention (gpt-oss).
4. ``sliding_window`` **and** ``use_sliding_window`` -> uniform sliding window. Qwen2.5
   ships ``sliding_window: 131072`` with ``use_sliding_window: false``; honouring the first
   without the second mis-sizes the cache.
5. Otherwise MHA or GQA.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

from ..core.cache import (
    AttentionKind,
    AttentionSpec,
    CacheSpec,
    LayeredCacheSpec,
    MLASpec,
    RecurrentSpec,
)
from ..core.dtypes import DType
from ..core.model import ModelProfile, MoESpec, QuantSpec
from .safetensors import HF_ENDPOINT, ModelMetadataError, WeightMeasurement, measure_hub

Config = dict[str, Any]
"""A parsed Hugging Face ``config.json``."""

_TIMEOUT_S = 30.0

_TORCH_DTYPES: dict[str, DType] = {
    "float32": DType.FP32,
    "float16": DType.FP16,
    "bfloat16": DType.BF16,
    "float8_e4m3fn": DType.FP8_E4M3,
    "float8_e5m2": DType.FP8_E5M2,
}

_QUANT_DTYPES: dict[int, DType] = {4: DType.INT4, 8: DType.INT8}

_PACKED_QUANT_METHODS = frozenset({"awq", "gptq", "mxfp4", "awq_marlin", "gptq_marlin"})
"""Methods that pack sub-byte weights into wider storage elements."""


class UnknownArchitectureError(ModelMetadataError):
    """Raised when a config cannot be mapped onto a known cache model."""


def fetch_config(repo_id: str, *, revision: str = "main", token: str | None = None) -> Config:
    """Fetch and parse ``config.json`` for a Hub repository."""
    token = token or os.environ.get("HF_TOKEN")
    url = f"{HF_ENDPOINT}/{repo_id}/resolve/{revision}/config.json"
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ModelMetadataError(
                f"access to {repo_id} is restricted (HTTP {exc.code}). Set HF_TOKEN."
            ) from exc
        raise ModelMetadataError(f"HTTP {exc.code} fetching config for {repo_id}") from exc
    except urllib.error.URLError as exc:
        raise ModelMetadataError(f"network error fetching config: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ModelMetadataError(f"{repo_id}: malformed config.json: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelMetadataError(f"{repo_id}: config.json is not an object")
    return data


def text_config(config: Config) -> Config:
    """Return the language-model sub-config, unwrapping multimodal wrappers."""
    inner = config.get("text_config")
    if isinstance(inner, dict):
        merged = {**config, **inner}
        merged.pop("text_config", None)
        return merged
    return config


def _head_dim(cfg: Config) -> int:
    explicit = cfg.get("head_dim")
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    if heads < 1:
        raise UnknownArchitectureError(f"num_attention_heads must be >= 1, got {heads}")
    if hidden % heads:
        raise UnknownArchitectureError(
            f"cannot infer head_dim: hidden_size {hidden} is not divisible by "
            f"num_attention_heads {heads}, and no explicit head_dim is set"
        )
    return hidden // heads


def _kv_heads(cfg: Config) -> int:
    kv = cfg.get("num_key_value_heads")
    if isinstance(kv, int) and kv > 0:
        return kv
    return int(cfg["num_attention_heads"])


def _attention_kind(n_heads: int, n_kv_heads: int) -> AttentionKind:
    return AttentionKind.MHA if n_kv_heads >= n_heads else AttentionKind.GQA


def build_cache_spec(cfg: Config) -> tuple[CacheSpec, list[str]]:
    """Select and construct the cache model for a config.

    Returns the spec and any warnings worth surfacing to the user.
    """
    warnings: list[str] = []
    n_layers = int(cfg["num_hidden_layers"])
    head_dim = _head_dim(cfg)
    n_kv_heads = _kv_heads(cfg)
    kind = _attention_kind(int(cfg["num_attention_heads"]), n_kv_heads)

    # 1. MLA — checked first, because DeepSeek also sets num_key_value_heads.
    if isinstance(cfg.get("kv_lora_rank"), int):
        rope_dim = cfg.get("qk_rope_head_dim")
        if not isinstance(rope_dim, int):
            raise UnknownArchitectureError(
                "kv_lora_rank is set (MLA) but qk_rope_head_dim is missing"
            )
        warnings.append(
            "MLA detected: the cache is one compressed latent per token per layer, so "
            "num_key_value_heads is not used for sizing and tensor parallelism does not "
            "shard it"
        )
        return MLASpec(
            kv_lora_rank=int(cfg["kv_lora_rank"]),
            qk_rope_head_dim=rope_dim,
            n_kv_layers=n_layers,
        ), warnings

    # 2. Attention/Mamba hybrid.
    pattern = cfg.get("hybrid_override_pattern")
    if isinstance(pattern, str) and pattern:
        if len(pattern) != n_layers:
            warnings.append(
                f"hybrid_override_pattern length {len(pattern)} != num_hidden_layers "
                f"{n_layers}; trusting the pattern"
            )
        counts = Counter(pattern)
        n_attention = counts.get("*", 0)
        n_mamba = counts.get("M", 0)
        if n_attention == 0 and n_mamba == 0:
            raise UnknownArchitectureError(
                f"unrecognised hybrid_override_pattern {pattern!r}: expected '*' for "
                "attention and 'M' for Mamba layers"
            )
        parts: list[CacheSpec] = []
        if n_attention:
            parts.append(
                AttentionSpec(
                    kind=kind,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    n_kv_layers=n_attention,
                )
            )
        if n_mamba:
            parts.append(_recurrent_spec(cfg, n_mamba))
        warnings.append(
            f"attention/Mamba hybrid: only {n_attention} of {n_layers} layers hold a KV "
            f"cache, and {n_mamba} Mamba layers hold fixed state per sequence rather than "
            "per token"
        )
        return LayeredCacheSpec(tuple(parts)), warnings

    # 3. Interleaved local/global attention.
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        counts = Counter(str(t) for t in layer_types)
        n_full = counts.get("full_attention", 0)
        n_sliding = counts.get("sliding_attention", 0)
        unknown = set(counts) - {"full_attention", "sliding_attention"}
        if unknown:
            raise UnknownArchitectureError(
                f"unrecognised layer_types values {sorted(unknown)}; expected "
                "'full_attention' and/or 'sliding_attention'"
            )
        raw_window = cfg.get("sliding_window")
        window = raw_window if isinstance(raw_window, int) else None
        if n_sliding and window is None:
            raise UnknownArchitectureError(
                "layer_types declares sliding_attention layers but sliding_window is unset"
            )
        parts = []
        if n_full:
            parts.append(
                AttentionSpec(
                    kind=kind, n_kv_heads=n_kv_heads, head_dim=head_dim, n_kv_layers=n_full
                )
            )
        if n_sliding:
            parts.append(
                AttentionSpec(
                    kind=AttentionKind.SLIDING_WINDOW,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    n_kv_layers=n_sliding,
                    sliding_window_tokens=window,
                )
            )
            warnings.append(
                f"interleaved attention: {n_sliding} of {len(layer_types)} layers are capped "
                f"at a {window}-token window, so cache growth is sub-linear in context"
            )
        return LayeredCacheSpec(tuple(parts)), warnings

    # 4. Uniform sliding window — only when actually enabled.
    window = cfg.get("sliding_window")
    if isinstance(window, int) and window > 0:
        if cfg.get("use_sliding_window") is False:
            warnings.append(
                f"sliding_window is set to {window} but use_sliding_window is false; "
                "treating attention as full-context, which is what the engine will do"
            )
        else:
            return AttentionSpec(
                kind=AttentionKind.SLIDING_WINDOW,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                n_kv_layers=n_layers,
                sliding_window_tokens=window,
            ), warnings

    # 5. Plain MHA/GQA.
    return AttentionSpec(
        kind=kind, n_kv_heads=n_kv_heads, head_dim=head_dim, n_kv_layers=n_layers
    ), warnings


def _recurrent_spec(cfg: Config, n_mamba_layers: int) -> RecurrentSpec:
    n_mamba_heads = cfg.get("mamba_num_heads")
    mamba_head_dim = cfg.get("mamba_head_dim")
    if isinstance(n_mamba_heads, int) and isinstance(mamba_head_dim, int):
        d_inner = n_mamba_heads * mamba_head_dim
    else:
        raw_expand = cfg.get("mamba_expand", cfg.get("expand", 2))
        expand = raw_expand if isinstance(raw_expand, int) else 2
        d_inner = expand * int(cfg["hidden_size"])
    state = cfg.get("ssm_state_size", cfg.get("mamba_d_state", cfg.get("state_size")))
    conv = cfg.get("conv_kernel", cfg.get("mamba_d_conv", cfg.get("d_conv")))
    if not isinstance(state, int) or not isinstance(conv, int):
        raise UnknownArchitectureError(
            "Mamba layers detected but ssm_state_size/conv_kernel are missing; refusing to "
            "guess recurrent state size"
        )
    return RecurrentSpec(
        d_inner=d_inner,
        n_groups=int(cfg.get("n_groups", 1)),
        ssm_state_size=state,
        conv_kernel=conv,
        n_recurrent_layers=n_mamba_layers,
    )


def build_moe_spec(cfg: Config) -> MoESpec | None:
    """Extract MoE configuration, tolerating the several naming conventions in use."""
    n_experts = next(
        (
            cfg[key]
            for key in ("n_routed_experts", "num_experts", "num_local_experts")
            if isinstance(cfg.get(key), int) and cfg[key]
        ),
        None,
    )
    if n_experts is None:
        return None
    top_k = cfg.get("num_experts_per_tok", cfg.get("moe_topk"))
    if not isinstance(top_k, int) or top_k < 1:
        return None
    return MoESpec(
        n_experts=int(n_experts),
        n_experts_per_token=top_k,
        n_shared_experts=int(cfg.get("n_shared_experts") or 0),
        n_dense_layers=int(cfg.get("first_k_dense_replace") or 0),
    )


def build_quant_spec(cfg: Config) -> QuantSpec | None:
    """Extract quantization configuration across AWQ/GPTQ/fp8/compressed-tensors/mxfp4."""
    quant = cfg.get("quantization_config")
    if not isinstance(quant, dict):
        return None
    method = str(quant.get("quant_method", "unknown"))

    bits = quant.get("bits")
    if not isinstance(bits, int):
        groups = quant.get("config_groups")
        if isinstance(groups, dict):
            for group in groups.values():
                weights = group.get("weights") if isinstance(group, dict) else None
                if isinstance(weights, dict) and isinstance(weights.get("num_bits"), int):
                    bits = int(weights["num_bits"])
                    break
    if not isinstance(bits, int):
        bits = 4 if "4" in method else 8

    excluded = quant.get("modules_to_not_convert") or quant.get("ignore") or ()
    if isinstance(excluded, str):
        excluded = (excluded,)

    group_size = quant.get("group_size")
    if not isinstance(group_size, int) or group_size <= 0:
        group_size = None

    return QuantSpec(
        method=method,
        weight_dtype=_QUANT_DTYPES.get(bits, DType.INT8),
        group_size=group_size,
        excluded_patterns=tuple(str(p) for p in excluded),
    )


def _effective_intermediate(cfg: Config, moe: MoESpec | None) -> int | None:
    """MLP inner width activated per token.

    For MoE, each token is routed to ``n_experts_per_token`` experts plus any shared
    experts, so the activated width is the per-expert width times that count -- not the
    dense ``intermediate_size``.
    """
    if moe is not None:
        per_expert = cfg.get("moe_intermediate_size")
        if isinstance(per_expert, int) and per_expert > 0:
            return per_expert * (moe.n_experts_per_token + moe.n_shared_experts)
    value = cfg.get("intermediate_size")
    return value if isinstance(value, int) and value > 0 else None


def _weight_dtype(cfg: Config) -> DType:
    raw = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    return _TORCH_DTYPES.get(str(raw), DType.BF16)


def _estimate_active_params(cfg: Config, total: int, moe: MoESpec | None) -> tuple[int, bool]:
    """Estimate parameters activated per token.

    Exact for dense models. For MoE, subtracts total expert parameters and adds back only
    the activated ones. Within about 8% on DeepSeek-V3 (40B estimated vs 37B published),
    which is adequate for a compute-side roofline term and is flagged as an estimate.
    """
    if moe is None:
        return total, True

    n_layers = int(cfg["num_hidden_layers"])
    if n_layers < 1:
        return total, False
    hidden = int(cfg["hidden_size"])
    moe_intermediate = cfg.get("moe_intermediate_size")
    if not isinstance(moe_intermediate, int) or moe_intermediate <= 0:
        return total, False

    moe_layers = max(0, n_layers - moe.n_dense_layers)
    per_expert = 3 * hidden * moe_intermediate
    total_expert = moe_layers * moe.n_experts * per_expert
    active_expert = moe_layers * (moe.n_experts_per_token + moe.n_shared_experts) * per_expert

    non_expert = total - total_expert
    if non_expert <= 0:
        return total, False
    return max(1, min(total, non_expert + active_expert)), False


def profile_from_config(
    repo_id: str,
    config: Config,
    measurement: WeightMeasurement,
) -> ModelProfile:
    """Assemble a :class:`ModelProfile` from a config and a weight measurement."""
    cfg = text_config(config)
    for required in ("num_hidden_layers", "hidden_size", "num_attention_heads", "vocab_size"):
        if required not in cfg:
            raise UnknownArchitectureError(f"{repo_id}: config.json is missing {required!r}")

    cache, warnings = build_cache_spec(cfg)
    moe = build_moe_spec(cfg)
    quant = build_quant_spec(cfg)
    active, exact_active = _estimate_active_params(cfg, measurement.n_params, moe)

    warnings.extend(measurement.warnings)

    packed = quant is not None and quant.method.lower() in _PACKED_QUANT_METHODS
    if packed and quant is not None:
        # AWQ/GPTQ/mxfp4 pack several logical weights into one storage element (4-bit
        # values inside int32 or uint8 tensors), so counting elements does not count
        # parameters. Byte totals stay exact, which is what the memory ledger needs.
        warnings.append(
            f"{quant.method} packs sub-byte weights into wider storage elements, so the "
            "parameter count is a storage-element count and understates logical parameters; "
            "the byte total is unaffected and remains exact"
        )
    if moe is not None and not exact_active:
        if active == measurement.n_params:
            warnings.append(
                "MoE detected but active parameters could not be derived from this config "
                "(no usable moe_intermediate_size); reporting total as active, which "
                "overstates the compute-side roofline"
            )
        else:
            warnings.append(
                "active parameter count is estimated from expert arithmetic, not measured; "
                "it affects the compute-side roofline only, not the memory ledger"
            )
    if not measurement.exact_params:
        warnings.append("parameter count is approximate; byte total is exact")

    return ModelProfile(
        model_id=repo_id,
        n_params_total=measurement.n_params,
        n_params_active=active,
        weight_bytes=measurement.total_bytes,
        n_layers=int(cfg["num_hidden_layers"]),
        hidden_size=int(cfg["hidden_size"]),
        n_heads=int(cfg["num_attention_heads"]),
        vocab_size=int(cfg["vocab_size"]),
        max_position_embeddings=int(cfg.get("max_position_embeddings") or 4096),
        cache=cache,
        weight_dtype=_weight_dtype(cfg),
        moe=moe,
        quant=quant,
        tied_embeddings=bool(cfg.get("tie_word_embeddings")),
        intermediate_size=_effective_intermediate(cfg, moe),
        weight_bytes_source=measurement.source,
        warnings=tuple(warnings),
    )


def analyze(
    repo_id: str,
    *,
    revision: str = "main",
    token: str | None = None,
    max_workers: int = 16,
) -> ModelProfile:
    """Fetch a model's config and measure its weights, without downloading weights."""
    config = fetch_config(repo_id, revision=revision, token=token)
    tied = bool(text_config(config).get("tie_word_embeddings"))
    measurement = measure_hub(
        repo_id, revision=revision, tied_embeddings=tied, token=token, max_workers=max_workers
    )
    return profile_from_config(repo_id, config, measurement)
