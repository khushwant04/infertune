"""vLLM adapter.

The schema is read from the **installed** vLLM (or a recorded dump of one), never from a
table in this file. That decision earns its keep immediately: vLLM 0.19.1 — the newest
release that runs on an Azure A10, see ``docs/gpu-validation-a10.md`` — has **no**
``--kv-cache-memory`` flag, though current docs describe it. The adapter detects that and
falls back to inverting ``--gpu-memory-utilization``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from ..core.dtypes import DType
from ..core.plan import ResourcePlan
from ..core.units import GIB, MIB, fmt_bytes
from .base import (
    Capabilities,
    Diagnostic,
    FrameworkAdapter,
    LaunchSpec,
    ParamRole,
    ParamSchema,
    ParamSpec,
    Severity,
    StartupFacts,
)

# Semantic role -> candidate vLLM destinations, in preference order.
ROLE_DESTS: dict[ParamRole, tuple[str, ...]] = {
    ParamRole.KV_BUDGET_BYTES: ("kv_cache_memory",),
    ParamRole.MEMORY_FRACTION: ("gpu_memory_utilization",),
    ParamRole.MAX_CONTEXT: ("max_model_len",),
    ParamRole.MAX_SEQS: ("max_num_seqs",),
    ParamRole.PREFILL_TOKEN_BUDGET: ("max_num_batched_tokens",),
    ParamRole.TENSOR_PARALLEL: ("tensor_parallel_size",),
    ParamRole.PIPELINE_PARALLEL: ("pipeline_parallel_size",),
    ParamRole.DATA_PARALLEL: ("data_parallel_size",),
    ParamRole.EXPERT_PARALLEL: ("enable_expert_parallel",),
    ParamRole.KV_DTYPE: ("kv_cache_dtype",),
    ParamRole.WEIGHT_DTYPE: ("dtype",),
    ParamRole.QUANTIZATION: ("quantization",),
    ParamRole.PREFIX_CACHING: ("enable_prefix_caching",),
    ParamRole.EAGER: ("enforce_eager",),
    ParamRole.BLOCK_SIZE: ("block_size",),
}

# vLLM spells fp8 KV several ways depending on version; prefer the most explicit available.
#
# For 16-bit KV, prefer "auto" so vLLM derives the cache dtype from the model. That is not
# cosmetic: vLLM 0.19.1 *lists* "bfloat16" among kv_cache_dtype's choices, but passing it
# explicitly makes engine-core initialisation fail, while omitting the flag works. Only
# request an explicit dtype where it actually changes behaviour, i.e. fp8.
KV_DTYPE_PREFERENCE: dict[DType, tuple[str, ...]] = {
    DType.FP8_E4M3: ("fp8_e4m3", "fp8"),
    DType.FP8_E5M2: ("fp8_e5m2", "fp8"),
    DType.BF16: ("auto",),
    DType.FP16: ("auto",),
    DType.FP32: ("auto",),
}

MAX_GPU_MEMORY_UTILIZATION = 0.98
"""Above this, vLLM routinely fails during sampler warm-up."""


class VLLMSchemaError(RuntimeError):
    """Raised when the vLLM parameter schema cannot be obtained."""


def _coerce_default(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def introspect_installed() -> ParamSchema:
    """Read the parameter schema from the vLLM installed in this environment."""
    try:
        import argparse
        import dataclasses

        import vllm
        from vllm.engine.arg_utils import EngineArgs
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise VLLMSchemaError(
            "vLLM is not installed here. Use a recorded schema dump instead "
            "(ParamSchema via load_schema), or install vLLM in this environment."
        ) from exc

    params: dict[str, ParamSpec] = {}
    defaults: dict[str, Any] = {}
    types: dict[str, str] = {}
    if dataclasses.is_dataclass(EngineArgs):
        for f in dataclasses.fields(EngineArgs):
            d = f.default
            if d is dataclasses.MISSING:
                d = "<factory>" if f.default_factory is not dataclasses.MISSING else None
            defaults[f.name] = _coerce_default(d)
            types[f.name] = str(f.type)

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser())
    for action in parser._actions:
        if not action.option_strings:
            continue
        choices = tuple(sorted(str(c) for c in action.choices)) if action.choices else ()
        params[action.dest] = ParamSpec(
            dest=action.dest,
            flags=tuple(sorted(action.option_strings)),
            default=defaults.get(action.dest),
            choices=choices,
            type_hint=types.get(action.dest, ""),
        )
    return ParamSchema(
        engine="vllm", version=vllm.__version__, params=params, source="introspected"
    )


def load_schema(path: str | Path) -> ParamSchema:
    """Load a recorded schema dump, so adapter logic is testable without vLLM present."""
    data = json.loads(Path(path).read_text())
    fields = data.get("fields", {})
    choices = data.get("choices", {})
    params: dict[str, ParamSpec] = {}
    for dest, flags in data.get("cli_flags", {}).items():
        params[dest] = ParamSpec(
            dest=dest,
            flags=tuple(sorted(flags)),
            default=fields.get(dest, {}).get("default"),
            choices=tuple(choices.get(dest, ())),
            type_hint=fields.get(dest, {}).get("type", ""),
        )
    if not params:
        raise VLLMSchemaError(f"{path}: no cli_flags in schema dump")
    return ParamSchema(
        engine="vllm",
        version=str(data.get("vllm_version", "unknown")),
        params=params,
        source=f"recorded:{Path(path).name}",
    )


class VLLMAdapter(FrameworkAdapter):
    """Compile framework-independent plans into vLLM invocations."""

    name = "vllm"

    def __init__(self, schema: ParamSchema | None = None) -> None:
        self._schema = schema or introspect_installed()
        if self._schema.engine != "vllm":
            raise VLLMSchemaError(f"schema is for {self._schema.engine!r}, not vllm")

    def schema(self) -> ParamSchema:
        return self._schema

    def _choices(self, dest: str) -> tuple[str, ...]:
        spec = self._schema.get(dest)
        return tuple(spec.choices) if spec is not None else ()

    def capabilities(self) -> Capabilities:
        s = self._schema
        prefix = s.get("enable_prefix_caching")
        return Capabilities(
            engine="vllm",
            version=s.version,
            absolute_kv_budget=s.has("kv_cache_memory"),
            kv_dtypes=self._choices("kv_cache_dtype"),
            weight_dtypes=self._choices("dtype"),
            quantizations=self._choices("quantization"),
            supports_expert_parallel=s.has("enable_expert_parallel"),
            supports_prefix_caching=prefix is not None,
            # A None default means "decide at runtime", which in V1 means on where possible.
            prefix_caching_default_on=bool(prefix is not None and prefix.default is not False),
        )

    def _kv_dtype_value(self, dtype: DType) -> tuple[str | None, Diagnostic | None]:
        spec = self._schema.get("kv_cache_dtype")
        if spec is None:
            return None, Diagnostic(
                Severity.WARNING,
                f"vLLM {self._schema.version} exposes no kv_cache_dtype; KV precision cannot "
                "be set",
                ParamRole.KV_DTYPE,
            )
        for candidate in KV_DTYPE_PREFERENCE.get(dtype, ()):
            if spec.supports(candidate):
                return candidate, None
        return None, Diagnostic(
            Severity.ERROR,
            f"vLLM {self._schema.version} does not accept a KV dtype for {dtype}; "
            f"available: {', '.join(spec.choices) or 'unconstrained'}",
            ParamRole.KV_DTYPE,
        )

    def invert_memory_fraction(
        self,
        plan: ResourcePlan,
        total_vram_bytes: int,
        max_utilization: float = MAX_GPU_MEMORY_UTILIZATION,
    ) -> tuple[float, Diagnostic]:
        """Solve for the ``--gpu-memory-utilization`` that yields the plan's KV budget.

        vLLM sizes its cache as ``gmu * total_gpu_memory - (measured non-KV usage)``, so
        hitting a chosen KV budget means predicting that measurement:

            gmu = (kv_budget + weights + activations + fixed_overhead) / total

        The plan's safety margin is deliberately excluded: it becomes part of the
        ``(1 - gmu)`` slice vLLM already leaves untouched, and counting it twice would
        under-size the cache.

        This inversion is exactly what an absolute byte lever avoids, which is why it is
        used only when ``kv_cache_memory`` is absent.
        """
        if total_vram_bytes <= 0:
            raise ValueError("total_vram_bytes must be > 0")
        needed = (
            plan.kv_budget_bytes
            + plan.weight_bytes_per_gpu
            + plan.activation_peak_bytes
            + plan.fixed_overhead_bytes
        )
        raw = needed / total_vram_bytes
        ceiling = min(MAX_GPU_MEMORY_UTILIZATION, max_utilization)
        gmu = min(ceiling, max(0.05, round(raw, 4)))
        note = Diagnostic(
            Severity.INFO,
            f"vLLM {self._schema.version} has no --kv-cache-memory flag, so the "
            f"{fmt_bytes(plan.kv_budget_bytes)} KV budget was inverted into "
            f"--gpu-memory-utilization {gmu:.4f}. This depends on predicting vLLM's own "
            "memory measurement; compare against the KV figure in its startup log and "
            "recalibrate if they diverge.",
            ParamRole.MEMORY_FRACTION,
        )
        if raw > ceiling:
            note = Diagnostic(
                Severity.WARNING,
                f"required utilization {raw:.4f} exceeds this GPU's safe ceiling "
                f"{ceiling:.4f}; clamped. The KV budget will be smaller than "
                "planned. Reduce max_num_seqs or max_model_len, or shard further.",
                ParamRole.MEMORY_FRACTION,
            )
        return gmu, note

    def compile(
        self,
        plan: ResourcePlan,
        model: str,
        *,
        max_num_seqs: int | None = None,
        max_model_len: int | None = None,
        max_num_batched_tokens: int | None = None,
        total_vram_bytes: int | None = None,
        max_utilization: float = MAX_GPU_MEMORY_UTILIZATION,
        enforce_eager: bool = False,
        **_: Any,
    ) -> LaunchSpec:
        s = self._schema
        args: list[tuple[str, str | None]] = []
        diags: list[Diagnostic] = []
        notes: list[str] = []

        def emit(role: ParamRole, value: str | None) -> None:
            spec = s.resolve(ROLE_DESTS[role])
            if spec is None:
                diags.append(
                    Diagnostic(
                        Severity.WARNING,
                        f"vLLM {s.version} has no parameter for {role.value}; skipped",
                        role,
                    )
                )
                return
            args.append((spec.primary_flag, value))

        # KV budget: absolute lever when available, otherwise invert the fraction.
        kv_spec = s.resolve(ROLE_DESTS[ParamRole.KV_BUDGET_BYTES])
        if kv_spec is not None:
            args.append((kv_spec.primary_flag, str(plan.kv_budget_bytes)))
            notes.append("KV budget asserted in bytes; no inversion required")
        else:
            if total_vram_bytes is None:
                diags.append(
                    Diagnostic(
                        Severity.ERROR,
                        "this vLLM version needs --gpu-memory-utilization, which requires "
                        "total_vram_bytes to invert; pass it",
                        ParamRole.MEMORY_FRACTION,
                    )
                )
            else:
                gmu, note = self.invert_memory_fraction(plan, total_vram_bytes, max_utilization)
                emit(ParamRole.MEMORY_FRACTION, f"{gmu:.4f}")
                diags.append(note)

        if max_model_len is not None:
            emit(ParamRole.MAX_CONTEXT, str(max_model_len))
        if max_num_seqs is not None:
            emit(ParamRole.MAX_SEQS, str(max_num_seqs))
        if max_num_batched_tokens is not None:
            emit(ParamRole.PREFILL_TOKEN_BUDGET, str(max_num_batched_tokens))

        p = plan.parallelism
        if p.tensor > 1:
            emit(ParamRole.TENSOR_PARALLEL, str(p.tensor))
        if p.pipeline > 1:
            emit(ParamRole.PIPELINE_PARALLEL, str(p.pipeline))
        if p.data > 1:
            emit(ParamRole.DATA_PARALLEL, str(p.data))
        if p.expert:
            spec = s.resolve(ROLE_DESTS[ParamRole.EXPERT_PARALLEL])
            if spec is None:
                diags.append(
                    Diagnostic(
                        Severity.WARNING,
                        f"vLLM {s.version} has no expert-parallel flag; skipped",
                        ParamRole.EXPERT_PARALLEL,
                    )
                )
            else:
                args.append((spec.primary_flag, None))

        kv_value, kv_diag = self._kv_dtype_value(plan.dtypes.kv_cache)
        if kv_diag is not None:
            diags.append(kv_diag)
        if kv_value is not None and kv_value != "auto":
            emit(ParamRole.KV_DTYPE, kv_value)

        if enforce_eager:
            spec = s.resolve(ROLE_DESTS[ParamRole.EAGER])
            if spec is not None:
                args.append((spec.primary_flag, None))
                notes.append("eager mode: no CUDA graph pool, lower decode throughput")

        spec_out = LaunchSpec(
            engine="vllm",
            version=s.version,
            model=model,
            args=tuple(args),
            diagnostics=tuple(diags),
            notes=tuple(notes),
        )
        return LaunchSpec(
            engine=spec_out.engine,
            version=spec_out.version,
            model=spec_out.model,
            args=spec_out.args,
            diagnostics=(*diags, *self.validate(spec_out)),
            notes=spec_out.notes,
        )

    # ---------------------------------------------------------------- log parsing

    _RE_KV_BYTES = re.compile(
        r"Available KV cache memory:\s*([\d.]+)\s*(GiB|MiB|GB|MB)", re.IGNORECASE
    )
    _RE_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.IGNORECASE)
    _RE_WEIGHTS = re.compile(r"Model loading took\s*([\d.]+)\s*(GiB|MiB|GB|MB)", re.IGNORECASE)
    _RE_GRAPH_POOL = re.compile(
        r"CUDA graph pool memory:\s*([\d.]+)\s*GiB\s*\(actual\),\s*([\d.]+)\s*GiB\s*\(estimated\)",
        re.IGNORECASE,
    )
    _RE_GRAPH_TOOK = re.compile(
        r"Graph capturing finished in\s*\d+\s*secs?,\s*took\s*([\d.]+)\s*GiB", re.IGNORECASE
    )
    _RE_GRAPH_EST = re.compile(r"Estimated CUDA graph memory:\s*([\d.]+)\s*GiB", re.IGNORECASE)
    _RE_CONCURRENCY = re.compile(
        r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", re.IGNORECASE
    )
    _RE_NON_DEFAULT = re.compile(r"non-default args:\s*(\{.*?\})")
    _RE_ADVISORY = re.compile(r"(increase --gpu-memory-utilization from [\d.]+ to [\d.]+)")

    @staticmethod
    def _to_bytes(value: str, unit: str) -> int:
        scale = {"GIB": GIB, "MIB": MIB, "GB": 10**9, "MB": 10**6}[unit.upper()]
        return round(float(value) * scale)

    def parse_startup_log(self, text: str) -> StartupFacts:
        """Extract measured memory facts from a vLLM startup log.

        These are ground truth for scoring the memory ledger, and obtaining them costs one
        boot with no load generation.
        """
        kv_bytes = kv_tokens = weights = graph = graph_est = None
        concurrency = context = None

        if m := self._RE_KV_BYTES.search(text):
            kv_bytes = self._to_bytes(m.group(1), m.group(2))
        if m := self._RE_KV_TOKENS.search(text):
            kv_tokens = int(m.group(1).replace(",", ""))
        if m := self._RE_WEIGHTS.search(text):
            weights = self._to_bytes(m.group(1), m.group(2))
        if m := self._RE_GRAPH_POOL.search(text):
            graph = self._to_bytes(m.group(1), "GiB")
            graph_est = self._to_bytes(m.group(2), "GiB")
        else:
            if m := self._RE_GRAPH_TOOK.search(text):
                graph = self._to_bytes(m.group(1), "GiB")
            if m := self._RE_GRAPH_EST.search(text):
                graph_est = self._to_bytes(m.group(1), "GiB")
        if m := self._RE_CONCURRENCY.search(text):
            context = int(m.group(1).replace(",", ""))
            concurrency = float(m.group(2))

        effective: dict[str, Any] = {}
        if m := self._RE_NON_DEFAULT.search(text):
            try:
                parsed = ast.literal_eval(m.group(1))
                if isinstance(parsed, dict):
                    effective = parsed
            except (ValueError, SyntaxError):
                pass

        return StartupFacts(
            engine="vllm",
            version=self._schema.version,
            kv_cache_bytes=kv_bytes,
            kv_cache_tokens=kv_tokens,
            weight_bytes=weights,
            cuda_graph_bytes=graph,
            cuda_graph_estimated_bytes=graph_est,
            max_concurrency=concurrency,
            context_per_request=context,
            effective_args=effective,
            advisories=tuple(sorted(set(self._RE_ADVISORY.findall(text)))),
        )


__all__ = [
    "KV_DTYPE_PREFERENCE",
    "MAX_GPU_MEMORY_UTILIZATION",
    "ROLE_DESTS",
    "VLLMAdapter",
    "VLLMSchemaError",
    "introspect_installed",
    "load_schema",
]
