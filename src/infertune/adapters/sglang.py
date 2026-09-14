"""SGLang adapter.

This adapter is the acceptance test for the whole abstraction: if adding a second engine
required changes to the estimator, the cache models or the benchmark harness, the
framework-independent core would not really be framework-independent.

Three genuine differences from vLLM, none of which is cosmetic:

1. **SGLang has an absolute token lever.** ``--max-total-tokens`` sizes the KV pool directly,
   so no inversion of a memory fraction is needed. vLLM 0.19.1 has no equivalent, which is why
   :data:`infertune.adapters.vllm.ROLE_DESTS` leaves ``KV_BUDGET_TOKENS`` unmapped. The role
   exists in the enum precisely so the abstraction spans both.

2. **The memory fraction means something different.** ``--mem-fraction-static`` covers the
   *static* allocation — weights plus the KV pool — and leaves activations *outside* it. vLLM's
   ``--gpu-memory-utilization`` covers weights, activations and KV together. The same numeric
   value therefore describes different machines, which is the whole reason plans carry bytes
   rather than fractions.

3. **Polarity is inverted.** SGLang exposes ``--disable-radix-cache`` and
   ``--disable-cuda-graph`` where vLLM exposes ``--enable-prefix-caching`` and
   ``--enforce-eager``. A role means the same thing to a user; the flag that expresses it may
   be negated.

SGLang also moved its entire argument surface into ``sglang.srt.arg_groups.fields.*`` modules,
which is a live demonstration of why the schema is introspected rather than tabulated here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.dtypes import DType
from ..core.plan import ResourcePlan
from ..core.units import fmt_bytes
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

# Semantic role -> candidate SGLang destinations, in preference order.
ROLE_DESTS: dict[ParamRole, tuple[str, ...]] = {
    ParamRole.KV_BUDGET_TOKENS: ("max_total_tokens",),
    ParamRole.MEMORY_FRACTION: ("mem_fraction_static",),
    ParamRole.MAX_CONTEXT: ("context_length",),
    ParamRole.MAX_SEQS: ("max_running_requests",),
    ParamRole.PREFILL_TOKEN_BUDGET: ("chunked_prefill_size",),
    ParamRole.TENSOR_PARALLEL: ("tp_size",),
    ParamRole.PIPELINE_PARALLEL: ("pp_size",),
    ParamRole.DATA_PARALLEL: ("dp_size",),
    ParamRole.EXPERT_PARALLEL: ("ep_size",),
    ParamRole.KV_DTYPE: ("kv_cache_dtype",),
    ParamRole.WEIGHT_DTYPE: ("dtype",),
    ParamRole.QUANTIZATION: ("quantization",),
    ParamRole.PREFIX_CACHING: ("disable_radix_cache",),
    ParamRole.EAGER: ("disable_cuda_graph",),
    ParamRole.BLOCK_SIZE: ("page_size",),
}

NEGATED_ROLES = frozenset({ParamRole.PREFIX_CACHING, ParamRole.EAGER})
"""Roles SGLang expresses as ``--disable-x`` rather than ``--enable-x``."""

KV_DTYPE_PREFERENCE: dict[DType, tuple[str, ...]] = {
    DType.FP8_E4M3: ("fp8_e4m3",),
    DType.FP8_E5M2: ("fp8_e5m2",),
    DType.BF16: ("auto",),
    DType.FP16: ("auto",),
    DType.FP32: ("auto",),
}

MAX_MEM_FRACTION_STATIC = 0.95
"""SGLang's own troubleshooting guidance is to lower this on OOM; above ~0.95 it is unsafe."""


class SGLangSchemaError(RuntimeError):
    """Raised when the SGLang parameter schema cannot be obtained."""


def introspect_installed() -> ParamSchema:
    """Read the schema from the SGLang installed in this environment."""
    try:
        import argparse

        import sglang
        from sglang.srt.server_args import ServerArgs
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SGLangSchemaError(
            "SGLang is not installed here. Use a recorded schema dump via load_schema(), "
            "or install SGLang in this environment."
        ) from exc

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    params: dict[str, ParamSpec] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        choices = tuple(sorted(str(c) for c in action.choices)) if action.choices else ()
        params[action.dest] = ParamSpec(
            dest=action.dest,
            flags=tuple(sorted(action.option_strings)),
            default=action.default if isinstance(action.default, (int, float, str, bool)) else None,
            choices=choices,
            type_hint=getattr(action.type, "__name__", ""),
        )
    version = getattr(sglang, "__version__", "unknown")
    return ParamSchema(engine="sglang", version=str(version), params=params, source="introspected")


def load_schema(path: str | Path) -> ParamSchema:
    """Load a recorded schema dump, so adapter logic is testable without SGLang installed."""
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
            type_hint=str(fields.get(dest, {}).get("type", "")),
        )
    if not params:
        raise SGLangSchemaError(f"{path}: no cli_flags in schema dump")
    return ParamSchema(
        engine="sglang",
        version=str(data.get("sglang_version", "unknown")),
        params=params,
        source=f"recorded:{Path(path).name}",
    )


class SGLangAdapter(FrameworkAdapter):
    """Compile framework-independent plans into SGLang invocations."""

    name = "sglang"

    def __init__(self, schema: ParamSchema | None = None) -> None:
        self._schema = schema or introspect_installed()
        if self._schema.engine != "sglang":
            raise SGLangSchemaError(f"schema is for {self._schema.engine!r}, not sglang")

    def schema(self) -> ParamSchema:
        return self._schema

    def _choices(self, dest: str) -> tuple[str, ...]:
        spec = self._schema.get(dest)
        return tuple(spec.choices) if spec is not None else ()

    def capabilities(self) -> Capabilities:
        s = self._schema
        radix = s.get("disable_radix_cache")
        return Capabilities(
            engine="sglang",
            version=s.version,
            # A token-denominated pool is still an absolute lever: no inversion required.
            absolute_kv_budget=s.has("max_total_tokens"),
            kv_dtypes=self._choices("kv_cache_dtype"),
            weight_dtypes=self._choices("dtype"),
            quantizations=self._choices("quantization"),
            supports_expert_parallel=s.has("ep_size"),
            supports_prefix_caching=radix is not None,
            # RadixAttention is on unless explicitly disabled.
            prefix_caching_default_on=radix is not None,
        )

    def invert_memory_fraction(
        self,
        plan: ResourcePlan,
        total_vram_bytes: int,
        max_utilization: float = MAX_MEM_FRACTION_STATIC,
    ) -> tuple[float, Diagnostic]:
        """Solve for ``--mem-fraction-static`` yielding the plan's KV budget.

        Note the numerator differs from vLLM's. ``mem_fraction_static`` bounds the **static**
        allocation only — weights plus the KV pool — with activations living outside it:

            mem_fraction_static = (weights + kv_budget) / total

        Using vLLM's formula here would include activations and over-reserve; using SGLang's
        formula on vLLM would under-reserve and OOM. This is the concrete reason
        ``ResourcePlan`` carries bytes rather than a portable "memory fraction".
        """
        if total_vram_bytes <= 0:
            raise ValueError("total_vram_bytes must be > 0")
        needed = plan.kv_budget_bytes + plan.weight_bytes_per_gpu
        raw = needed / total_vram_bytes
        ceiling = min(MAX_MEM_FRACTION_STATIC, max_utilization)
        fraction = min(ceiling, max(0.05, round(raw, 4)))

        note = Diagnostic(
            Severity.INFO,
            f"--mem-fraction-static {fraction:.4f} covers weights + KV pool only "
            f"({fmt_bytes(needed)}); activations sit outside it, unlike vLLM's "
            "--gpu-memory-utilization",
            ParamRole.MEMORY_FRACTION,
        )
        if raw > ceiling:
            note = Diagnostic(
                Severity.WARNING,
                f"required mem-fraction-static {raw:.4f} exceeds the safe ceiling "
                f"{ceiling:.4f}; clamped. SGLang's own guidance on OOM is to lower this "
                "value, so reduce max_running_requests or context_length instead.",
                ParamRole.MEMORY_FRACTION,
            )
        return fraction, note

    def compile(
        self,
        plan: ResourcePlan,
        model: str,
        *,
        max_num_seqs: int | None = None,
        max_model_len: int | None = None,
        max_num_batched_tokens: int | None = None,
        total_vram_bytes: int | None = None,
        max_utilization: float = MAX_MEM_FRACTION_STATIC,
        enforce_eager: bool = False,
        prefer_absolute_kv: bool = True,
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
                        f"SGLang {s.version} has no parameter for {role.value}; skipped",
                        role,
                    )
                )
                return
            args.append((spec.primary_flag, value))

        # Prefer the absolute token lever; that is the point of §2.2.
        tokens_spec = s.resolve(ROLE_DESTS[ParamRole.KV_BUDGET_TOKENS])
        if prefer_absolute_kv and tokens_spec is not None:
            args.append((tokens_spec.primary_flag, str(plan.kv_budget_tokens)))
            notes.append(
                "KV pool sized in tokens via --max-total-tokens; no fraction inversion needed"
            )
        elif total_vram_bytes is None:
            diags.append(
                Diagnostic(
                    Severity.ERROR,
                    "without --max-total-tokens, --mem-fraction-static must be inverted, "
                    "which requires total_vram_bytes; pass it",
                    ParamRole.MEMORY_FRACTION,
                )
            )
        else:
            fraction, note = self.invert_memory_fraction(plan, total_vram_bytes, max_utilization)
            emit(ParamRole.MEMORY_FRACTION, f"{fraction:.4f}")
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
            emit(ParamRole.EXPERT_PARALLEL, str(max(1, p.tensor)))

        kv_spec = s.resolve(ROLE_DESTS[ParamRole.KV_DTYPE])
        if kv_spec is None:
            diags.append(
                Diagnostic(
                    Severity.WARNING,
                    f"SGLang {s.version} exposes no kv_cache_dtype",
                    ParamRole.KV_DTYPE,
                )
            )
        else:
            chosen = next(
                (
                    c
                    for c in KV_DTYPE_PREFERENCE.get(plan.dtypes.kv_cache, ())
                    if kv_spec.supports(c)
                ),
                None,
            )
            if chosen is None:
                diags.append(
                    Diagnostic(
                        Severity.ERROR,
                        f"SGLang {s.version} does not accept a KV dtype for "
                        f"{plan.dtypes.kv_cache}; available: "
                        f"{', '.join(kv_spec.choices) or 'unconstrained'}",
                        ParamRole.KV_DTYPE,
                    )
                )
            elif chosen != "auto":
                args.append((kv_spec.primary_flag, chosen))

        # Inverted polarity: eager mode is expressed by *disabling* CUDA graphs.
        if enforce_eager:
            spec = s.resolve(ROLE_DESTS[ParamRole.EAGER])
            if spec is not None:
                args.append((spec.primary_flag, None))
                notes.append(
                    "eager mode via --disable-cuda-graph (SGLang negates this role, "
                    "where vLLM uses --enforce-eager)"
                )

        draft = LaunchSpec(
            engine="sglang",
            version=s.version,
            model=model,
            args=tuple(args),
            diagnostics=tuple(diags),
            notes=tuple(notes),
        )
        return LaunchSpec(
            engine=draft.engine,
            version=draft.version,
            model=draft.model,
            args=draft.args,
            diagnostics=(*diags, *self.validate(draft)),
            notes=draft.notes,
        )

    def command(self, spec: LaunchSpec) -> list[str]:
        """SGLang launches via a module and ``--model-path``, not ``serve <model>``."""
        cmd = ["python", "-m", "sglang.launch_server", "--model-path", spec.model]
        for flag, value in spec.args:
            cmd.append(flag)
            if value is not None:
                cmd.append(value)
        return cmd

    def parse_startup_log(self, text: str) -> StartupFacts:
        """Extract measured facts from an SGLang startup log.

        SGLang reports ``max_total_num_tokens``, which is its KV pool capacity in tokens — the
        same quantity vLLM prints as ``GPU KV cache size``.
        """
        import re

        tokens = None
        if m := re.search(r"max_total_num_tokens\s*[=:]\s*([\d,]+)", text):
            tokens = int(m.group(1).replace(",", ""))
        chunked = None
        if m := re.search(r"chunked_prefill_size\s*[=:]\s*(-?[\d,]+)", text):
            chunked = int(m.group(1).replace(",", ""))
        weights = None
        if m := re.search(r"[Ww]eight.*?([\d.]+)\s*GB", text):
            weights = round(float(m.group(1)) * 1024**3)

        effective: dict[str, Any] = {}
        if chunked is not None:
            effective["chunked_prefill_size"] = chunked

        return StartupFacts(
            engine="sglang",
            version=self._schema.version,
            kv_cache_tokens=tokens,
            weight_bytes=weights,
            effective_args=effective,
        )


__all__ = [
    "KV_DTYPE_PREFERENCE",
    "MAX_MEM_FRACTION_STATIC",
    "NEGATED_ROLES",
    "ROLE_DESTS",
    "SGLangAdapter",
    "SGLangSchemaError",
    "introspect_installed",
    "load_schema",
]
