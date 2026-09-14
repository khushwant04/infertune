"""Framework adapter abstraction.

The load-bearing idea: InferTune reasons about **semantic roles** (a KV budget, a context
limit, a concurrency cap), and each adapter translates roles into whatever flags the
*installed* engine version actually accepts. Adapters never carry a hardcoded flag table.

This is not defensive over-engineering; it is already required. `docs/plan.md` §2.2 argued
for compiling to vLLM's absolute `--kv-cache-memory` lever. Introspecting the vLLM the A10
forces us onto (0.19.1) shows that flag **does not exist** — it arrived later. A static table
would have emitted an unparseable command line. The role indirection lets the adapter detect
the absence and fall back to the fractional lever, reporting that it did so.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.plan import ResourcePlan


class ParamRole(Enum):
    """What a parameter *means*, independent of any engine's spelling."""

    KV_BUDGET_BYTES = "kv_budget_bytes"
    """Absolute KV cache size. Preferred: asserting bytes avoids predicting the engine's
    own memory measurement. Not available on every version."""

    MEMORY_FRACTION = "memory_fraction"
    """Fractional memory lever. Fallback, and semantically engine-specific: vLLM's covers
    weights + activations + KV; SGLang's covers weights + KV only."""

    KV_BUDGET_TOKENS = "kv_budget_tokens"
    MAX_CONTEXT = "max_context"
    MAX_SEQS = "max_seqs"
    PREFILL_TOKEN_BUDGET = "prefill_token_budget"
    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    DATA_PARALLEL = "data_parallel"
    EXPERT_PARALLEL = "expert_parallel"
    KV_DTYPE = "kv_dtype"
    WEIGHT_DTYPE = "weight_dtype"
    QUANTIZATION = "quantization"
    PREFIX_CACHING = "prefix_caching"
    EAGER = "eager"
    BLOCK_SIZE = "block_size"


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Something the user should know about a generated configuration."""

    severity: Severity
    message: str
    role: ParamRole | None = None

    def __str__(self) -> str:
        prefix = {Severity.INFO: "info", Severity.WARNING: "warning", Severity.ERROR: "error"}
        return f"{prefix[self.severity]}: {self.message}"


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One parameter as the installed engine actually declares it."""

    dest: str
    flags: tuple[str, ...]
    default: Any = None
    choices: tuple[str, ...] = ()
    type_hint: str = ""

    @property
    def primary_flag(self) -> str:
        """Longest long-form flag, which is the readable one."""
        long = [f for f in self.flags if f.startswith("--") and not f.startswith("--no-")]
        return max(long, key=len) if long else (self.flags[0] if self.flags else f"--{self.dest}")

    def negated_flag(self) -> str | None:
        """The ``--no-x`` form, for booleans that default to on."""
        return next((f for f in self.flags if f.startswith("--no-")), None)

    def supports(self, value: str) -> bool:
        return not self.choices or value in self.choices


@dataclass(frozen=True, slots=True)
class ParamSchema:
    """Every parameter the installed engine version accepts."""

    engine: str
    version: str
    params: dict[str, ParamSpec] = field(default_factory=dict)
    source: str = "introspected"

    def get(self, dest: str) -> ParamSpec | None:
        return self.params.get(dest)

    def has(self, dest: str) -> bool:
        return dest in self.params

    def resolve(self, candidates: tuple[str, ...]) -> ParamSpec | None:
        """First candidate destination that the engine actually declares."""
        for dest in candidates:
            spec = self.params.get(dest)
            if spec is not None:
                return spec
        return None


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What the installed engine version can do, derived from its own schema."""

    engine: str
    version: str
    absolute_kv_budget: bool
    """True when an exact-bytes KV lever exists, removing the need to invert the engine's
    own memory profiling."""

    kv_dtypes: tuple[str, ...] = ()
    weight_dtypes: tuple[str, ...] = ()
    quantizations: tuple[str, ...] = ()
    supports_expert_parallel: bool = False
    supports_prefix_caching: bool = False
    prefix_caching_default_on: bool = False


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """A concrete, runnable engine invocation."""

    engine: str
    version: str
    model: str
    args: tuple[tuple[str, str | None], ...]
    """Ordered (flag, value) pairs. ``None`` value means a bare boolean flag."""

    diagnostics: tuple[Diagnostic, ...] = ()
    notes: tuple[str, ...] = ()

    def command(self, executable: str | None = None) -> list[str]:
        exe = executable or self.engine
        cmd = [exe, "serve", self.model]
        for flag, value in self.args:
            cmd.append(flag)
            if value is not None:
                cmd.append(value)
        return cmd

    def command_line(self, executable: str | None = None) -> str:
        return " ".join(self.command(executable))

    def python_kwargs(self) -> dict[str, str | bool]:
        """Equivalent keyword arguments for the offline Python API."""
        out: dict[str, str | bool] = {}
        for flag, value in self.args:
            key = flag.lstrip("-").replace("-", "_")
            if value is None:
                if key.startswith("no_"):
                    out[key[3:]] = False
                else:
                    out[key] = True
            else:
                out[key] = value
        return out

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity is Severity.ERROR)

    @property
    def is_runnable(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class StartupFacts:
    """Measured facts parsed from an engine's own startup log.

    This is the ground truth the memory ledger is scored against, and it is free: one boot,
    no benchmarking. Fields are ``None`` when the engine version does not report them.
    """

    engine: str = ""
    version: str = ""
    kv_cache_bytes: int | None = None
    kv_cache_tokens: int | None = None
    weight_bytes: int | None = None
    cuda_graph_bytes: int | None = None
    cuda_graph_estimated_bytes: int | None = None
    max_concurrency: float | None = None
    context_per_request: int | None = None
    effective_args: dict[str, Any] = field(default_factory=dict)
    advisories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_kv_measurement(self) -> bool:
        return self.kv_cache_bytes is not None or self.kv_cache_tokens is not None

    def bytes_per_token(self) -> float | None:
        """Engine-observed KV bytes per token, a direct check on the cache model."""
        if self.kv_cache_bytes and self.kv_cache_tokens:
            return self.kv_cache_bytes / self.kv_cache_tokens
        return None


class FrameworkAdapter(ABC):
    """Translate a framework-independent plan into one engine's invocation."""

    name: str

    @abstractmethod
    def schema(self) -> ParamSchema:
        """Parameters the installed engine version accepts."""

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def compile(self, plan: ResourcePlan, model: str, **options: Any) -> LaunchSpec:
        """Render a plan as a concrete invocation for this engine."""

    @abstractmethod
    def parse_startup_log(self, text: str) -> StartupFacts:
        """Extract measured memory facts from the engine's own startup output."""

    def validate(self, spec: LaunchSpec) -> tuple[Diagnostic, ...]:
        """Check a spec against the installed schema.

        Default implementation checks that every flag is one the engine declares, which is
        what catches version drift before it becomes a failed boot.
        """
        schema = self.schema()
        known = {f for p in schema.params.values() for f in p.flags}
        out: list[Diagnostic] = []
        for flag, _ in spec.args:
            if flag not in known:
                out.append(
                    Diagnostic(
                        Severity.ERROR,
                        f"{flag!r} is not accepted by {schema.engine} {schema.version}",
                    )
                )
        return tuple(out)


__all__ = [
    "Capabilities",
    "Diagnostic",
    "FrameworkAdapter",
    "LaunchSpec",
    "ParamRole",
    "ParamSchema",
    "ParamSpec",
    "Severity",
    "StartupFacts",
]
