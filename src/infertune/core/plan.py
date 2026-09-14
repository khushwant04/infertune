"""The framework-independent output: a memory ledger and a resource plan.

Two commitments are encoded here.

First, the plan carries **bytes**, never a memory fraction. vLLM's
``--gpu-memory-utilization`` covers weights, activations and KV; SGLang's
``--mem-fraction-static`` covers weights and KV but not activations. The same fraction
describes different machines, so a portable "fraction" field would be a bug. Adapters
receive a byte budget and solve for their own knob.

Second, every plan names its :class:`BindingConstraint`. Without it the tool can state
a recommendation but cannot answer "why not higher?", which is the question users
actually have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .dtypes import DType
from .units import fmt_bytes


class BindingConstraint(Enum):
    """What stopped the recommendation from going further."""

    KV_WORKING_SET = "kv_working_set"
    WEIGHTS_DO_NOT_FIT = "weights_do_not_fit"
    MAX_MODEL_LEN = "max_model_len"
    SLA_TTFT = "sla_ttft"
    SLA_TPOT = "sla_tpot"
    COMPUTE_BOUND = "compute_bound"
    USER_LIMIT = "user_limit"

    def explain(self) -> str:
        """A one-line reason, suitable for printing next to the recommendation."""
        return _BINDING_EXPLANATIONS[self]


_BINDING_EXPLANATIONS: dict[BindingConstraint, str] = {
    BindingConstraint.KV_WORKING_SET: ("the KV cache budget, not compute, limits concurrency"),
    BindingConstraint.WEIGHTS_DO_NOT_FIT: (
        "model weights alone exceed usable VRAM; shard or quantize"
    ),
    BindingConstraint.MAX_MODEL_LEN: ("the requested context length caps what can be cached"),
    BindingConstraint.SLA_TTFT: "the time-to-first-token target is binding",
    BindingConstraint.SLA_TPOT: "the inter-token latency target is binding",
    BindingConstraint.COMPUTE_BOUND: (
        "the GPU is compute-saturated; more concurrency adds latency, not throughput"
    ),
    BindingConstraint.USER_LIMIT: "an explicit user-supplied limit is binding",
}


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One line of the memory ledger.

    ``formula`` and ``provenance`` are not decoration: a recommendation the user cannot
    audit is a recommendation they cannot safely deploy.
    """

    label: str
    bytes_: int
    formula: str = ""
    provenance: str = ""
    is_available: bool = False
    """True for capacity lines (total/usable VRAM), False for consumption lines."""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("ledger entry label must be non-empty")

    def __str__(self) -> str:
        return f"{self.label}: {fmt_bytes(self.bytes_)}"


@dataclass(frozen=True, slots=True)
class Parallelism:
    """Parallelism degrees, framework-independent."""

    tensor: int = 1
    pipeline: int = 1
    data: int = 1
    expert: bool = False

    def __post_init__(self) -> None:
        for name in ("tensor", "pipeline", "data"):
            value: int = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} parallel size must be >= 1, got {value}")

    @property
    def gpus_required(self) -> int:
        return self.tensor * self.pipeline * self.data


@dataclass(frozen=True, slots=True)
class DtypePlan:
    """Chosen precisions."""

    weights: DType
    activations: DType
    kv_cache: DType


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    """A complete, auditable memory plan for one GPU.

    ``kv_budget_bytes`` is the value adapters compile into engine flags, because both
    supported engines expose an absolute lever for it (vLLM ``--kv-cache-memory``,
    SGLang ``--max-total-tokens``). Asserting bytes avoids having to predict what the
    engine's own memory-profiling pass will measure.
    """

    weight_bytes_per_gpu: int
    activation_peak_bytes: int
    fixed_overhead_bytes: int
    safety_bytes: int
    kv_budget_bytes: int
    kv_bytes_per_token: int
    parallelism: Parallelism
    dtypes: DtypePlan
    binding_constraint: BindingConstraint
    ledger: tuple[LedgerEntry, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    headroom_concurrency: int | None = None
    """Largest concurrency whose p95 working set fits this budget.

    The actionable form of the binding constraint: it answers "how far can I raise
    concurrency?" rather than merely asserting that the cache is the limit.
    """

    critical_batch_size: float | None = None
    """Concurrency at which decode stops being bandwidth-bound.

    Compared against :attr:`headroom_concurrency` this says whether the cache or the GPU is
    the wall — and therefore whether fp8 KV would buy anything.
    """

    def __post_init__(self) -> None:
        for name in (
            "weight_bytes_per_gpu",
            "activation_peak_bytes",
            "fixed_overhead_bytes",
            "safety_bytes",
            "kv_budget_bytes",
        ):
            value: int = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.kv_bytes_per_token < 1:
            raise ValueError(f"kv_bytes_per_token must be >= 1, got {self.kv_bytes_per_token}")

    @property
    def kv_budget_tokens(self) -> int:
        """KV capacity in tokens — the number the scheduler actually lives within."""
        return self.kv_budget_bytes // self.kv_bytes_per_token

    @property
    def cache_limited(self) -> bool | None:
        """Whether the cache runs out before the GPU saturates.

        ``True`` means KV capacity, not compute, is the wall — so a smaller KV dtype or
        more GPUs would raise throughput, while a faster GPU would not. ``None`` when either
        figure is unavailable.
        """
        if self.headroom_concurrency is None or self.critical_batch_size is None:
            return None
        return self.headroom_concurrency < self.critical_batch_size

    @property
    def total_allocated_bytes(self) -> int:
        return (
            self.weight_bytes_per_gpu
            + self.activation_peak_bytes
            + self.fixed_overhead_bytes
            + self.safety_bytes
            + self.kv_budget_bytes
        )

    def reconciles_with(self, usable_bytes: int) -> bool:
        """Whether the ledger's consumption lines account for usable VRAM exactly.

        A ledger that does not reconcile has a missing or double-counted term, which is
        precisely the class of bug that produces boot-time OOM.
        """
        return self.total_allocated_bytes == usable_bytes
