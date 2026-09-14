"""Pure domain model for InferTune.

This package is **stdlib-only, by design**. No torch, no pydantic, no NVML, no HTTP.
That constraint is enforced by ``tests/test_core_purity.py`` and it buys two things:

1. The analytical engine is testable on CPU-only CI, with no GPU and no model downloads.
2. The tool can answer "what would I need to serve this model?" before any hardware is
   rented, because hardware facts arrive as a :class:`~infertune.core.gpu.GPUProfile`
   that may equally come from NVML or from a spec database.

Anything requiring I/O, hardware access, or third-party packages belongs in a layer
above this one.
"""

from __future__ import annotations

from .dtypes import DType, bytes_for
from .gpu import GPUProfile, Interconnect
from .model import (
    AttentionKind,
    AttentionSpec,
    ModelProfile,
    MoESpec,
    QuantSpec,
    UnsupportedArchitectureError,
)
from .plan import (
    BindingConstraint,
    DtypePlan,
    LedgerEntry,
    Parallelism,
    ResourcePlan,
)
from .units import (
    GB,
    GIB,
    MIB,
    fmt_bytes,
    fmt_tokens,
    gb,
    gib,
    mib,
    parse_size,
    to_gb,
    to_gib,
    vram_nameplate,
)
from .workload import (
    SLA,
    Constant,
    Distribution,
    Empirical,
    LogNormal,
    WorkingSet,
    WorkloadProfile,
)

__all__ = [
    "GB",
    "GIB",
    "MIB",
    "SLA",
    "AttentionKind",
    "AttentionSpec",
    "BindingConstraint",
    "Constant",
    "DType",
    "Distribution",
    "DtypePlan",
    "Empirical",
    "GPUProfile",
    "Interconnect",
    "LedgerEntry",
    "LogNormal",
    "MoESpec",
    "ModelProfile",
    "Parallelism",
    "QuantSpec",
    "ResourcePlan",
    "UnsupportedArchitectureError",
    "WorkingSet",
    "WorkloadProfile",
    "bytes_for",
    "fmt_bytes",
    "fmt_tokens",
    "gb",
    "gib",
    "mib",
    "parse_size",
    "to_gb",
    "to_gib",
    "vram_nameplate",
]
