"""GPU hardware description.

Deliberately a plain data structure with no detection logic, so it can be built either
from NVML on a live machine or from a spec database on a laptop. That is what makes
capacity planning ("what would I need to serve this?") a first-class mode rather than a
degraded one — and what keeps the analytical core testable without a GPU.

``source`` records which path produced the data, because a spec-sheet profile cannot
know about co-tenants or driver reservations and the report must say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from .dtypes import DType


class Interconnect(Enum):
    """Inter-GPU link, with an approximate unidirectional bandwidth in bytes/s.

    Tensor parallelism pays two costs per all-reduce: a bandwidth term and a fixed
    latency term. With two all-reduces per layer, a 64-layer model synchronises ~128
    times per decode step, so on PCIe the latency term alone can dominate. Modelling
    only bandwidth would make TP over PCIe look far cheaper than it is.
    """

    NONE = ("none", 0.0, 0.0)
    PCIE_GEN3_X16 = ("pcie-gen3-x16", 13e9, 12e-6)
    PCIE_GEN4_X16 = ("pcie-gen4-x16", 25e9, 10e-6)
    PCIE_GEN5_X16 = ("pcie-gen5-x16", 50e9, 9e-6)
    NVLINK_3 = ("nvlink-3", 300e9, 5e-6)
    NVLINK_4 = ("nvlink-4", 450e9, 4e-6)
    NVLINK_5 = ("nvlink-5", 900e9, 3e-6)

    def __init__(self, label: str, bytes_s: float, latency_s: float) -> None:
        self.label = label
        self.bytes_s = bytes_s
        self.latency_s = latency_s

    @property
    def is_pcie(self) -> bool:
        return self.label.startswith("pcie")

    def __str__(self) -> str:
        return self.label


@dataclass(frozen=True, slots=True)
class GPUProfile:
    """A GPU (or a homogeneous group of them) as far as configuration is concerned."""

    name: str
    vram_bytes: int
    """Per-device total, as the vendor quotes it (decimal GB)."""

    vram_usable_bytes: int
    """Per-device, after driver and display reservation. Always less than total."""

    compute_capability: tuple[int, int]
    sm_count: int
    mem_bandwidth_bytes_s: float
    dense_flops: dict[str, float]
    """Dense (non-sparse) peak FLOPS keyed by dtype label."""

    count: int = 1
    interconnect: Interconnect = Interconnect.NONE
    source: Literal["nvml", "specdb"] = "specdb"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.vram_bytes < 1:
            raise ValueError(f"vram_bytes must be >= 1, got {self.vram_bytes}")
        if not 0 < self.vram_usable_bytes <= self.vram_bytes:
            raise ValueError(
                f"vram_usable_bytes ({self.vram_usable_bytes}) must be in "
                f"(0, vram_bytes={self.vram_bytes}]"
            )
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        if self.sm_count < 1:
            raise ValueError(f"sm_count must be >= 1, got {self.sm_count}")
        if self.mem_bandwidth_bytes_s <= 0:
            raise ValueError(f"mem_bandwidth_bytes_s must be > 0, got {self.mem_bandwidth_bytes_s}")
        if not self.dense_flops:
            raise ValueError("dense_flops must contain at least one dtype")
        if any(v <= 0 for v in self.dense_flops.values()):
            raise ValueError("dense_flops values must all be > 0")
        if self.count > 1 and self.interconnect is Interconnect.NONE:
            raise ValueError("multi-GPU profiles must specify an interconnect")

    def flops(self, dtype: DType) -> float:
        """Dense peak FLOPS for ``dtype``.

        Raises:
            ValueError: if this GPU has no throughput figure for the dtype, which is
                also how unsupported dtypes are detected.
        """
        try:
            return self.dense_flops[dtype.label]
        except KeyError:
            known = ", ".join(sorted(self.dense_flops))
            raise ValueError(
                f"{self.name} has no dense FLOPS figure for {dtype.label!r}; known: {known}"
            ) from None

    def supports(self, dtype: DType) -> bool:
        return dtype.label in self.dense_flops

    def critical_batch_size(self, dtype: DType, *, mfu: float = 0.5, mbu: float = 0.8) -> float:
        """Batch size at which decode stops being bandwidth-bound.

        Below this, adding concurrency buys throughput almost for free; above it,
        concurrency mostly buys latency. Arguably the single most useful number the
        profiler can print, because it says whether the GPU or the KV budget is the
        thing standing in the way.
        """
        if not 0 < mfu <= 1:
            raise ValueError(f"mfu must be in (0, 1], got {mfu}")
        if not 0 < mbu <= 1:
            raise ValueError(f"mbu must be in (0, 1], got {mbu}")
        return (self.flops(dtype) * mfu) / (self.mem_bandwidth_bytes_s * mbu)
