"""Live GPU discovery via NVML.

Kept deliberately thin, and strictly optional. The rest of the system consumes a
:class:`~infertune.core.gpu.GPUProfile`, so a measured profile and a spec-sheet profile are
interchangeable everywhere downstream.

NVML supplies capacity, SM count and compute capability, but not peak FLOPS or memory
bandwidth — those come from the spec database, matched on device name. When the name is
unrecognised the profile is still built, with FLOPS derived from the compute capability
family and a warning attached, because a missing throughput figure should not prevent
memory analysis.
"""

from __future__ import annotations

import contextlib
from typing import Any

from ..core.gpu import GPUProfile, Interconnect
from . import specdb

# Conservative dense bf16 throughput by compute-capability major version, used only when a
# device is absent from the spec database.
_FALLBACK_FLOPS: dict[int, dict[str, float]] = {
    7: {"fp16": 100e12},
    8: {"fp16": 150e12, "bf16": 150e12},
    9: {"fp16": 700e12, "bf16": 700e12, "fp8_e4m3": 1400e12},
    10: {"bf16": 1800e12, "fp8_e4m3": 3600e12},
    12: {"bf16": 100e12, "fp8_e4m3": 200e12},
}


class NVMLUnavailableError(RuntimeError):
    """Raised when NVML cannot be loaded or no device is visible."""


def _pcie_generation_link(generation: int) -> Interconnect:
    if generation >= 5:
        return Interconnect.PCIE_GEN5_X16
    if generation == 4:
        return Interconnect.PCIE_GEN4_X16
    return Interconnect.PCIE_GEN3_X16


_NVLINK_BY_VERSION = {
    2: Interconnect.NVLINK_3,
    3: Interconnect.NVLINK_3,
    4: Interconnect.NVLINK_4,
}


def _detect_interconnect(pynvml: Any, handle: Any) -> tuple[Interconnect, list[str]]:
    """Determine the inter-GPU link from the live device.

    This must be measured rather than inherited from the spec database: a spec entry
    describes a *device*, and the same device ships in NVLink-bridged and
    PCIe-only machines. Reading it off the spec entry is what previously made
    discovery fail outright on multi-GPU hosts.
    """
    notes: list[str] = []

    active = 0
    version = 0
    for link in range(18):
        try:
            state = pynvml.nvmlDeviceGetNvLinkState(handle, link)
        except Exception:  # unsupported link index, or no NVLink on this device at all
            break
        if not state:
            continue
        active += 1
        with contextlib.suppress(Exception):
            version = max(version, int(pynvml.nvmlDeviceGetNvLinkVersion(handle, link)))
    if active:
        link_type = _NVLINK_BY_VERSION.get(version, Interconnect.NVLINK_5)
        notes.append(f"{active} active NVLink(s), version {version or 'unknown'}")
        return link_type, notes

    generation = 0
    with contextlib.suppress(Exception):
        generation = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
    if not generation:
        notes.append(
            "could not read the PCIe link generation; assuming gen3 x16, which is the "
            "pessimistic case for tensor parallelism"
        )
    return _pcie_generation_link(generation or 3), notes


def _peer_access_note(pynvml: Any, handle: Any, index: int, visible: int) -> str | None:
    """Warn when peer-to-peer is unavailable between devices.

    Without P2P, every all-reduce is staged through host memory, so the effective
    tensor-parallel cost is worse than the raw link bandwidth suggests. Virtualised GPUs
    commonly disable it, and nothing else in the stack would reveal that.
    """
    peer_index = 1 if index == 0 else 0
    if peer_index >= visible:
        return None
    try:
        peer = pynvml.nvmlDeviceGetHandleByIndex(peer_index)
        status = pynvml.nvmlDeviceGetP2PStatus(handle, peer, pynvml.NVML_P2P_CAPS_INDEX_READ)
    except Exception:
        return None
    if int(status) == int(pynvml.NVML_P2P_STATUS_OK):
        return None
    return (
        "peer-to-peer access between GPUs is NOT available, so tensor-parallel "
        "all-reduces are staged through host memory. Treat the interconnect term as an "
        "upper bound on performance."
    )


def _import_nvml() -> Any:
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise NVMLUnavailableError(
            "NVML bindings are not installed. Install the optional extra with "
            "'pip install infertune[nvml]', or use the spec-database path "
            "(infertune plan --gpu ...) which needs no hardware."
        ) from exc
    return pynvml


def discover(index: int = 0, *, count: int | None = None) -> GPUProfile:
    """Build a :class:`GPUProfile` from a live device.

    Args:
        index: Device index to inspect.
        count: Devices to assume are available. Defaults to the visible device count.
    """
    pynvml = _import_nvml()
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # pragma: no cover - hardware dependent
        raise NVMLUnavailableError(f"nvmlInit failed: {exc}") from exc

    try:
        visible = int(pynvml.nvmlDeviceGetCount())
        if visible < 1:
            raise NVMLUnavailableError("NVML reports no devices")
        if not 0 <= index < visible:
            raise NVMLUnavailableError(f"device index {index} out of range (0..{visible - 1})")

        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):  # pragma: no cover - binding version dependent
            name = name.decode()
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        sm_count = int(pynvml.nvmlDeviceGetNumGpuCores(handle) or 0) or None

        effective_count = count if count is not None else visible
        detected_link = Interconnect.NONE
        link_notes: list[str] = []
        peer_note: str | None = None
        if effective_count > 1:
            detected_link, link_notes = _detect_interconnect(pynvml, handle)
            peer_note = _peer_access_note(pynvml, handle, index, visible)
    finally:
        with contextlib.suppress(Exception):  # best effort; shutdown must not mask errors
            pynvml.nvmlShutdown()

    total = int(memory.total)
    # NVML's `free` already excludes the driver reservation and any co-tenant allocations,
    # which is exactly the figure the estimator should budget against.
    usable = int(memory.free)

    notes: list[str] = []
    flops: dict[str, float] = {}
    resolved_sm = sm_count or 0

    try:
        reference = specdb.load(str(name))
        flops = dict(reference.dense_flops)
        bandwidth = reference.mem_bandwidth_bytes_s
        resolved_sm = resolved_sm or reference.sm_count
    except (specdb.UnknownGPUError, RuntimeError):
        flops = dict(_FALLBACK_FLOPS.get(int(major), {"bf16": 100e12}))
        bandwidth = 1000e9
        resolved_sm = resolved_sm or 1
        notes.append(
            f"{name!r} is not in the spec database; peak FLOPS and memory bandwidth are "
            "coarse fallbacks derived from compute capability. Memory analysis is unaffected; "
            "throughput predictions will be unreliable until calibrated."
        )

    if usable < total:
        notes.append(
            f"budgeting against NVML free memory ({usable / total:.1%} of total), which "
            "already accounts for the driver reservation and any co-tenant allocations"
        )

    notes.extend(link_notes)
    if peer_note is not None:
        notes.append(peer_note)

    return GPUProfile(
        name=str(name),
        vram_bytes=total,
        vram_usable_bytes=usable,
        compute_capability=(int(major), int(minor)),
        sm_count=resolved_sm,
        mem_bandwidth_bytes_s=bandwidth,
        dense_flops=flops,
        count=effective_count,
        interconnect=detected_link,
        source="nvml",
        notes=tuple(notes),
    )


def is_available() -> bool:
    """Whether a live GPU can be inspected. Never raises."""
    try:
        discover()
    except Exception:
        return False
    return True


__all__ = ["NVMLUnavailableError", "discover", "is_available"]
