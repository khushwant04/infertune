"""Live GPU discovery.

Exercised against a fake ``pynvml`` module rather than hardware, so the multi-GPU paths are
testable on a laptop and in CI. The interconnect logic in particular was wrong for months
precisely because no CPU-only test could reach it.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from infertune.core.gpu import Interconnect
from infertune.hardware import nvml


class _Memory:
    def __init__(self, total: int, free: int) -> None:
        self.total = total
        self.free = free


class FakeNVML:
    """Minimal stand-in for the NVML bindings.

    ``nvlink_links`` of 0 means the device raises on NVLink queries, which is how a card with
    no NVLink support actually behaves rather than returning a falsy state.
    """

    NVML_P2P_CAPS_INDEX_READ = 0
    NVML_P2P_STATUS_OK = 0

    def __init__(
        self,
        *,
        count: int = 1,
        name: str = "NVIDIA A10",
        total: int = 24 * 1024**3,
        free: int = 21 * 1024**3,
        capability: tuple[int, int] = (8, 6),
        cores: int = 72,
        nvlink_links: int = 0,
        nvlink_version: int = 4,
        pcie_generation: int | None = 4,
        p2p_ok: bool = True,
    ) -> None:
        self._count = count
        self._name = name
        self._total = total
        self._free = free
        self._capability = capability
        self._cores = cores
        self._nvlink_links = nvlink_links
        self._nvlink_version = nvlink_version
        self._pcie_generation = pcie_generation
        self._p2p_ok = p2p_ok
        self.shutdown_calls = 0

    # -- lifecycle ------------------------------------------------------------------
    def nvmlInit(self) -> None:  # noqa: N802
        return None

    def nvmlShutdown(self) -> None:  # noqa: N802
        self.shutdown_calls += 1

    # -- device basics --------------------------------------------------------------
    def nvmlDeviceGetCount(self) -> int:  # noqa: N802
        return self._count

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:  # noqa: N802
        if not 0 <= index < self._count:
            raise RuntimeError("bad index")
        return f"handle{index}"

    def nvmlDeviceGetName(self, handle: str) -> str:  # noqa: N802
        return self._name

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> _Memory:  # noqa: N802
        return _Memory(self._total, self._free)

    def nvmlDeviceGetCudaComputeCapability(self, handle: str) -> tuple[int, int]:  # noqa: N802
        return self._capability

    def nvmlDeviceGetNumGpuCores(self, handle: str) -> int:  # noqa: N802
        return self._cores

    # -- interconnect ---------------------------------------------------------------
    def nvmlDeviceGetNvLinkState(self, handle: str, link: int) -> int:  # noqa: N802
        if link >= self._nvlink_links:
            raise RuntimeError("link not supported")
        return 1

    def nvmlDeviceGetNvLinkVersion(self, handle: str, link: int) -> int:  # noqa: N802
        return self._nvlink_version

    def nvmlDeviceGetMaxPcieLinkGeneration(self, handle: str) -> int:  # noqa: N802
        if self._pcie_generation is None:
            raise RuntimeError("not supported under virtualisation")
        return self._pcie_generation

    def nvmlDeviceGetP2PStatus(self, a: str, b: str, index: int) -> int:  # noqa: N802
        return 0 if self._p2p_ok else 3


@pytest.fixture
def install_nvml(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _install(fake: FakeNVML) -> FakeNVML:
        module = types.ModuleType("pynvml")
        for attr in dir(fake):
            if attr.startswith("nvml") or attr.startswith("NVML_"):
                setattr(module, attr, getattr(fake, attr))
        monkeypatch.setitem(sys.modules, "pynvml", module)
        return fake

    return _install


class TestSingleGPU:
    def test_reports_capacity_and_free_memory(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(total=24 * 1024**3, free=21 * 1024**3))
        gpu = nvml.discover()
        assert gpu.count == 1
        assert gpu.vram_bytes == 24 * 1024**3
        assert gpu.vram_usable_bytes == 21 * 1024**3
        assert gpu.source == "nvml"

    def test_single_gpu_has_no_interconnect(self, install_nvml: Any) -> None:
        """A lone device must not claim a link, or the roofline adds phantom all-reduces."""
        install_nvml(FakeNVML(count=1, pcie_generation=4))
        assert nvml.discover().interconnect is Interconnect.NONE

    def test_notes_that_the_budget_is_free_memory(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(total=100, free=90, name="NVIDIA A10"))
        gpu = nvml.discover()
        assert any("free memory" in note for note in gpu.notes)


class TestMultiGPUInterconnect:
    """Regression tests for a bug that broke discovery on every multi-GPU host.

    ``interconnect`` used to be copied from the spec-database entry for the device. Spec
    entries describe a single card and carry ``NONE``, so ``GPUProfile`` rejected the profile
    outright with "multi-GPU profiles must specify an interconnect" and ``infertune profile``
    was unusable on exactly the hardware people serve on.
    """

    def test_multi_gpu_discovery_succeeds(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(count=2, pcie_generation=4))
        gpu = nvml.discover(count=2)
        assert gpu.count == 2
        assert gpu.interconnect is not Interconnect.NONE

    @pytest.mark.parametrize(
        ("generation", "expected"),
        [
            (3, Interconnect.PCIE_GEN3_X16),
            (4, Interconnect.PCIE_GEN4_X16),
            (5, Interconnect.PCIE_GEN5_X16),
            (6, Interconnect.PCIE_GEN5_X16),
        ],
    )
    def test_pcie_generation_maps_to_a_link(
        self, install_nvml: Any, generation: int, expected: Interconnect
    ) -> None:
        install_nvml(FakeNVML(count=2, pcie_generation=generation))
        assert nvml.discover(count=2).interconnect is expected

    def test_nvlink_takes_precedence_over_pcie(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(count=2, nvlink_links=12, nvlink_version=4, pcie_generation=4))
        gpu = nvml.discover(count=2)
        assert gpu.interconnect is Interconnect.NVLINK_4
        assert any("NVLink" in note for note in gpu.notes)

    def test_unknown_nvlink_version_assumes_the_newest(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(count=2, nvlink_links=4, nvlink_version=9))
        assert nvml.discover(count=2).interconnect is Interconnect.NVLINK_5

    def test_unreadable_pcie_generation_falls_back_pessimistically(self, install_nvml: Any) -> None:
        """Virtualised GPUs hide the link generation; guessing high would flatter TP."""
        install_nvml(FakeNVML(count=2, pcie_generation=None))
        gpu = nvml.discover(count=2)
        assert gpu.interconnect is Interconnect.PCIE_GEN3_X16
        assert any("pessimistic" in note for note in gpu.notes)

    def test_count_override_below_visible_devices(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(count=4, pcie_generation=4))
        assert nvml.discover(count=1).interconnect is Interconnect.NONE


class TestPeerAccess:
    def test_missing_p2p_is_reported(self, install_nvml: Any) -> None:
        """Without P2P, all-reduces stage through host memory and TP costs more than the link."""
        install_nvml(FakeNVML(count=2, pcie_generation=4, p2p_ok=False))
        gpu = nvml.discover(count=2)
        assert any("peer-to-peer" in note for note in gpu.notes)

    def test_working_p2p_is_not_mentioned(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(count=2, pcie_generation=4, p2p_ok=True))
        gpu = nvml.discover(count=2)
        assert not any("peer-to-peer" in note for note in gpu.notes)


class TestFailureModes:
    def test_no_devices_raises(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(count=0))
        with pytest.raises(nvml.NVMLUnavailableError, match="no devices"):
            nvml.discover()

    def test_index_out_of_range_raises(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(count=1))
        with pytest.raises(nvml.NVMLUnavailableError, match="out of range"):
            nvml.discover(index=5)

    def test_missing_bindings_explain_the_alternative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "pynvml", None)
        with pytest.raises(nvml.NVMLUnavailableError):
            nvml.discover()

    def test_is_available_never_raises(self, install_nvml: Any) -> None:
        install_nvml(FakeNVML(count=0))
        assert nvml.is_available() is False
