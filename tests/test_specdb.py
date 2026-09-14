"""GPU specification database."""

from __future__ import annotations

import pytest

from infertune.core.dtypes import DType
from infertune.core.gpu import Interconnect
from infertune.core.units import gb, gib, to_gb, to_gib
from infertune.hardware import specdb


def test_database_is_populated() -> None:
    keys = specdb.available()
    assert len(keys) >= 15
    assert "h100-sxm" in keys
    assert "rtx-4090" in keys


@pytest.mark.parametrize("key", specdb.available())
def test_every_entry_builds_a_valid_profile(key: str) -> None:
    """Validation lives in GPUProfile, so this also checks every entry's internal consistency."""
    gpu = specdb.load(key)
    assert gpu.vram_usable_bytes <= gpu.vram_bytes
    assert gpu.mem_bandwidth_bytes_s > 0
    assert gpu.dense_flops
    assert gpu.source == "specdb"
    assert gpu.notes, "spec-sheet profiles must warn that they are not measured"


def test_vram_is_stored_in_binary_units() -> None:
    """A "24 GB" 4090 must be ~24 GiB, not 24 decimal GB. See docs/plan.md 2.6."""
    gpu = specdb.load("rtx-4090")
    assert to_gib(gpu.vram_bytes) == pytest.approx(23.99, abs=0.02)
    assert gpu.vram_bytes > gb(24)


def test_h100_80gb_is_about_85_decimal_gb() -> None:
    gpu = specdb.load("h100-sxm")
    assert to_gb(gpu.vram_bytes) == pytest.approx(85.5, abs=0.3)
    assert to_gib(gpu.vram_bytes) == pytest.approx(79.65, abs=0.05)


def test_lookup_by_product_name() -> None:
    assert specdb.load("NVIDIA H100 SXM").name == "NVIDIA H100 SXM"


def test_lookup_is_case_and_separator_insensitive() -> None:
    assert specdb.load("H100_SXM").name == specdb.load("h100-sxm").name


def test_unique_prefix_match() -> None:
    assert specdb.load("h200").name == "NVIDIA H200 SXM"


def test_ambiguous_match_refuses() -> None:
    with pytest.raises(specdb.UnknownGPUError, match="ambiguous"):
        specdb.load("h100")


def test_unknown_gpu_lists_known_keys() -> None:
    with pytest.raises(specdb.UnknownGPUError, match="known keys"):
        specdb.load("gtx-750-ti")


def test_multi_gpu_sets_an_interconnect() -> None:
    single = specdb.load("h100-sxm")
    pair = specdb.load("h100-sxm", count=2)
    assert single.interconnect is Interconnect.NONE
    assert pair.interconnect is Interconnect.NVLINK_4
    assert pair.count == 2


def test_pcie_and_sxm_h100_differ_in_interconnect_and_throughput() -> None:
    """Same VRAM, different optimal configuration — the premise of hardware awareness."""
    sxm = specdb.load("h100-sxm", count=2)
    pcie = specdb.load("h100-pcie", count=2)
    assert sxm.vram_bytes == pcie.vram_bytes
    assert sxm.interconnect.bytes_s > pcie.interconnect.bytes_s
    assert sxm.flops(DType.BF16) > pcie.flops(DType.BF16)
    assert sxm.critical_batch_size(DType.BF16) != pcie.critical_batch_size(DType.BF16)


def test_dense_flops_are_not_sparsity_inflated() -> None:
    """H100 dense bf16 is ~990 TFLOPS; the 1979 figure requires structured sparsity."""
    assert specdb.load("h100-sxm").flops(DType.BF16) == pytest.approx(990e12, rel=0.02)


def test_critical_batch_size_spans_an_order_of_magnitude() -> None:
    """Why the same model needs different configs on different GPUs."""
    consumer = specdb.load("rtx-4090").critical_batch_size(DType.BF16)
    datacenter = specdb.load("h100-sxm").critical_batch_size(DType.BF16)
    assert consumer == pytest.approx(51, abs=3)
    assert datacenter == pytest.approx(185, abs=6)
    assert datacenter / consumer > 3


def test_fp8_support_is_capability_gated() -> None:
    assert specdb.load("h100-sxm").supports(DType.FP8_E4M3)
    assert specdb.load("rtx-4090").supports(DType.FP8_E4M3)
    assert not specdb.load("a100-80gb").supports(DType.FP8_E4M3)
    assert not specdb.load("v100-32gb").supports(DType.FP8_E4M3)


def test_usable_fraction_leaves_room_for_the_driver() -> None:
    for key in specdb.available():
        gpu = specdb.load(key)
        overhead = 1 - gpu.vram_usable_bytes / gpu.vram_bytes
        assert 0.005 < overhead < 0.05, f"{key} reserves an implausible {overhead:.1%}"


def test_load_is_cached_but_returns_equal_values() -> None:
    assert specdb.load("l40s") == specdb.load("l40s")


def test_gib_helper_matches_database_units() -> None:
    assert specdb.load("a100-80gb").vram_bytes == gib(79.15)
