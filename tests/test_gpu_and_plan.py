"""GPU profile and resource plan tests."""

from __future__ import annotations

import pytest

from infertune.core.dtypes import DType
from infertune.core.gpu import GPUProfile, Interconnect
from infertune.core.plan import (
    BindingConstraint,
    DtypePlan,
    LedgerEntry,
    Parallelism,
    ResourcePlan,
)
from infertune.core.units import gb, gib, mib, to_gb, to_gib, vram_nameplate

# VRAM figures are the *measured* nvidia-smi totals, which are binary. A "24 GB" 4090
# reports 24564 MiB and an "80 GB" H100 reports 81559 MiB. Treating these as decimal GB
# understates capacity by 7.4% and silently costs KV cache.
RTX_4090 = GPUProfile(
    name="NVIDIA RTX 4090",
    vram_bytes=mib(24564),
    vram_usable_bytes=gib(23.6),
    compute_capability=(8, 9),
    sm_count=128,
    mem_bandwidth_bytes_s=1008e9,
    dense_flops={"bf16": 82.6e12, "fp16": 82.6e12, "fp8_e4m3": 165.2e12},
    source="nvml",
)

H100_SXM = GPUProfile(
    name="NVIDIA H100 SXM",
    vram_bytes=mib(81559),
    vram_usable_bytes=gib(78.0),
    compute_capability=(9, 0),
    sm_count=132,
    mem_bandwidth_bytes_s=3350e9,
    dense_flops={"bf16": 990e12, "fp8_e4m3": 1979e12},
    source="nvml",
)


def test_usable_vram_is_below_nameplate() -> None:
    assert RTX_4090.vram_usable_bytes < RTX_4090.vram_bytes


def test_vram_is_binary_not_decimal() -> None:
    """The 7.4% trap.

    A "24 GB" card holds ~24 GiB, not 24 decimal GB. Reading it as decimal loses
    1.65 GiB — over 13,000 tokens of KV cache for Llama-3.1-8B at bf16.
    """
    assert vram_nameplate(24) == gib(24)
    assert vram_nameplate(24) > gb(24)
    assert RTX_4090.vram_bytes == pytest.approx(gib(24), rel=0.001)
    assert to_gib(RTX_4090.vram_bytes) == pytest.approx(23.99, abs=0.01)

    lost = vram_nameplate(24) - gb(24)
    assert to_gib(lost) == pytest.approx(1.65, abs=0.02)


def test_h100_80gb_is_about_85_decimal_gb() -> None:
    """The same arithmetic, from the other direction."""
    assert to_gb(H100_SXM.vram_bytes) == pytest.approx(85.5, abs=0.3)
    assert to_gib(H100_SXM.vram_bytes) == pytest.approx(79.6, abs=0.2)


def test_vram_nameplate_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="advertised_gb"):
        vram_nameplate(0)


def test_usable_cannot_exceed_total() -> None:
    with pytest.raises(ValueError, match="vram_usable_bytes"):
        GPUProfile(
            name="bogus",
            vram_bytes=gb(24),
            vram_usable_bytes=gb(25),
            compute_capability=(8, 9),
            sm_count=128,
            mem_bandwidth_bytes_s=1008e9,
            dense_flops={"bf16": 82.6e12},
        )


def test_dtype_support_is_derived_from_flops_table() -> None:
    assert RTX_4090.supports(DType.FP8_E4M3)
    assert not RTX_4090.supports(DType.FP32)
    with pytest.raises(ValueError, match="no dense FLOPS figure"):
        RTX_4090.flops(DType.FP32)


def test_h100_critical_batch_size_is_around_185() -> None:
    """B* = (FLOPS x MFU) / (bandwidth x MBU).

    ~185 for H100 at bf16, consistent with the widely reported few-hundred range where
    decode transitions from bandwidth-bound to compute-bound.
    """
    assert H100_SXM.critical_batch_size(DType.BF16) == pytest.approx(185, abs=5)


def test_4090_critical_batch_size_is_far_lower_than_h100() -> None:
    """Same arithmetic, very different answer — which is why GPU choice changes the config."""
    assert RTX_4090.critical_batch_size(DType.BF16) == pytest.approx(51, abs=3)
    assert RTX_4090.critical_batch_size(DType.BF16) < H100_SXM.critical_batch_size(DType.BF16)


@pytest.mark.parametrize(("mfu", "mbu"), [(0.0, 0.8), (1.1, 0.8), (0.5, 0.0), (0.5, 1.1)])
def test_critical_batch_size_validates_efficiency_factors(mfu: float, mbu: float) -> None:
    with pytest.raises(ValueError):
        RTX_4090.critical_batch_size(DType.BF16, mfu=mfu, mbu=mbu)


def test_multi_gpu_requires_an_interconnect() -> None:
    with pytest.raises(ValueError, match="interconnect"):
        GPUProfile(
            name="pair",
            vram_bytes=mib(81559),
            vram_usable_bytes=gib(78),
            compute_capability=(9, 0),
            sm_count=132,
            mem_bandwidth_bytes_s=3350e9,
            dense_flops={"bf16": 990e12},
            count=2,
        )


def test_pcie_has_higher_latency_and_lower_bandwidth_than_nvlink() -> None:
    """Both terms matter: TP does two all-reduces per layer, so latency compounds."""
    assert Interconnect.PCIE_GEN4_X16.bytes_s < Interconnect.NVLINK_5.bytes_s
    assert Interconnect.PCIE_GEN4_X16.latency_s > Interconnect.NVLINK_5.latency_s
    assert Interconnect.PCIE_GEN4_X16.is_pcie
    assert not Interconnect.NVLINK_5.is_pcie


def test_parallelism_gpu_count() -> None:
    assert Parallelism(tensor=2, pipeline=2).gpus_required == 4
    assert Parallelism().gpus_required == 1
    with pytest.raises(ValueError, match="tensor parallel size"):
        Parallelism(tensor=0)


def _plan(**overrides: object) -> ResourcePlan:
    defaults: dict[str, object] = {
        "weight_bytes_per_gpu": gib(14.96),
        "activation_peak_bytes": gib(0.85),
        "fixed_overhead_bytes": gib(1.75),
        "safety_bytes": gib(0.67),
        "kv_budget_bytes": gib(3.77),
        "kv_bytes_per_token": 131_072,
        "parallelism": Parallelism(),
        "dtypes": DtypePlan(weights=DType.BF16, activations=DType.BF16, kv_cache=DType.BF16),
        "binding_constraint": BindingConstraint.KV_WORKING_SET,
    }
    defaults.update(overrides)
    return ResourcePlan(**defaults)  # type: ignore[arg-type]


def test_kv_budget_converts_to_token_capacity() -> None:
    assert _plan().kv_budget_tokens == pytest.approx(30_900, abs=50)


def test_ledger_reconciliation_detects_a_missing_term() -> None:
    """A ledger that does not sum to usable VRAM has a missing or double-counted term.

    That is exactly the bug class that produces boot-time OOM, so it is checkable.
    """
    plan = _plan()
    assert plan.reconciles_with(plan.total_allocated_bytes)
    assert not plan.reconciles_with(plan.total_allocated_bytes + gib(1))


def test_binding_constraint_always_explains_itself() -> None:
    for constraint in BindingConstraint:
        assert constraint.explain().strip()


def test_plan_rejects_negative_terms() -> None:
    with pytest.raises(ValueError, match="safety_bytes"):
        _plan(safety_bytes=-1)


def test_plan_rejects_zero_kv_bytes_per_token() -> None:
    with pytest.raises(ValueError, match="kv_bytes_per_token"):
        _plan(kv_bytes_per_token=0)


def test_ledger_entry_requires_a_label() -> None:
    with pytest.raises(ValueError, match="label"):
        LedgerEntry(label="  ", bytes_=1)


def test_ledger_entry_renders_in_binary_units() -> None:
    assert str(LedgerEntry(label="weights", bytes_=gib(14.96))) == "weights: 14.96 GiB"
