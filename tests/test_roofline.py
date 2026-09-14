"""Roofline performance model.

Pure arithmetic, so every property here is checkable without a GPU. The tests assert
*physical behaviour* — which regime binds, how cost scales — rather than pinning magic numbers,
because the numbers move when coefficients are calibrated while the physics does not.
"""

from __future__ import annotations

import pytest

from infertune.core.dtypes import DType
from infertune.core.gpu import GPUProfile, Interconnect
from infertune.core.plan import Parallelism, ResourcePlan
from infertune.core.units import gib, mib
from infertune.estimator.roofline import (
    Bound,
    RooflineCoefficients,
    communication_time,
    critical_batch_size,
    decode_step,
    predict,
    prefill_step,
    throughput_curve,
)
from infertune.hardware import specdb
from tests.test_estimator import LLAMA_31_8B, WORKLOAD  # reuse the shared fixtures


def _plan(tp: int = 1) -> ResourcePlan:
    from infertune.estimator import estimate_plan

    gpu = specdb.load("h100-sxm", count=max(1, tp))
    return estimate_plan(
        LLAMA_31_8B,
        gpu,
        WORKLOAD,
        max_num_seqs=32,
        max_model_len=8192,
        samples=300,
        parallelism=Parallelism(tensor=tp),
    )


class TestCoefficients:
    def test_priors_are_mid_range(self) -> None:
        c = RooflineCoefficients()
        assert 0 < c.mfu <= 1
        assert 0 < c.mbu <= 1
        assert c.source == "prior"

    def test_uncertainty_is_wide_when_uncalibrated(self) -> None:
        assert RooflineCoefficients(samples=0).relative_uncertainty == pytest.approx(0.50)

    def test_uncertainty_narrows_with_evidence_but_has_a_floor(self) -> None:
        """An analytical model over a real serving stack is never better than ~10%."""
        widths = [RooflineCoefficients(samples=n).relative_uncertainty for n in (1, 4, 16, 64)]
        assert widths == sorted(widths, reverse=True), "more evidence must not widen it"
        assert RooflineCoefficients(samples=10_000).relative_uncertainty == pytest.approx(0.10)

    @pytest.mark.parametrize(("mfu", "mbu"), [(0, 0.8), (1.1, 0.8), (0.5, 0), (0.5, 1.1)])
    def test_validation(self, mfu: float, mbu: float) -> None:
        with pytest.raises(ValueError):
            RooflineCoefficients(mfu=mfu, mbu=mbu)


class TestDecode:
    def test_decode_is_memory_bound_at_small_batch(self) -> None:
        """The premise of the whole model: one token per sequence reads all the weights."""
        step = decode_step(
            LLAMA_31_8B,
            specdb.load("h100-sxm"),
            batch=1,
            avg_context_tokens=1024,
            kv_bytes_per_token=128 * 1024,
        )
        assert step.bound is Bound.MEMORY
        assert step.t_memory_s > step.t_compute_s

    def test_weights_dominate_bytes_moved_at_small_batch(self) -> None:
        step = decode_step(
            LLAMA_31_8B,
            specdb.load("h100-sxm"),
            batch=1,
            avg_context_tokens=128,
            kv_bytes_per_token=128 * 1024,
        )
        assert step.bytes_moved > LLAMA_31_8B.weight_bytes
        assert step.bytes_moved < LLAMA_31_8B.weight_bytes * 1.2

    def test_throughput_rises_sublinearly_then_saturates(self) -> None:
        """Below B*, extra concurrency is nearly free; above it, it is not."""
        gpu = specdb.load("h100-sxm")
        kw = {"avg_context_tokens": 1024, "kv_bytes_per_token": 128 * 1024}
        t1 = decode_step(LLAMA_31_8B, gpu, batch=1, **kw).total_s  # type: ignore[arg-type]
        t8 = decode_step(LLAMA_31_8B, gpu, batch=8, **kw).total_s  # type: ignore[arg-type]
        # 8x the work in far less than 8x the time.
        assert t8 < t1 * 3
        assert 8 / t8 > 1 / t1 * 3

    def test_compute_takes_over_above_the_critical_batch(self) -> None:
        gpu = specdb.load("h100-sxm")
        crit = critical_batch_size(gpu, LLAMA_31_8B.weight_dtype)
        step = decode_step(
            LLAMA_31_8B,
            gpu,
            batch=int(crit * 6),
            avg_context_tokens=64,
            kv_bytes_per_token=128 * 1024,
        )
        assert step.bound is Bound.COMPUTE

    def test_kv_cache_reads_matter_at_long_context(self) -> None:
        gpu = specdb.load("h100-sxm")
        short = decode_step(
            LLAMA_31_8B, gpu, batch=32, avg_context_tokens=128, kv_bytes_per_token=128 * 1024
        )
        long = decode_step(
            LLAMA_31_8B, gpu, batch=32, avg_context_tokens=32768, kv_bytes_per_token=128 * 1024
        )
        assert long.bytes_moved > short.bytes_moved * 5
        assert long.total_s > short.total_s

    def test_tensor_parallelism_reduces_per_gpu_bytes(self) -> None:
        gpu = specdb.load("h100-sxm", count=4)
        kw = {"batch": 8, "avg_context_tokens": 1024, "kv_bytes_per_token": 128 * 1024}
        one = decode_step(LLAMA_31_8B, gpu, tensor_parallel_size=1, **kw)  # type: ignore[arg-type]
        four = decode_step(LLAMA_31_8B, gpu, tensor_parallel_size=4, **kw)  # type: ignore[arg-type]
        assert four.bytes_moved < one.bytes_moved

    @pytest.mark.parametrize("batch", [0, -1])
    def test_validation(self, batch: int) -> None:
        with pytest.raises(ValueError, match="batch"):
            decode_step(
                LLAMA_31_8B,
                specdb.load("h100-sxm"),
                batch=batch,
                avg_context_tokens=1,
                kv_bytes_per_token=1,
            )


class TestPrefill:
    def test_prefill_is_compute_bound(self) -> None:
        step = prefill_step(LLAMA_31_8B, specdb.load("h100-sxm"), prompt_tokens=4096)
        assert step.bound is Bound.COMPUTE

    def test_ttft_is_superlinear_in_prompt_length(self) -> None:
        """Attention is quadratic, so TTFT stops being linear at long context."""
        gpu = specdb.load("h100-sxm")
        short = prefill_step(LLAMA_31_8B, gpu, prompt_tokens=1024).total_s
        long = prefill_step(LLAMA_31_8B, gpu, prompt_tokens=32768).total_s
        assert long / short > 32, "quadratic attention term is missing"

    def test_faster_gpu_lowers_ttft(self) -> None:
        a10 = prefill_step(LLAMA_31_8B, specdb.load("a10"), prompt_tokens=4096).total_s
        h100 = prefill_step(LLAMA_31_8B, specdb.load("h100-sxm"), prompt_tokens=4096).total_s
        assert h100 < a10


class TestCommunication:
    def test_no_cost_without_tensor_parallelism(self) -> None:
        assert communication_time(specdb.load("h100-sxm"), LLAMA_31_8B, 32, 1) == 0.0

    def test_pcie_costs_far_more_than_nvlink(self) -> None:
        """The quantitative form of 'topology matters'."""
        nvlink = specdb.load("h100-sxm", count=2)
        pcie = specdb.load("h100-pcie", count=2)
        assert nvlink.interconnect is Interconnect.NVLINK_4
        assert pcie.interconnect is Interconnect.PCIE_GEN5_X16
        t_nv = communication_time(nvlink, LLAMA_31_8B, 32, 2)
        t_pcie = communication_time(pcie, LLAMA_31_8B, 32, 2)
        assert t_pcie > t_nv * 2

    def test_latency_term_dominates_at_small_payloads(self) -> None:
        """Two all-reduces per layer means dozens of synchronisations per decode step."""
        pcie = specdb.load("h100-pcie", count=2)
        tiny = communication_time(pcie, LLAMA_31_8B, 1, 2)
        expected_latency = 2 * LLAMA_31_8B.n_layers * pcie.interconnect.latency_s
        assert tiny == pytest.approx(expected_latency, rel=0.2)

    def test_communication_can_become_the_bound(self) -> None:
        step = decode_step(
            LLAMA_31_8B,
            specdb.load("h100-pcie", count=8),
            batch=1,
            avg_context_tokens=16,
            kv_bytes_per_token=128 * 1024,
            tensor_parallel_size=8,
        )
        assert step.t_communication_s > 0


class TestCriticalBatchSize:
    def test_h100_is_around_185(self) -> None:
        assert critical_batch_size(specdb.load("h100-sxm"), DType.BF16) == pytest.approx(185, abs=6)

    def test_a10_is_far_lower_than_h100(self) -> None:
        a10 = critical_batch_size(specdb.load("a10"), DType.BF16)
        h100 = critical_batch_size(specdb.load("h100-sxm"), DType.BF16)
        assert a10 == pytest.approx(130, abs=5)
        assert a10 < h100

    def test_scales_with_coefficients(self) -> None:
        gpu = specdb.load("h100-sxm")
        base = critical_batch_size(gpu, DType.BF16, RooflineCoefficients(mfu=0.5, mbu=0.8))
        higher_mfu = critical_batch_size(gpu, DType.BF16, RooflineCoefficients(mfu=0.8, mbu=0.8))
        assert higher_mfu > base


class TestPredict:
    def test_prediction_carries_intervals_and_names_the_bound(self) -> None:
        plan = _plan()
        p = predict(LLAMA_31_8B, specdb.load("h100-sxm"), plan, batch=32, prompt_tokens=1024)
        assert p.ttft_s.low < p.ttft_s.value < p.ttft_s.high
        assert p.tpot_s.low < p.tpot_s.value < p.tpot_s.high
        assert p.decode_bound in (Bound.MEMORY, Bound.COMPUTE, Bound.COMMUNICATION)
        assert p.prefill_bound is Bound.COMPUTE

    def test_uncalibrated_predictions_are_reported_as_wide(self) -> None:
        p = predict(LLAMA_31_8B, specdb.load("h100-sxm"), _plan(), batch=8, prompt_tokens=512)
        assert p.ttft_s.width_fraction == pytest.approx(1.0, abs=0.01)  # +/-50%

    def test_calibration_narrows_the_interval(self) -> None:
        plan = _plan()
        gpu = specdb.load("h100-sxm")
        loose = predict(LLAMA_31_8B, gpu, plan, batch=8, prompt_tokens=512)
        tight = predict(
            LLAMA_31_8B,
            gpu,
            plan,
            batch=8,
            prompt_tokens=512,
            coefficients=RooflineCoefficients(samples=100, source="calibrated"),
        )
        assert tight.ttft_s.width_fraction < loose.ttft_s.width_fraction

    def test_headroom_to_critical_batch(self) -> None:
        p = predict(LLAMA_31_8B, specdb.load("h100-sxm"), _plan(), batch=16, prompt_tokens=512)
        assert p.headroom_to_critical_batch > 1
        assert p.is_bandwidth_bound

    def test_throughput_curve_is_monotone_in_concurrency(self) -> None:
        curve = throughput_curve(
            LLAMA_31_8B,
            specdb.load("h100-sxm"),
            _plan(),
            batches=(1, 2, 4, 8, 16, 32),
        )
        tps = [c.output_throughput_tokens_s.value for c in curve]
        assert tps == sorted(tps), "throughput must not fall as concurrency rises"


def test_model_without_flops_for_dtype_still_predicts() -> None:
    """A missing FLOPS figure must degrade, not crash: memory analysis is still valid."""
    gpu = GPUProfile(
        name="odd",
        vram_bytes=gib(24),
        vram_usable_bytes=gib(22),
        compute_capability=(8, 6),
        sm_count=72,
        mem_bandwidth_bytes_s=600e9,
        dense_flops={"fp16": 100e12},
    )
    step = decode_step(
        LLAMA_31_8B,
        gpu,
        batch=4,
        avg_context_tokens=512,
        kv_bytes_per_token=mib(1),
        dtype=DType.FP32,
    )
    assert step.total_s > 0
