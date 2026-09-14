"""Measurement store and roofline calibration.

The load-bearing test here is a **round trip**: synthesise measurements from known
coefficients, feed them through calibration, and check the original coefficients come back.
That validates the fitting arithmetic without a GPU, which is the only way to trust it before
spending money on one.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from infertune.core.dtypes import DType
from infertune.estimator.calibration import (
    CalibrationResult,
    calibrate,
    calibrated_for,
    fit_mbu_from_decode,
    fit_mfu_from_prefill,
)
from infertune.estimator.roofline import (
    RooflineCoefficients,
    decode_step,
    prefill_step,
)
from infertune.hardware import specdb
from infertune.store import MeasurementStore, RunKey, StoredPoint, StoredRun
from tests.test_estimator import LLAMA_31_8B

GPU = specdb.load("h100-sxm")
KEY = RunKey(
    gpu="NVIDIA H100 SXM",
    model=LLAMA_31_8B.model_id,
    engine="vllm",
    engine_version="0.19.1",
    tensor_parallel=1,
)
KV_PER_TOKEN = 128 * 1024


class TestStore:
    def test_round_trips_a_run(self, tmp_path: Path) -> None:
        store = MeasurementStore(tmp_path / "t.db")
        run = StoredRun(
            key=KEY,
            max_num_seqs=32,
            max_model_len=8192,
            kv_dtype="auto",
            weight_bytes=LLAMA_31_8B.weight_bytes,
            kv_bytes_per_token=KV_PER_TOKEN,
            predicted_kv_bytes=1000,
            actual_kv_bytes=1050,
            notes="hello",
            extra={"driver": "570.237"},
            points=(
                StoredPoint(
                    concurrency=8,
                    duration_s=4.0,
                    requests_ok=8,
                    requests_failed=0,
                    prompt_tokens=8192,
                    output_tokens=800,
                    output_tps=200.0,
                    ttft_p50_ms=95.0,
                    ttft_p99_ms=180.0,
                    tpot_p50_ms=12.0,
                    tpot_p99_ms=20.0,
                    avg_context_tokens=1024,
                ),
            ),
        )
        run_id = store.record(run)
        assert run_id > 0

        back = store.runs(KEY)
        assert len(back) == 1
        assert back[0].key == KEY
        assert back[0].extra == {"driver": "570.237"}
        assert len(back[0].points) == 1
        assert back[0].points[0].tpot_p50_ms == 12.0
        assert store.count() == (1, 1)

    def test_kv_error_fraction(self, tmp_path: Path) -> None:
        run = StoredRun(key=KEY, predicted_kv_bytes=105, actual_kv_bytes=100)
        assert run.kv_error_fraction == pytest.approx(0.05)
        assert StoredRun(key=KEY).kv_error_fraction is None

    def test_engine_version_separates_measurements(self, tmp_path: Path) -> None:
        """A measurement from one engine version is not evidence about another.

        vLLM changes memory accounting and scheduling defaults between releases, so pooling
        them silently corrupts a calibration set.
        """
        store = MeasurementStore(tmp_path / "t.db")
        store.record(StoredRun(key=KEY))
        other = dataclasses.replace(KEY, engine_version="0.29.0")
        store.record(StoredRun(key=other))

        assert len(store.runs(KEY)) == 1
        assert len(store.runs(other)) == 1
        assert len(store.runs()) == 2

    def test_tensor_parallel_separates_measurements(self, tmp_path: Path) -> None:
        store = MeasurementStore(tmp_path / "t.db")
        store.record(StoredRun(key=KEY))
        store.record(StoredRun(key=dataclasses.replace(KEY, tensor_parallel=2)))
        assert len(store.runs(KEY)) == 1

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "t.db"
        MeasurementStore(path).record(StoredRun(key=KEY))
        assert MeasurementStore(path).count() == (1, 0)

    def test_schema_mismatch_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "t.db"
        store = MeasurementStore(path)
        with store._connect() as conn:
            conn.execute("UPDATE schema_info SET version = 999")
        with pytest.raises(RuntimeError, match="schema version"):
            MeasurementStore(path)

    def test_json_export(self, tmp_path: Path) -> None:
        store = MeasurementStore(tmp_path / "t.db")
        store.record(StoredRun(key=KEY, notes="n"))
        assert '"notes": "n"' in store.to_json()


class TestFitOneSample:
    def test_mbu_round_trips_through_decode(self) -> None:
        """Generate a decode time from a known MBU, then recover that MBU."""
        truth = RooflineCoefficients(mfu=0.5, mbu=0.62)
        step = decode_step(
            LLAMA_31_8B,
            GPU,
            batch=16,
            avg_context_tokens=1024,
            kv_bytes_per_token=KV_PER_TOKEN,
            coefficients=truth,
        )
        sample = fit_mbu_from_decode(
            LLAMA_31_8B,
            GPU,
            tpot_s=step.total_s,
            concurrency=16,
            avg_context_tokens=1024,
            kv_bytes_per_token=KV_PER_TOKEN,
            fixed_step_overhead_s=truth.fixed_step_overhead_s,
        )
        assert sample is not None
        assert sample.value == pytest.approx(0.62, rel=0.02)
        assert sample.plausible

    def test_mfu_round_trips_through_prefill(self) -> None:
        truth = RooflineCoefficients(mfu=0.37, mbu=0.8)
        step = prefill_step(LLAMA_31_8B, GPU, prompt_tokens=2048, coefficients=truth)
        sample = fit_mfu_from_prefill(
            LLAMA_31_8B,
            GPU,
            ttft_s=step.total_s,
            prompt_tokens=2048,
            fixed_step_overhead_s=truth.fixed_step_overhead_s,
        )
        assert sample is not None
        assert sample.value == pytest.approx(0.37, rel=0.05)

    def test_impossible_measurement_yields_no_sample(self) -> None:
        """If overheads alone exceed the measurement there is nothing to attribute."""
        assert (
            fit_mbu_from_decode(
                LLAMA_31_8B,
                GPU,
                tpot_s=0.0001,
                concurrency=1,
                avg_context_tokens=1,
                kv_bytes_per_token=KV_PER_TOKEN,
                fixed_step_overhead_s=0.01,
            )
            is None
        )
        assert (
            fit_mfu_from_prefill(
                LLAMA_31_8B,
                GPU,
                ttft_s=0.001,
                prompt_tokens=1024,
                fixed_step_overhead_s=0.01,
            )
            is None
        )

    def test_nonpositive_measurements_rejected(self) -> None:
        assert (
            fit_mbu_from_decode(
                LLAMA_31_8B,
                GPU,
                tpot_s=0,
                concurrency=1,
                avg_context_tokens=1,
                kv_bytes_per_token=1,
            )
            is None
        )
        assert fit_mfu_from_prefill(LLAMA_31_8B, GPU, ttft_s=0, prompt_tokens=1) is None


def _synthetic_run(
    truth: RooflineCoefficients, concurrencies: tuple[int, ...] = (1, 2, 8, 32)
) -> StoredRun:
    """Build a run whose latencies come from the roofline model at known coefficients."""
    points = []
    for c in concurrencies:
        decode = decode_step(
            LLAMA_31_8B,
            GPU,
            batch=c,
            avg_context_tokens=1024,
            kv_bytes_per_token=KV_PER_TOKEN,
            coefficients=truth,
        )
        prefill = prefill_step(LLAMA_31_8B, GPU, prompt_tokens=1024, coefficients=truth)
        points.append(
            StoredPoint(
                concurrency=c,
                duration_s=1.0,
                requests_ok=c,
                requests_failed=0,
                prompt_tokens=1024 * c,
                output_tokens=64 * c,
                output_tps=c / decode.total_s,
                ttft_p50_ms=prefill.total_s * 1000,
                ttft_p99_ms=prefill.total_s * 1200,
                tpot_p50_ms=decode.total_s * 1000,
                tpot_p99_ms=decode.total_s * 1200,
                avg_context_tokens=1024,
            )
        )
    return StoredRun(
        key=KEY,
        kv_bytes_per_token=KV_PER_TOKEN,
        weight_bytes=LLAMA_31_8B.weight_bytes,
        points=tuple(points),
    )


class TestCalibrationRoundTrip:
    def test_recovers_known_coefficients(self) -> None:
        """The central validation: synthesise from truth, fit, and recover it."""
        truth = RooflineCoefficients(mfu=0.42, mbu=0.71)
        result = calibrate([_synthetic_run(truth)], LLAMA_31_8B, GPU)

        assert result.is_calibrated
        assert result.coefficients.mbu == pytest.approx(0.71, rel=0.05)
        assert result.coefficients.mfu == pytest.approx(0.42, rel=0.10)
        assert result.coefficients.source == "calibrated"

    def test_calibration_increases_the_sample_count_and_narrows_uncertainty(self) -> None:
        truth = RooflineCoefficients(mfu=0.4, mbu=0.7)
        result = calibrate([_synthetic_run(truth)], LLAMA_31_8B, GPU)
        assert result.coefficients.samples > 0
        assert (
            result.coefficients.relative_uncertainty < RooflineCoefficients().relative_uncertainty
        )

    def test_median_ignores_a_single_absurd_outlier(self) -> None:
        """One evicted spot instance or cold cache must not move the fit."""
        truth = RooflineCoefficients(mfu=0.4, mbu=0.7)
        run = _synthetic_run(truth)
        poisoned = StoredRun(
            key=run.key,
            kv_bytes_per_token=run.kv_bytes_per_token,
            weight_bytes=run.weight_bytes,
            points=(
                *run.points,
                StoredPoint(  # a request that took 100x too long
                    concurrency=8,
                    duration_s=1.0,
                    requests_ok=8,
                    requests_failed=0,
                    prompt_tokens=8192,
                    output_tokens=512,
                    output_tps=1.0,
                    ttft_p50_ms=50_000,
                    tpot_p50_ms=50_000,
                    avg_context_tokens=1024,
                ),
            ),
        )
        result = calibrate([poisoned], LLAMA_31_8B, GPU)
        assert result.coefficients.mbu == pytest.approx(0.7, rel=0.10)

    def test_falls_back_to_the_prior_without_evidence(self) -> None:
        prior = RooflineCoefficients(mfu=0.33, mbu=0.66)
        result = calibrate([], LLAMA_31_8B, GPU, prior=prior)
        assert not result.is_calibrated
        assert result.coefficients.mfu == prior.mfu
        assert result.coefficients.mbu == prior.mbu
        assert result.coefficients.source == "prior"

    def test_implausible_fits_are_rejected_with_a_reason(self) -> None:
        """A compute-bound decode point carries no bandwidth information.

        Fitting MBU from it would produce a confident, wrong number, so it must be counted as
        rejected rather than silently absorbed.
        """
        run = StoredRun(
            key=KEY,
            kv_bytes_per_token=KV_PER_TOKEN,
            points=(
                StoredPoint(
                    concurrency=4,
                    duration_s=1.0,
                    requests_ok=4,
                    requests_failed=0,
                    prompt_tokens=4096,
                    output_tokens=64,
                    output_tps=64.0,
                    tpot_p50_ms=4.0,  # clears overheads but implies mbu ~2.5
                    avg_context_tokens=1024,
                ),
            ),
        )
        result = calibrate([run], LLAMA_31_8B, GPU)
        assert result.rejected
        assert any("implausible mbu" in r for r in result.rejected)
        assert not result.mbu_samples

    def test_points_with_no_successful_requests_are_skipped(self) -> None:
        run = StoredRun(
            key=KEY,
            kv_bytes_per_token=KV_PER_TOKEN,
            points=(
                StoredPoint(
                    concurrency=4,
                    duration_s=1.0,
                    requests_ok=0,
                    requests_failed=4,
                    prompt_tokens=0,
                    output_tokens=0,
                    output_tps=0.0,
                ),
            ),
        )
        result = calibrate([run], LLAMA_31_8B, GPU)
        assert any("no successful requests" in r for r in result.rejected)

    def test_mfu_only_fitted_from_low_concurrency(self) -> None:
        """At high concurrency, queueing inflates TTFT and would depress apparent MFU."""
        truth = RooflineCoefficients(mfu=0.4, mbu=0.7)
        result = calibrate([_synthetic_run(truth, concurrencies=(64, 128))], LLAMA_31_8B, GPU)
        assert not result.mfu_samples, "high-concurrency TTFT must not be used for MFU"

    def test_calibrated_for_filters_by_key(self) -> None:
        truth = RooflineCoefficients(mfu=0.4, mbu=0.7)
        mine = _synthetic_run(truth)
        theirs = StoredRun(key=dataclasses.replace(KEY, engine_version="9.9.9"))
        result = calibrated_for([mine, theirs], KEY, LLAMA_31_8B, GPU)
        assert result.is_calibrated

    def test_summary_is_human_readable(self) -> None:
        result = calibrate([_synthetic_run(RooflineCoefficients())], LLAMA_31_8B, GPU)
        text = result.summary()
        assert "mfu=" in text
        assert "mbu=" in text


def test_calibration_result_is_reportable_when_empty() -> None:
    result = CalibrationResult(coefficients=RooflineCoefficients())
    assert result.n_accepted == 0
    assert not result.is_calibrated
    assert "rejected" in result.summary()


def test_end_to_end_store_then_calibrate(tmp_path: Path) -> None:
    """Persist measurements, read them back, and calibrate from the database."""
    truth = RooflineCoefficients(mfu=0.45, mbu=0.68)
    store = MeasurementStore(tmp_path / "e2e.db")
    store.record(_synthetic_run(truth))

    result = calibrated_for(store.runs(), KEY, LLAMA_31_8B, GPU)
    assert result.coefficients.mbu == pytest.approx(0.68, rel=0.06)
    assert result.coefficients.samples > 0


def test_gpu_without_dtype_flops_still_calibrates() -> None:
    """Missing FLOPS must degrade gracefully, as it does in the roofline model."""
    sample = fit_mfu_from_prefill(
        LLAMA_31_8B,
        specdb.load("a100-80gb"),
        ttft_s=0.5,
        prompt_tokens=1024,
        dtype=DType.FP8_E4M3,  # A100 has no fp8
    )
    assert sample is not None
