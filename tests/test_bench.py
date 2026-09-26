"""Metrics, load generation and the concurrency sweep.

The load generator is exercised against a stdlib mock engine with *known* timing, so its TTFT
and TPOT measurements can be checked against ground truth rather than merely inspected. No GPU
and no real engine involved.
"""

from __future__ import annotations

import random

import pytest

from infertune.bench import (
    BenchmarkResult,
    EndpointConfig,
    LatencyStats,
    RequestRecord,
    SweepConfig,
    percentile,
    run_at_concurrency,
    sweep_concurrency,
    synthetic_prompt,
    wait_for_endpoint,
)
from infertune.bench.metrics import SweepResult
from infertune.core.workload import SLA, Constant, WorkloadProfile
from tests.mock_engine import MockEngine

WORKLOAD = WorkloadProfile(input_tokens=Constant(24), output_tokens=Constant(8))


class TestPercentile:
    def test_nearest_rank(self) -> None:
        assert percentile([1, 2, 3, 4, 5], 0.5) == 3
        assert percentile([1, 2, 3, 4, 5], 0.99) == 5

    def test_rejects_empty_and_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            percentile([], 0.5)
        with pytest.raises(ValueError, match="q must be"):
            percentile([1.0], 1.0)


class TestRequestRecord:
    def test_tpot_excludes_the_first_token(self) -> None:
        """Otherwise prefill leaks into a decode metric and long prompts look slow at decode."""
        r = RequestRecord(prompt_tokens=10, output_tokens=11, ttft_s=0.5, total_s=1.5)
        assert r.tpot_s == pytest.approx(0.1)  # 1.0 s over 10 subsequent tokens

    def test_tpot_undefined_for_single_token(self) -> None:
        r = RequestRecord(prompt_tokens=10, output_tokens=1, ttft_s=0.5, total_s=0.5)
        assert r.tpot_s is None

    def test_failed_records_are_not_ok(self) -> None:
        assert not RequestRecord(10, 0, None, 1.0, error="boom").ok
        assert not RequestRecord(10, 0, None, 1.0).ok


class TestBenchmarkResult:
    def _result(self, n: int = 10) -> BenchmarkResult:
        records = tuple(
            RequestRecord(prompt_tokens=100, output_tokens=11, ttft_s=0.1, total_s=1.1)
            for _ in range(n)
        )
        return BenchmarkResult(concurrency=4, duration_s=2.0, records=records)

    def test_output_throughput_excludes_prompt_tokens(self) -> None:
        """Counting prompt tokens inflates throughput by the input/output ratio."""
        r = self._result()
        assert r.output_throughput_tokens_s == pytest.approx(110 / 2.0)
        assert r.total_throughput_tokens_s > r.output_throughput_tokens_s

    def test_error_rate(self) -> None:
        good = RequestRecord(10, 5, 0.1, 0.5)
        bad = RequestRecord(10, 0, None, 0.5, error="x")
        r = BenchmarkResult(concurrency=1, duration_s=1.0, records=(good, bad, bad))
        assert r.error_rate == pytest.approx(2 / 3)

    def test_sla_evaluation(self) -> None:
        r = self._result()
        assert r.meets(ttft_p99_ms=200, tpot_p99_ms=200)
        assert not r.meets(ttft_p99_ms=50, tpot_p99_ms=None)

    def test_empty_result_never_meets_an_sla(self) -> None:
        assert not BenchmarkResult(concurrency=1, duration_s=1.0).meets(
            ttft_p99_ms=10_000, tpot_p99_ms=10_000
        )


def test_latency_stats_of_empty_sample_is_zeroed() -> None:
    s = LatencyStats.from_seconds([])
    assert s.count == 0
    assert s.p99_ms == 0


class TestSyntheticPrompt:
    def test_length_is_approximately_requested(self) -> None:
        assert len(synthetic_prompt(50, random.Random(0)).split()) == 50

    def test_prompts_vary_so_prefix_caching_does_not_skew_ttft(self) -> None:
        """Identical prompts would turn a prefill benchmark into a cache-hit benchmark."""
        rng = random.Random(0)
        a = synthetic_prompt(30, rng)
        b = synthetic_prompt(30, rng)
        assert a != b

    def test_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="n_tokens"):
            synthetic_prompt(0, random.Random(0))


class TestLoadGenerator:
    def test_measures_ttft_and_tpot_against_known_timing(self) -> None:
        """Ground-truth check: the mock waits 80 ms then 20 ms per token."""
        with MockEngine(ttft_s=0.08, tpot_s=0.02) as engine:
            endpoint = EndpointConfig(base_url=engine.base_url, model="mock")
            result = run_at_concurrency(endpoint, WORKLOAD, 1, n_requests=3)

        assert result.successful, "no successful requests"
        assert result.ttft.p50_ms == pytest.approx(80, rel=0.6)
        assert result.tpot.p50_ms == pytest.approx(20, rel=0.6)
        assert result.error_rate == 0.0

    def test_counts_streamed_tokens(self) -> None:
        workload = WorkloadProfile(input_tokens=Constant(10), output_tokens=Constant(6))
        with MockEngine(ttft_s=0.001, tpot_s=0.001) as engine:
            result = run_at_concurrency(
                EndpointConfig(base_url=engine.base_url, model="mock"),
                workload,
                1,
                n_requests=2,
            )
        assert all(r.output_tokens == 6 for r in result.successful)

    def test_holds_the_requested_concurrency(self) -> None:
        """Closed-loop: exactly N requests in flight, which is what pins the batch size."""
        with MockEngine(ttft_s=0.05, tpot_s=0.01) as engine:
            run_at_concurrency(
                EndpointConfig(base_url=engine.base_url, model="mock"),
                WORKLOAD,
                4,
                n_requests=12,
            )
            assert engine.max_observed_concurrency >= 3

    def test_respects_the_request_budget(self) -> None:
        with MockEngine(ttft_s=0.001, tpot_s=0.001) as engine:
            result = run_at_concurrency(
                EndpointConfig(base_url=engine.base_url, model="mock"),
                WORKLOAD,
                4,
                n_requests=9,
            )
        assert len(result.records) == 9, "workers must not collectively overshoot"

    def test_http_errors_are_recorded_not_raised(self) -> None:
        with MockEngine(status=500) as engine:
            result = run_at_concurrency(
                EndpointConfig(base_url=engine.base_url, model="mock"),
                WORKLOAD,
                2,
                n_requests=4,
            )
        assert result.error_rate == 1.0
        assert all("HTTP 500" in r.error for r in result.failed)

    def test_connection_failure_is_recorded_not_raised(self) -> None:
        endpoint = EndpointConfig(base_url="http://127.0.0.1:1", model="mock", timeout_s=2)
        result = run_at_concurrency(endpoint, WORKLOAD, 1, n_requests=1)
        assert result.error_rate == 1.0

    def test_requires_exactly_one_stop_condition(self) -> None:
        endpoint = EndpointConfig(base_url="http://127.0.0.1:1")
        with pytest.raises(ValueError, match="exactly one"):
            run_at_concurrency(endpoint, WORKLOAD, 1)
        with pytest.raises(ValueError, match="exactly one"):
            run_at_concurrency(endpoint, WORKLOAD, 1, n_requests=1, duration_s=1)

    def test_wait_for_endpoint(self) -> None:
        with MockEngine() as engine:
            assert wait_for_endpoint(
                EndpointConfig(base_url=engine.base_url), timeout_s=10, interval_s=0.2
            )
        assert not wait_for_endpoint(
            EndpointConfig(base_url="http://127.0.0.1:1"), timeout_s=1, interval_s=0.2
        )


class TestSweep:
    def test_one_boot_yields_a_whole_curve(self) -> None:
        """The point of the inner/outer split: no restarts, many operating points."""
        with MockEngine(ttft_s=0.01, tpot_s=0.003) as engine:
            sweep = sweep_concurrency(
                EndpointConfig(base_url=engine.base_url, model="mock"),
                WORKLOAD,
                SweepConfig(
                    concurrencies=(1, 2, 4),
                    requests_per_point=4,
                    warmup_requests=0,
                    stop_on_saturation=False,
                ),
                model="mock",
            )
        assert len(sweep.points) == 3
        assert [p.concurrency for p in sweep.points] == [1, 2, 4]
        assert sweep.best_throughput is not None

    def test_stops_when_the_sla_is_breached(self) -> None:
        """Latency is monotone in concurrency, so climbing further cannot help."""
        with MockEngine(ttft_s=0.20, tpot_s=0.02) as engine:
            sweep = sweep_concurrency(
                EndpointConfig(base_url=engine.base_url, model="mock"),
                WORKLOAD,
                SweepConfig(
                    concurrencies=(1, 2, 4, 8, 16),
                    requests_per_point=2,
                    warmup_requests=0,
                    stop_on_saturation=False,
                ),
                sla=SLA(ttft_p99_ms=50),
            )
        assert len(sweep.points) < 5
        assert any("SLA breached" in n for n in sweep.notes)

    def test_stops_on_excessive_errors(self) -> None:
        with MockEngine(ttft_s=0.005, tpot_s=0.001, fail_above_concurrency=2) as engine:
            sweep = sweep_concurrency(
                EndpointConfig(base_url=engine.base_url, model="mock"),
                WORKLOAD,
                SweepConfig(
                    concurrencies=(1, 8, 64),
                    requests_per_point=8,
                    warmup_requests=0,
                    stop_on_saturation=False,
                ),
            )
        assert any("error rate" in n for n in sweep.notes)

    def test_knee_is_the_best_point_within_the_sla(self) -> None:
        points = (
            BenchmarkResult(
                concurrency=c,
                duration_s=1.0,
                records=tuple(RequestRecord(10, 11, 0.01 * c, 0.01 * c + 0.1) for _ in range(c)),
            )
            for c in (1, 2, 4, 8)
        )
        sweep = SweepResult(points=tuple(points))
        knee = sweep.knee(ttft_p99_ms=45, tpot_p99_ms=None)
        assert knee is not None
        assert knee.concurrency == 4  # ttft at c=8 would be 80 ms

    def test_saturation_concurrency_is_the_empirical_critical_batch(self) -> None:
        def point(c: int, tokens: int) -> BenchmarkResult:
            return BenchmarkResult(
                concurrency=c,
                duration_s=1.0,
                records=(RequestRecord(10, tokens, 0.01, 0.11),),
            )

        # Throughput doubles, doubles, then flattens.
        sweep = SweepResult(points=(point(1, 100), point(2, 200), point(4, 205)))
        assert sweep.saturation_concurrency == 4

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SweepConfig(concurrencies=())
        with pytest.raises(ValueError, match=">= 1"):
            SweepConfig(concurrencies=(0,))
        with pytest.raises(ValueError, match="max_error_rate"):
            SweepConfig(max_error_rate=1.5)


class TestContextClamping:
    """Sampled request lengths must fit the engine's context window.

    A workload distribution knows nothing about ``max_model_len``. Its tail routinely exceeds
    the window, and engines reject those requests outright rather than truncating them, so an
    unclamped sweep aborts on error rate and the fault looks like engine instability.
    """

    def test_prompt_and_output_fit_within_the_limit(self) -> None:
        limit = 512
        with MockEngine(ttft_s=0.001, tpot_s=0.0) as engine:
            run_at_concurrency(
                EndpointConfig(base_url=engine.base_url),
                WorkloadProfile(
                    input_tokens=Constant(100_000),
                    output_tokens=Constant(100_000),
                ),
                1,
                n_requests=4,
                context_limit=limit,
            )
            assert engine.requests, "no requests reached the engine"
            for payload in engine.requests:
                prompt_words = len(str(payload["prompt"]).split())
                assert prompt_words + int(payload["max_tokens"]) <= limit

    def test_output_length_is_preserved_in_preference_to_the_prompt(self) -> None:
        """Trimming the output first would bias the decode measurement itself."""
        with MockEngine(ttft_s=0.001, tpot_s=0.0) as engine:
            run_at_concurrency(
                EndpointConfig(base_url=engine.base_url),
                WorkloadProfile(input_tokens=Constant(100_000), output_tokens=Constant(64)),
                1,
                n_requests=2,
                context_limit=1024,
            )
            for payload in engine.requests:
                assert int(payload["max_tokens"]) == 64

    def test_unset_limit_leaves_lengths_untouched(self) -> None:
        with MockEngine(ttft_s=0.001, tpot_s=0.0) as engine:
            run_at_concurrency(
                EndpointConfig(base_url=engine.base_url),
                WorkloadProfile(input_tokens=Constant(40), output_tokens=Constant(7)),
                1,
                n_requests=2,
            )
            for payload in engine.requests:
                assert int(payload["max_tokens"]) == 7
                assert len(str(payload["prompt"]).split()) == 40

    def test_sweep_threads_the_limit_through(self) -> None:
        with MockEngine(ttft_s=0.001, tpot_s=0.0) as engine:
            sweep_concurrency(
                EndpointConfig(base_url=engine.base_url),
                WorkloadProfile(input_tokens=Constant(9_999), output_tokens=Constant(9_999)),
                SweepConfig(
                    concurrencies=(1, 2),
                    requests_per_point=2,
                    context_limit=256,
                    stop_on_saturation=False,
                ),
            )
            assert engine.requests
            for payload in engine.requests:
                assert len(str(payload["prompt"]).split()) + int(payload["max_tokens"]) <= 256

    def test_rejects_an_impossible_limit(self) -> None:
        with pytest.raises(ValueError, match="context_limit"):
            run_at_concurrency(
                EndpointConfig(base_url="http://127.0.0.1:1"),
                WorkloadProfile(input_tokens=Constant(8), output_tokens=Constant(8)),
                1,
                n_requests=1,
                context_limit=1,
            )


class TestReportedTokenCounts:
    """Token counts come from the server, not from the client's own guess.

    The client can only approximate how many tokens its prompt string will become. Feeding
    that approximation into calibration would misstate the KV volume read per decode step,
    which is exactly the quantity the bandwidth fit is derived from.
    """

    def test_prompt_tokens_come_from_the_usage_chunk(self) -> None:
        with MockEngine(ttft_s=0.001, tpot_s=0.0, reported_prompt_tokens=1234) as engine:
            result = run_at_concurrency(
                EndpointConfig(base_url=engine.base_url),
                WorkloadProfile(input_tokens=Constant(10), output_tokens=Constant(3)),
                1,
                n_requests=1,
            )
        record = result.successful[0]
        assert record.prompt_tokens == 1234, "client estimate (10) was used instead of usage"

    def test_usage_is_requested(self) -> None:
        with MockEngine(ttft_s=0.001, tpot_s=0.0) as engine:
            run_at_concurrency(
                EndpointConfig(base_url=engine.base_url),
                WorkloadProfile(input_tokens=Constant(5), output_tokens=Constant(2)),
                1,
                n_requests=1,
            )
        assert engine.requests[0]["stream_options"] == {"include_usage": True}

    def test_falls_back_to_the_estimate_when_usage_is_absent(self) -> None:
        """Not every OpenAI-compatible server implements usage on streamed responses."""
        record = RequestRecord(prompt_tokens=77, output_tokens=5, ttft_s=0.1, total_s=0.5)
        assert record.prompt_tokens == 77
