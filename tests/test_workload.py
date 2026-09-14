"""Workload and working-set tests.

The central assertion here is methodological: composing per-request p95 lengths into an
aggregate p95 overstates the tail, because summing independent sequences concentrates
the total. That overstatement is what makes naive profilers recommend needlessly
conservative configurations.
"""

from __future__ import annotations

import random

import pytest

from infertune.core.dtypes import DType
from infertune.core.units import gib
from infertune.core.workload import (
    SLA,
    Constant,
    Empirical,
    LogNormal,
    WorkloadProfile,
)
from tests.test_kv_cache import LLAMA_31_8B


def test_constant_distribution() -> None:
    dist = Constant(1024)
    assert dist.mean() == 1024
    assert dist.percentile(0.5) == 1024
    assert dist.percentile(0.99) == 1024


def test_lognormal_from_median_p95_recovers_its_inputs() -> None:
    dist = LogNormal.from_median_p95(median=1024, p95=4096)
    assert dist.percentile(0.5) == pytest.approx(1024, rel=1e-6)
    assert dist.percentile(0.95) == pytest.approx(4096, rel=1e-6)


def test_lognormal_is_right_skewed() -> None:
    """Mean above median is the property that makes tail behaviour matter."""
    dist = LogNormal.from_median_p95(median=1024, p95=4096)
    assert dist.mean() > dist.percentile(0.5)


def test_lognormal_rejects_p95_below_median() -> None:
    with pytest.raises(ValueError, match="must exceed median"):
        LogNormal.from_median_p95(median=1024, p95=512)


def test_empirical_percentile_and_mean() -> None:
    dist = Empirical(tuple(range(1, 101)))
    assert dist.mean() == pytest.approx(50.5)
    assert dist.percentile(0.5) == pytest.approx(50, abs=1)
    assert dist.percentile(0.99) == pytest.approx(99, abs=1)


def test_empirical_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        Empirical(())


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.5])
def test_percentile_rejects_out_of_range_quantiles(q: float) -> None:
    with pytest.raises(ValueError, match="quantile"):
        Constant(10).percentile(q)


def test_distributions_are_reproducible_under_a_seed() -> None:
    dist = LogNormal.from_median_p95(1024, 4096)
    first = [dist.sample(random.Random(7)) for _ in range(3)]
    second = [dist.sample(random.Random(7)) for _ in range(3)]
    assert first == second


def test_working_set_scales_with_concurrency() -> None:
    workload = WorkloadProfile(input_tokens=Constant(1000), output_tokens=Constant(200))
    small = workload.working_set(10, samples=2000)
    large = workload.working_set(40, samples=2000)
    assert large.p50_tokens > small.p50_tokens * 3.5


def test_working_set_counts_half_the_output_on_average() -> None:
    """A request observed mid-decode has produced part of its output, not all of it."""
    workload = WorkloadProfile(input_tokens=Constant(1000), output_tokens=Constant(200))
    result = workload.working_set(100, samples=4000)
    expected_mean = 100 * (1000 + 200 / 2)
    assert result.mean_tokens == pytest.approx(expected_mean, rel=0.02)


def test_naive_per_request_p95_overstates_the_aggregate_tail() -> None:
    """The methodological finding, asserted.

    With heavy-tailed lengths at realistic concurrency, composing per-request p95s
    overstates the true aggregate p95 substantially. Sizing KV against the naive figure
    wastes a large fraction of the cache.
    """
    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 2048),
        output_tokens=LogNormal.from_median_p95(256, 512),
        target_concurrency=24,
    )
    result = workload.working_set(samples=4000)

    assert result.p95_tokens > result.p50_tokens
    assert result.naive_p95_tokens > result.p95_tokens
    assert result.naive_overstatement > 1.5


def test_overstatement_grows_with_concurrency() -> None:
    """Concentration strengthens as more sequences are summed, so the naive error grows."""
    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 2048),
        output_tokens=LogNormal.from_median_p95(256, 512),
    )
    low = workload.working_set(4, samples=4000)
    high = workload.working_set(64, samples=4000)
    assert high.naive_overstatement > low.naive_overstatement


def test_working_set_is_deterministic_for_a_fixed_seed() -> None:
    """Recommendations must be reproducible; a Monte Carlo estimate must not drift."""
    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 2048),
        output_tokens=Constant(256),
        target_concurrency=16,
    )
    first = workload.working_set(samples=1500, seed=42)
    second = workload.working_set(samples=1500, seed=42)
    assert first == second


def test_percentiles_are_ordered() -> None:
    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 4096),
        output_tokens=LogNormal.from_median_p95(256, 1024),
    )
    result = workload.working_set(32, samples=4000)
    assert result.p50_tokens <= result.p95_tokens <= result.p99_tokens


def test_working_set_requires_a_concurrency() -> None:
    workload = WorkloadProfile(input_tokens=Constant(1024), output_tokens=Constant(256))
    with pytest.raises(ValueError, match="needs a concurrency"):
        workload.working_set()


def test_working_set_uses_target_concurrency_by_default() -> None:
    workload = WorkloadProfile(
        input_tokens=Constant(1024), output_tokens=Constant(256), target_concurrency=12
    )
    assert workload.working_set(samples=500).concurrency == 12


KV_BUDGET_4090 = gib(4.98)
"""Llama-3.1-8B on a 24 GiB RTX 4090, from the README ledger."""


def test_readme_example_fits_in_bf16_with_headroom() -> None:
    """End-to-end reproduction of the README's worked example.

    Two independent errors in the original plan compounded here, both conservative:
    reading "24 GB" as decimal GB understated the KV budget by 32%, and composing
    per-request p95 lengths overstated the aggregate tail by 1.78x. Together they turned
    "fits at p99 with room to grow" into "preempts, switch to fp8" — which would have
    traded away quality for no reason.
    """
    per_token_bf16 = LLAMA_31_8B.kv_bytes_per_token(DType.BF16)
    per_token_fp8 = LLAMA_31_8B.kv_bytes_per_token(DType.FP8_E4M3)

    capacity_bf16 = KV_BUDGET_4090 // per_token_bf16
    capacity_fp8 = KV_BUDGET_4090 // per_token_fp8
    assert capacity_bf16 == 40_796
    assert capacity_fp8 == pytest.approx(2 * capacity_bf16, abs=2)

    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 2048),
        output_tokens=LogNormal.from_median_p95(256, 512),
        target_concurrency=24,
        sla=SLA(ttft_p99_ms=500),
    )
    result = workload.working_set(samples=20_000)

    assert result.p50_tokens == pytest.approx(30_092, rel=0.02)
    assert result.p95_tokens == pytest.approx(34_476, rel=0.02)
    assert result.p99_tokens < capacity_bf16, "bf16 must clear the p99 tail; no fp8 needed"
    assert result.naive_overstatement == pytest.approx(1.78, abs=0.05)


def test_decimal_gb_bug_would_have_forced_an_unnecessary_mitigation() -> None:
    """Guards against regressing to decimal VRAM units.

    Under the buggy 3.77 GiB budget the p99 tail does not fit, so a recommender would
    reach for fp8 KV or a shorter context. Under the correct 4.98 GiB budget it does.
    """
    per_token = LLAMA_31_8B.kv_bytes_per_token(DType.BF16)
    buggy_capacity = gib(3.77) // per_token
    correct_capacity = KV_BUDGET_4090 // per_token

    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 2048),
        output_tokens=LogNormal.from_median_p95(256, 512),
        target_concurrency=24,
    )
    result = workload.working_set(samples=20_000)

    assert result.p99_tokens > buggy_capacity, "the bug made a fitting config look infeasible"
    assert result.p99_tokens < correct_capacity
    assert correct_capacity - buggy_capacity == 9_913
    assert correct_capacity / buggy_capacity == pytest.approx(1.32, abs=0.01)


def test_bf16_headroom_runs_out_before_the_compute_knee() -> None:
    """Why the binding constraint is still KV, not compute.

    The 4090's critical batch size is ~51, but bf16 KV runs out around 28-32 concurrent
    requests. So concurrency is limited by cache, not by the GPU — and fp8 KV, not a
    bigger GPU, is the lever that reaches the compute knee.
    """
    per_token_bf16 = LLAMA_31_8B.kv_bytes_per_token(DType.BF16)
    per_token_fp8 = LLAMA_31_8B.kv_bytes_per_token(DType.FP8_E4M3)
    capacity_bf16 = KV_BUDGET_4090 // per_token_bf16
    capacity_fp8 = KV_BUDGET_4090 // per_token_fp8

    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 2048),
        output_tokens=LogNormal.from_median_p95(256, 512),
    )

    assert workload.working_set(28, samples=6000).p95_tokens < capacity_bf16
    assert workload.working_set(32, samples=6000).p95_tokens > capacity_bf16
    # fp8 KV carries concurrency up to the compute knee (~51).
    assert workload.working_set(48, samples=6000).p95_tokens < capacity_fp8


def test_sla_validation() -> None:
    assert SLA().is_unconstrained
    assert not SLA(ttft_p99_ms=200).is_unconstrained
    with pytest.raises(ValueError, match="ttft_p99_ms"):
        SLA(ttft_p99_ms=0)


def test_workload_validates_inputs() -> None:
    with pytest.raises(ValueError, match="target_concurrency"):
        WorkloadProfile(input_tokens=Constant(1), output_tokens=Constant(1), target_concurrency=0)
    with pytest.raises(ValueError, match="target_rps"):
        WorkloadProfile(input_tokens=Constant(1), output_tokens=Constant(1), target_rps=0)
    with pytest.raises(ValueError, match="shared_prefix_tokens"):
        WorkloadProfile(
            input_tokens=Constant(1), output_tokens=Constant(1), shared_prefix_tokens=-1
        )
