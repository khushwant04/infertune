"""Configuration search.

The acceptance criterion — *within 10% of a 50-point grid search using ≤12 boots* — is checked
here against a **synthetic objective**. That is deliberate: the search strategy is a property
of the algorithm, not of any particular GPU, so validating it needs a known ground truth rather
than hardware. If the strategy were unsound, this fails on a laptop instead of after an
afternoon of renting GPUs.

The synthetic engine is intentionally awkward: throughput saturates, fp8 buys capacity but
costs latency, and long contexts cost throughput. A strategy that simply picked the largest
``max_num_seqs`` would not win.
"""

from __future__ import annotations

import pytest

from infertune.bench.metrics import BenchmarkResult, RequestRecord, SweepResult
from infertune.core.dtypes import DType
from infertune.core.workload import SLA, Constant, LogNormal, WorkloadProfile
from infertune.estimator.roofline import RooflineCoefficients
from infertune.hardware import specdb
from infertune.search import (
    Candidate,
    ScoredCandidate,
    enumerate_candidates,
    grid_search,
    pareto_front,
    prune,
    score,
    search,
)
from tests.test_estimator import LLAMA_31_8B

WORKLOAD = WorkloadProfile(
    input_tokens=LogNormal.from_median_p95(1024, 2048),
    output_tokens=LogNormal.from_median_p95(256, 512),
)
GPU = specdb.load("h100-sxm")


# --------------------------------------------------------------------------- synthetic engine


def _sweep_from_throughput(
    throughput: float, concurrency: int, ttft_ms: float, tpot_ms: float
) -> SweepResult:
    """Build a SweepResult whose aggregate metrics match the requested figures."""
    # duration chosen so output_tokens / duration == throughput
    tokens_per_request = 64
    n = max(1, concurrency)
    total_tokens = tokens_per_request * n
    duration = total_tokens / throughput if throughput > 0 else 1.0
    records = tuple(
        RequestRecord(
            prompt_tokens=1024,
            output_tokens=tokens_per_request,
            ttft_s=ttft_ms / 1000.0,
            total_s=ttft_ms / 1000.0 + (tokens_per_request - 1) * tpot_ms / 1000.0,
        )
        for _ in range(n)
    )
    point = BenchmarkResult(concurrency=n, duration_s=duration, records=records)
    return SweepResult(points=(point,))


def true_throughput(c: Candidate) -> float:
    """Ground-truth objective the search must find the maximum of.

    Shaped to punish naive strategies: throughput saturates in concurrency, fp8 helps only
    because it permits more concurrency, and long contexts cost throughput.
    """
    saturating = 4000.0 * c.max_num_seqs / (c.max_num_seqs + 48)
    context_penalty = 1.0 - 0.15 * (c.max_model_len / 16384)
    kv_bonus = 1.10 if c.kv_dtype.is_float8 else 1.0
    eager_penalty = 0.80 if c.enforce_eager else 1.0
    return saturating * context_penalty * kv_bonus * eager_penalty


def synthetic_boot(scored: ScoredCandidate) -> SweepResult:
    """A fake engine boot: returns the ground-truth throughput for this candidate."""
    c = scored.candidate
    throughput = true_throughput(c)
    # fp8 trades a little latency for its capacity gain.
    tpot_ms = 8.0 + (2.0 if c.kv_dtype.is_float8 else 0.0) + c.max_num_seqs * 0.02
    ttft_ms = 60.0 + c.max_model_len / 400.0
    return _sweep_from_throughput(throughput, c.max_num_seqs, ttft_ms, tpot_ms)


# ------------------------------------------------------------------------------------- tests


class TestEnumeration:
    def test_enumerates_a_large_but_finite_space(self) -> None:
        candidates = enumerate_candidates(specdb.load("h100-sxm", count=4))
        assert len(candidates) > 50
        assert len({c.label() for c in candidates}) == len(candidates), "no duplicates"

    def test_tensor_parallel_options_respect_the_gpu_count(self) -> None:
        single = enumerate_candidates(specdb.load("h100-sxm"))
        assert {c.tensor_parallel for c in single} == {1}
        quad = enumerate_candidates(specdb.load("h100-sxm", count=4))
        assert {c.tensor_parallel for c in quad} == {1, 2, 4}

    def test_unsupported_kv_dtypes_are_dropped_up_front(self) -> None:
        """The A100 has no fp8; offering it would only fail later."""
        a100 = enumerate_candidates(specdb.load("a100-80gb"))
        assert all(not c.kv_dtype.is_float8 for c in a100)
        h100 = enumerate_candidates(specdb.load("h100-sxm"))
        assert any(c.kv_dtype.is_float8 for c in h100)


class TestScoringAndPruning:
    def test_infeasible_candidates_score_as_none(self) -> None:
        """A 16 GB model cannot fit a T4 at any context length."""
        result = score(
            Candidate(max_model_len=8192, max_num_seqs=32),
            LLAMA_31_8B,
            specdb.load("t4"),
            WORKLOAD,
            samples=100,
        )
        assert result is None

    def test_pruning_reduces_the_space_by_an_order_of_magnitude(self) -> None:
        candidates = enumerate_candidates(GPU)
        report = prune(candidates, LLAMA_31_8B, GPU, WORKLOAD, keep=8, samples=100)
        assert report.enumerated == len(candidates)
        assert 0 < len(report.survivors) <= 8
        assert report.reduction_factor > 3
        assert report.reasons

    def test_pruning_reports_why_candidates_were_dropped(self) -> None:
        report = prune(enumerate_candidates(GPU), LLAMA_31_8B, GPU, WORKLOAD, keep=4, samples=100)
        joined = " ".join(report.reasons)
        assert "infeasible" in joined
        assert "dominated" in joined

    def test_everything_infeasible_is_reported_not_crashed(self) -> None:
        report = prune(
            enumerate_candidates(specdb.load("t4")),
            LLAMA_31_8B,
            specdb.load("t4"),
            WORKLOAD,
            samples=60,
        )
        assert not report.survivors
        assert any("infeasible" in r for r in report.reasons)


class TestParetoFront:
    def _scored(self, kv: int, tps: float) -> ScoredCandidate:
        plan = score(Candidate(), LLAMA_31_8B, GPU, WORKLOAD, samples=60)
        assert plan is not None
        return ScoredCandidate(
            candidate=Candidate(),
            plan=plan.plan,
            kv_tokens=kv,
            predicted_throughput=tps,
            predicted_ttft_s=0.1,
            predicted_tpot_s=0.01,
            headroom_concurrency=32,
        )

    def test_dominated_points_are_excluded(self) -> None:
        good = self._scored(kv=1000, tps=100)
        worse = self._scored(kv=500, tps=50)  # worse on both axes
        front = pareto_front([good, worse])
        assert good in front
        assert worse not in front

    def test_tradeoffs_are_both_kept(self) -> None:
        more_kv = self._scored(kv=2000, tps=50)
        more_tps = self._scored(kv=500, tps=200)
        front = pareto_front([more_kv, more_tps])
        assert len(front) == 2, "neither dominates the other, so both must survive"

    def test_empty_input(self) -> None:
        assert pareto_front([]) == []


class TestSearch:
    def test_finds_a_configuration_and_reports_its_cost(self) -> None:
        result = search(
            LLAMA_31_8B,
            GPU,
            WORKLOAD,
            synthetic_boot,
            boot_budget=8,
            prune_samples=100,
        )
        assert result.best is not None
        assert result.boots_used <= 8
        assert result.best_throughput > 0
        assert any("analytic pruning took" in n for n in result.notes)

    def test_never_exceeds_the_boot_budget(self) -> None:
        """The budget is the whole point: boots are the expensive resource."""
        boots: list[str] = []

        def counting_boot(scored: ScoredCandidate) -> SweepResult:
            boots.append(scored.candidate.label())
            return synthetic_boot(scored)

        search(LLAMA_31_8B, GPU, WORKLOAD, counting_boot, boot_budget=5, prune_samples=100)
        assert len(boots) <= 5

    def test_boots_the_best_predicted_candidates_first(self) -> None:
        """If the budget runs out, it should have been spent on promising candidates."""
        order: list[float] = []

        def recording_boot(scored: ScoredCandidate) -> SweepResult:
            order.append(scored.predicted_throughput)
            return synthetic_boot(scored)

        search(LLAMA_31_8B, GPU, WORKLOAD, recording_boot, boot_budget=6, prune_samples=100)
        assert order == sorted(order, reverse=True)

    def test_respects_an_sla(self) -> None:
        strict = SLA(tpot_p99_ms=9.0)  # only small-concurrency configs can satisfy this
        result = search(
            LLAMA_31_8B,
            GPU,
            WORKLOAD,
            synthetic_boot,
            sla=strict,
            boot_budget=8,
            prune_samples=100,
        )
        if result.best is not None:
            knee = result.best.sweep.knee(ttft_p99_ms=None, tpot_p99_ms=9.0)
            assert knee is not None

    def test_reports_when_nothing_satisfies_the_sla(self) -> None:
        impossible = SLA(ttft_p99_ms=0.001)
        result = search(
            LLAMA_31_8B,
            GPU,
            WORKLOAD,
            synthetic_boot,
            sla=impossible,
            boot_budget=4,
            prune_samples=100,
        )
        assert result.best is None
        assert any("no configuration satisfied the SLA" in n for n in result.notes)

    def test_infeasible_hardware_spends_no_boots(self) -> None:
        boots: list[int] = []

        def boot(scored: ScoredCandidate) -> SweepResult:
            boots.append(1)
            return synthetic_boot(scored)

        result = search(
            LLAMA_31_8B,
            specdb.load("t4"),
            WORKLOAD,
            boot,
            boot_budget=12,
            prune_samples=60,
        )
        assert result.best is None
        assert result.boots_used == 0
        assert not boots, "analytic pruning must prevent doomed boots"

    def test_reports_prediction_error_for_the_winner(self) -> None:
        """A large gap means the coefficients need recalibrating, not that search failed."""
        result = search(
            LLAMA_31_8B,
            GPU,
            WORKLOAD,
            synthetic_boot,
            boot_budget=6,
            prune_samples=100,
        )
        assert result.prediction_error is not None
        assert result.prediction_error >= 0

    def test_rejects_a_nonsense_budget(self) -> None:
        with pytest.raises(ValueError, match="boot_budget"):
            search(LLAMA_31_8B, GPU, WORKLOAD, synthetic_boot, boot_budget=0)


class TestAcceptanceCriterion:
    """M4: within 10% of a 50-point grid search's best, using at most 12 boots."""

    def test_search_matches_grid_search_within_10_percent_on_12_boots(self) -> None:
        candidates = enumerate_candidates(GPU)[:60]

        grid = grid_search(
            LLAMA_31_8B,
            GPU,
            WORKLOAD,
            synthetic_boot,
            candidates=candidates,
            prune_samples=60,
        )
        assert grid.boots_used >= 20, "the baseline must actually be exhaustive"
        assert grid.best is not None

        pruned = search(
            LLAMA_31_8B,
            GPU,
            WORKLOAD,
            synthetic_boot,
            candidates=candidates,
            boot_budget=12,
            prune_samples=60,
        )
        assert pruned.best is not None
        assert pruned.boots_used <= 12

        gap = (grid.best_throughput - pruned.best_throughput) / grid.best_throughput
        assert gap <= 0.10, (
            f"pruned search gave up {gap:.1%} of the exhaustive best using "
            f"{pruned.boots_used} boots instead of {grid.boots_used}"
        )

    def test_search_is_far_cheaper_than_the_baseline(self) -> None:
        candidates = enumerate_candidates(GPU)[:60]
        grid = grid_search(
            LLAMA_31_8B,
            GPU,
            WORKLOAD,
            synthetic_boot,
            candidates=candidates,
            prune_samples=60,
        )
        pruned = search(
            LLAMA_31_8B,
            GPU,
            WORKLOAD,
            synthetic_boot,
            candidates=candidates,
            boot_budget=12,
            prune_samples=60,
        )
        assert pruned.boots_used < grid.boots_used / 2

    def test_the_synthetic_objective_is_not_trivially_maximised(self) -> None:
        """Guards the test itself: a naive 'largest max_num_seqs' rule must not be optimal."""
        biggest_seqs = Candidate(max_num_seqs=256, max_model_len=16384, kv_dtype=DType.BF16)
        better = Candidate(max_num_seqs=256, max_model_len=4096, kv_dtype=DType.FP8_E4M3)
        assert true_throughput(better) > true_throughput(biggest_seqs)


def test_prediction_uses_calibrated_coefficients_when_supplied() -> None:
    calibrated = RooflineCoefficients(mfu=0.6, mbu=0.85, samples=50, source="calibrated")
    a = score(Candidate(), LLAMA_31_8B, GPU, WORKLOAD, samples=80)
    b = score(Candidate(), LLAMA_31_8B, GPU, WORKLOAD, coefficients=calibrated, samples=80)
    assert a is not None
    assert b is not None
    assert a.predicted_throughput != b.predicted_throughput


def test_constant_workload_is_supported() -> None:
    flat = WorkloadProfile(input_tokens=Constant(512), output_tokens=Constant(128))
    result = score(Candidate(), LLAMA_31_8B, GPU, flat, samples=80)
    assert result is not None
    assert result.kv_tokens > 0
