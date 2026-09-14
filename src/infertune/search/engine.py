"""Search orchestration: analytic prune, then a boot budget spent carefully.

The cost model, from ``docs/plan.md`` §2.4, drives the whole structure:

* **Outer loop** — anything needing a restart (tensor parallelism, KV dtype, context length).
  Minutes each. Few candidates, chosen analytically.
* **Inner loop** — client-side concurrency. Seconds each, no restart. Swept densely.

So one boot yields a whole latency-throughput curve, and the search's budget is counted in
*boots*, not in configurations evaluated.

Booting is injected as a callable rather than performed here. That keeps the strategy testable
against a synthetic objective — which is how the "within 10% of a 50-point grid search using
≤12 boots" criterion is checked without renting a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..bench.metrics import SweepResult
from ..core.gpu import GPUProfile
from ..core.model import ModelProfile
from ..core.workload import SLA, WorkloadProfile
from ..estimator.roofline import RooflineCoefficients
from .space import (
    Candidate,
    PruneReport,
    ScoredCandidate,
    enumerate_candidates,
    prune,
    score_all,
)

BootAndMeasure = Callable[[ScoredCandidate], SweepResult]
"""Boot an engine for a candidate and return its concurrency sweep."""


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One candidate, actually measured."""

    scored: ScoredCandidate
    sweep: SweepResult

    @property
    def best_throughput(self) -> float:
        best = self.sweep.best_throughput
        return best.output_throughput_tokens_s if best else 0.0

    def throughput_within_sla(self, sla: SLA) -> float:
        if sla.is_unconstrained:
            return self.best_throughput
        knee = self.sweep.knee(ttft_p99_ms=sla.ttft_p99_ms, tpot_p99_ms=sla.tpot_p99_ms)
        return knee.output_throughput_tokens_s if knee else 0.0

    def best_concurrency(self, sla: SLA) -> int | None:
        if sla.is_unconstrained:
            best = self.sweep.best_throughput
            return best.concurrency if best else None
        knee = self.sweep.knee(ttft_p99_ms=sla.ttft_p99_ms, tpot_p99_ms=sla.tpot_p99_ms)
        return knee.concurrency if knee else None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Outcome of a search, including what it cost."""

    best: Evaluation | None
    evaluations: tuple[Evaluation, ...]
    prune_report: PruneReport
    boots_used: int
    sla: SLA
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def best_candidate(self) -> Candidate | None:
        return self.best.scored.candidate if self.best else None

    @property
    def best_throughput(self) -> float:
        return self.best.throughput_within_sla(self.sla) if self.best else 0.0

    @property
    def prediction_error(self) -> float | None:
        """How far the analytic prediction was from the measurement, for the winner.

        Reported because a large gap means the roofline coefficients need recalibrating, not
        that the search failed.
        """
        if self.best is None:
            return None
        predicted = self.best.scored.predicted_throughput
        measured = self.best.throughput_within_sla(self.sla)
        if measured <= 0:
            return None
        return abs(predicted - measured) / measured


def search(
    model: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    boot_and_measure: BootAndMeasure,
    *,
    sla: SLA | None = None,
    boot_budget: int = 12,
    coefficients: RooflineCoefficients | None = None,
    candidates: tuple[Candidate, ...] | None = None,
    prune_samples: int = 400,
) -> SearchResult:
    """Find a good configuration inside a boot budget.

    Strategy, in order of cost:

    1. Enumerate the restart-required space (free).
    2. Prune analytically to the Pareto frontier (free).
    3. Boot the survivors, best-predicted first, and sweep concurrency on each (expensive).
    4. Pick the highest measured throughput that satisfies the SLA.

    Booting best-predicted-first matters: if the budget runs out early, what has been spent is
    spent on the most promising candidates rather than an arbitrary prefix.
    """
    sla = sla or SLA()
    if boot_budget < 1:
        raise ValueError(f"boot_budget must be >= 1, got {boot_budget}")

    pool = candidates if candidates is not None else enumerate_candidates(gpu)
    report = prune(
        pool,
        model,
        gpu,
        workload,
        coefficients=coefficients,
        keep=boot_budget,
        samples=prune_samples,
    )
    notes: list[str] = list(report.reasons)
    if not report.survivors:
        return SearchResult(
            best=None,
            evaluations=(),
            prune_report=report,
            boots_used=0,
            sla=sla,
            notes=(*notes, "nothing feasible to boot"),
        )

    notes.append(
        f"analytic pruning took {report.enumerated} candidates to "
        f"{len(report.survivors)} boots ({report.reduction_factor:.0f}x fewer)"
    )

    evaluations: list[Evaluation] = []
    for scored in report.survivors[:boot_budget]:
        sweep = boot_and_measure(scored)
        evaluations.append(Evaluation(scored=scored, sweep=sweep))

    usable = [e for e in evaluations if e.throughput_within_sla(sla) > 0]
    if not usable:
        return SearchResult(
            best=None,
            evaluations=tuple(evaluations),
            prune_report=report,
            boots_used=len(evaluations),
            sla=sla,
            notes=(
                *notes,
                "no configuration satisfied the SLA; relax it or reduce max_model_len",
            ),
        )

    best = max(usable, key=lambda e: e.throughput_within_sla(sla))
    return SearchResult(
        best=best,
        evaluations=tuple(evaluations),
        prune_report=report,
        boots_used=len(evaluations),
        sla=sla,
        notes=tuple(notes),
    )


def grid_search(
    model: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    boot_and_measure: BootAndMeasure,
    *,
    sla: SLA | None = None,
    candidates: tuple[Candidate, ...] | None = None,
    prune_samples: int = 200,
) -> SearchResult:
    """Exhaustive baseline: boot every feasible candidate.

    Exists to quantify what the pruned search gives up. Far too expensive to run for real —
    which is the point of measuring against it once.
    """
    sla = sla or SLA()
    pool = candidates if candidates is not None else enumerate_candidates(gpu)
    # Bypasses the dominance filter deliberately: a grid search boots everything feasible,
    # otherwise it is not the exhaustive baseline the pruned search is measured against.
    feasible, infeasible = score_all(pool, model, gpu, workload, samples=prune_samples)
    report = PruneReport(
        enumerated=len(pool),
        infeasible=infeasible,
        dominated=0,
        survivors=tuple(feasible),
        reasons=("exhaustive: dominance filter deliberately not applied",),
    )
    evaluations = [Evaluation(scored=s, sweep=boot_and_measure(s)) for s in report.survivors]
    usable = [e for e in evaluations if e.throughput_within_sla(sla) > 0]
    best = max(usable, key=lambda e: e.throughput_within_sla(sla)) if usable else None
    return SearchResult(
        best=best,
        evaluations=tuple(evaluations),
        prune_report=report,
        boots_used=len(evaluations),
        sla=sla,
        notes=("exhaustive grid over feasible candidates",),
    )


__all__ = [
    "BootAndMeasure",
    "Evaluation",
    "SearchResult",
    "grid_search",
    "search",
]
