"""Configuration space: enumerate, then prune analytically.

The expensive resource is **engine boots**, not candidates. Enumerating a few thousand
configurations costs microseconds; booting one costs minutes. So the space is enumerated
generously and then cut down by two cheap filters before anything is launched:

* **Feasibility** — does the memory ledger leave a usable KV budget at all? This removes
  most of the space outright, and it is the filter the estimator exists to provide.
* **Dominance** — is another surviving candidate at least as good on every axis? A candidate
  with less KV capacity *and* a lower predicted throughput can never win, so booting it is
  wasted time.

Only survivors reach the outer loop. In practice this takes thousands of candidates to single
digits, which is what makes the search affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from ..core.dtypes import DType
from ..core.gpu import GPUProfile
from ..core.model import ModelProfile
from ..core.plan import Parallelism, ResourcePlan
from ..core.workload import WorkloadProfile
from ..estimator.memory import InfeasibleConfigurationError, estimate_plan
from ..estimator.roofline import RooflineCoefficients, predict


@dataclass(frozen=True, slots=True)
class Candidate:
    """One configuration to consider.

    Split by whether changing it requires a restart, because that is what determines cost:
    everything here is restart-required, and concurrency (which is not here) is not.
    """

    tensor_parallel: int = 1
    max_model_len: int = 8192
    max_num_seqs: int = 32
    kv_dtype: DType = DType.BF16
    enforce_eager: bool = False

    def label(self) -> str:
        return (
            f"tp={self.tensor_parallel} len={self.max_model_len} "
            f"seqs={self.max_num_seqs} kv={self.kv_dtype}"
            + (" eager" if self.enforce_eager else "")
        )


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """A candidate with its analytically predicted properties."""

    candidate: Candidate
    plan: ResourcePlan
    kv_tokens: int
    predicted_throughput: float
    predicted_ttft_s: float
    predicted_tpot_s: float
    headroom_concurrency: int | None

    @property
    def dominates(self) -> tuple[float, ...]:
        """Axes on which dominance is judged, all higher-is-better."""
        return (self.kv_tokens, self.predicted_throughput)


@dataclass(frozen=True, slots=True)
class PruneReport:
    """What the analytic filters removed, and why."""

    enumerated: int
    infeasible: int
    dominated: int
    survivors: tuple[ScoredCandidate, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reduction_factor(self) -> float:
        return self.enumerated / len(self.survivors) if self.survivors else float("inf")


def enumerate_candidates(
    gpu: GPUProfile,
    *,
    max_model_lens: tuple[int, ...] = (4096, 8192, 16384),
    max_num_seqs_options: tuple[int, ...] = (16, 32, 64, 128, 256),
    kv_dtypes: tuple[DType, ...] = (DType.BF16, DType.FP8_E4M3),
    tensor_parallel_options: tuple[int, ...] | None = None,
    include_eager: bool = False,
) -> tuple[Candidate, ...]:
    """Enumerate the restart-required configuration space.

    Tensor-parallel degrees are limited to powers of two that divide the available GPU count,
    since anything else either cannot be scheduled or shards badly. KV dtypes the device cannot
    execute are dropped here rather than failing later.
    """
    if tensor_parallel_options is None:
        tensor_parallel_options = tuple(
            tp for tp in (1, 2, 4, 8) if tp <= gpu.count and gpu.count % tp == 0
        ) or (1,)

    usable_kv = tuple(d for d in kv_dtypes if not d.is_float8 or gpu.supports(d))
    eager_options = (False, True) if include_eager else (False,)

    return tuple(
        Candidate(
            tensor_parallel=tp,
            max_model_len=length,
            max_num_seqs=seqs,
            kv_dtype=dtype,
            enforce_eager=eager,
        )
        for tp, length, seqs, dtype, eager in product(
            tensor_parallel_options,
            max_model_lens,
            max_num_seqs_options,
            usable_kv or (DType.BF16,),
            eager_options,
        )
    )


def score(
    candidate: Candidate,
    model: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    *,
    coefficients: RooflineCoefficients | None = None,
    samples: int = 400,
) -> ScoredCandidate | None:
    """Predict a candidate's properties, or ``None`` when it cannot fit."""
    try:
        plan = estimate_plan(
            model,
            gpu,
            workload,
            max_num_seqs=candidate.max_num_seqs,
            max_model_len=candidate.max_model_len,
            parallelism=Parallelism(tensor=candidate.tensor_parallel),
            kv_dtype=candidate.kv_dtype,
            enforce_eager=candidate.enforce_eager,
            samples=samples,
        )
    except (InfeasibleConfigurationError, ValueError):
        return None

    perf = predict(
        model,
        gpu,
        plan,
        batch=candidate.max_num_seqs,
        prompt_tokens=int(workload.input_tokens.percentile(0.5)),
        avg_context_tokens=int(workload.input_tokens.mean() + workload.output_tokens.mean() / 2),
        coefficients=coefficients,
    )
    return ScoredCandidate(
        candidate=candidate,
        plan=plan,
        kv_tokens=plan.kv_budget_tokens,
        predicted_throughput=perf.output_throughput_tokens_s.value,
        predicted_ttft_s=perf.ttft_s.value,
        predicted_tpot_s=perf.tpot_s.value,
        headroom_concurrency=plan.headroom_concurrency,
    )


def score_all(
    candidates: tuple[Candidate, ...],
    model: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    *,
    coefficients: RooflineCoefficients | None = None,
    samples: int = 400,
) -> tuple[list[ScoredCandidate], int]:
    """Score every candidate, returning the feasible ones and the infeasible count.

    Separated from :func:`prune` so an exhaustive baseline can boot *everything feasible*
    without inheriting the dominance filter — otherwise the "grid search" it is measured
    against would not actually be a grid search.
    """
    scored: list[ScoredCandidate] = []
    infeasible = 0
    for candidate in candidates:
        result = score(candidate, model, gpu, workload, coefficients=coefficients, samples=samples)
        if result is None:
            infeasible += 1
        else:
            scored.append(result)
    return scored, infeasible


def prune(
    candidates: tuple[Candidate, ...],
    model: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    *,
    coefficients: RooflineCoefficients | None = None,
    keep: int = 8,
    samples: int = 400,
) -> PruneReport:
    """Drop infeasible and dominated candidates, keeping at most ``keep`` to boot.

    ``keep`` exists because the outer loop's cost is linear in survivors, and the M4 budget is
    a dozen boots. Survivors are chosen to span the space rather than cluster: the Pareto
    frontier by construction contains the best candidate on each axis.
    """
    reasons: list[str] = []
    scored, infeasible = score_all(
        candidates, model, gpu, workload, coefficients=coefficients, samples=samples
    )

    if not scored:
        return PruneReport(
            enumerated=len(candidates),
            infeasible=infeasible,
            dominated=0,
            reasons=("every candidate was infeasible: no KV budget fits on this GPU",),
        )

    frontier = pareto_front(scored)
    dominated = len(scored) - len(frontier)
    reasons.append(
        f"{infeasible} infeasible (no KV budget), {dominated} dominated "
        f"(another candidate is at least as good on KV capacity and predicted throughput)"
    )

    # Prefer higher predicted throughput when trimming to the boot budget.
    survivors = sorted(frontier, key=lambda s: s.predicted_throughput, reverse=True)[:keep]
    if len(frontier) > keep:
        reasons.append(
            f"kept the {keep} highest-throughput frontier points of {len(frontier)} "
            "to stay within the boot budget"
        )
    return PruneReport(
        enumerated=len(candidates),
        infeasible=infeasible,
        dominated=dominated,
        survivors=tuple(survivors),
        reasons=tuple(reasons),
    )


def pareto_front(scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Candidates not dominated on every axis by some other candidate."""
    front: list[ScoredCandidate] = []
    for a in scored:
        if not any(
            b is not a
            and all(bv >= av for av, bv in zip(a.dominates, b.dominates, strict=True))
            and any(bv > av for av, bv in zip(a.dominates, b.dominates, strict=True))
            for b in scored
        ):
            front.append(a)
    return front


__all__ = [
    "Candidate",
    "PruneReport",
    "ScoredCandidate",
    "enumerate_candidates",
    "pareto_front",
    "prune",
    "score",
]
