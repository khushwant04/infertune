"""Workload description and KV working-set estimation.

Why this module is required rather than optional: paged-attention engines allocate KV
cache per *token*, not per sequence slot. ``max_num_seqs`` is a scheduler admission
limit, so ``max_num_seqs * max_model_len`` describes a worst case that essentially
never occurs. The quantity that actually has to fit in VRAM is the *aggregate in-flight
token count*, and that is undefined without knowing the request length distribution.

A subtlety that is easy to get wrong: the p95 of the aggregate working set is **not**
the working set computed from per-request p95 lengths. Summing many independent
sequences concentrates the total (central limit), so using per-request p95 lengths
overstates the aggregate tail — badly, at high concurrency. This module therefore
samples the joint distribution instead of composing percentiles.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import NormalDist


class Distribution(ABC):
    """A distribution over a token count."""

    @abstractmethod
    def percentile(self, q: float) -> float:
        """Value at quantile ``q``, with ``q`` in (0, 1)."""

    @abstractmethod
    def mean(self) -> float: ...

    @abstractmethod
    def sample(self, rng: random.Random) -> float: ...

    @staticmethod
    def _check_quantile(q: float) -> None:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {q}")


@dataclass(frozen=True, slots=True)
class Constant(Distribution):
    """A fixed token count. Useful for benchmarks and for reasoning by hand."""

    value: float

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"value must be non-negative, got {self.value}")

    def percentile(self, q: float) -> float:
        self._check_quantile(q)
        return self.value

    def mean(self) -> float:
        return self.value

    def sample(self, rng: random.Random) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class LogNormal(Distribution):
    """Log-normal token lengths, parameterised by observable statistics.

    Real request lengths are right-skewed, and the shape of that tail is what decides
    whether a configuration preempts. Constructed via :meth:`from_median_p95` so
    callers supply numbers they can actually measure.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

    @classmethod
    def from_median_p95(cls, median: float, p95: float) -> LogNormal:
        if median <= 0:
            raise ValueError(f"median must be > 0, got {median}")
        if p95 <= median:
            raise ValueError(f"p95 ({p95}) must exceed median ({median})")
        mu = math.log(median)
        sigma = (math.log(p95) - mu) / NormalDist().inv_cdf(0.95)
        return cls(mu=mu, sigma=sigma)

    def percentile(self, q: float) -> float:
        self._check_quantile(q)
        return math.exp(self.mu + self.sigma * NormalDist().inv_cdf(q))

    def mean(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2)

    def sample(self, rng: random.Random) -> float:
        return math.exp(rng.gauss(self.mu, self.sigma))


@dataclass(frozen=True, slots=True)
class Empirical(Distribution):
    """Token lengths drawn from an observed trace. The most trustworthy input."""

    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("Empirical requires at least one sample")
        if any(s < 0 for s in self.samples):
            raise ValueError("samples must be non-negative")

    def percentile(self, q: float) -> float:
        self._check_quantile(q)
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
        return ordered[max(0, index)]

    def mean(self) -> float:
        return sum(self.samples) / len(self.samples)

    def sample(self, rng: random.Random) -> float:
        return rng.choice(self.samples)


@dataclass(frozen=True, slots=True)
class SLA:
    """Latency targets. ``None`` means unconstrained."""

    ttft_p99_ms: float | None = None
    tpot_p99_ms: float | None = None

    def __post_init__(self) -> None:
        for name in ("ttft_p99_ms", "tpot_p99_ms"):
            value: float | None = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 or None, got {value}")

    @property
    def is_unconstrained(self) -> bool:
        return self.ttft_p99_ms is None and self.tpot_p99_ms is None


@dataclass(frozen=True, slots=True)
class WorkingSet:
    """Aggregate in-flight KV token counts at a given concurrency."""

    concurrency: int
    p50_tokens: int
    p95_tokens: int
    p99_tokens: int
    mean_tokens: int
    naive_p95_tokens: int
    """Per-request-p95 composition, retained to show how much it overstates the tail."""

    @property
    def naive_overstatement(self) -> float:
        """Ratio of the naive tail estimate to the correct one."""
        return self.naive_p95_tokens / self.p95_tokens if self.p95_tokens else float("nan")


@dataclass(frozen=True, slots=True)
class WorkloadProfile:
    """The request population a deployment must serve."""

    input_tokens: Distribution
    output_tokens: Distribution
    target_concurrency: int | None = None
    target_rps: float | None = None
    shared_prefix_tokens: int = 0
    sla: SLA = SLA()

    def __post_init__(self) -> None:
        if self.target_concurrency is not None and self.target_concurrency < 1:
            raise ValueError(f"target_concurrency must be >= 1, got {self.target_concurrency}")
        if self.target_rps is not None and self.target_rps <= 0:
            raise ValueError(f"target_rps must be > 0, got {self.target_rps}")
        if self.shared_prefix_tokens < 0:
            raise ValueError(f"shared_prefix_tokens must be >= 0, got {self.shared_prefix_tokens}")

    def working_set(
        self,
        concurrency: int | None = None,
        *,
        samples: int = 20_000,
        seed: int = 0,
    ) -> WorkingSet:
        """Estimate aggregate in-flight KV tokens by sampling the joint distribution.

        Each in-flight request is modelled as its full prompt plus a uniformly random
        fraction of its output, which is the steady-state assumption for a request
        observed at a random instant during decode.

        Deterministic for a fixed ``seed`` so that recommendations are reproducible.
        """
        n = concurrency if concurrency is not None else self.target_concurrency
        if n is None:
            raise ValueError(
                "working_set() needs a concurrency: pass one explicitly or set "
                "target_concurrency on the WorkloadProfile"
            )
        if n < 1:
            raise ValueError(f"concurrency must be >= 1, got {n}")
        if samples < 1:
            raise ValueError(f"samples must be >= 1, got {samples}")

        rng = random.Random(seed)
        totals: list[float] = []
        for _ in range(samples):
            total = 0.0
            for _ in range(n):
                prompt = self.input_tokens.sample(rng)
                generated = self.output_tokens.sample(rng) * rng.random()
                total += prompt + generated
            totals.append(total)
        totals.sort()

        def pct(q: float) -> int:
            index = min(len(totals) - 1, math.ceil(q * len(totals)) - 1)
            return int(totals[max(0, index)])

        naive_p95 = int(
            n * (self.input_tokens.percentile(0.95) + self.output_tokens.percentile(0.95))
        )

        return WorkingSet(
            concurrency=n,
            p50_tokens=pct(0.50),
            p95_tokens=pct(0.95),
            p99_tokens=pct(0.99),
            mean_tokens=int(sum(totals) / len(totals)),
            naive_p95_tokens=naive_p95,
        )
