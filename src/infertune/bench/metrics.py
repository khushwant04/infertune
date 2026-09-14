"""Latency and throughput metrics for a benchmark run.

The metrics that matter for serving are not averages. A deployment is judged on tail latency,
so percentiles are first-class and the mean is reported only as context.

Two distinctions worth keeping straight, because conflating them makes results
uncomparable:

* **TTFT** (time to first token) measures the prefill path and queueing.
* **TPOT** (time per output token) measures steady-state decode, and is computed *excluding*
  the first token — otherwise prefill leaks into a decode metric and TPOT looks worse for long
  prompts for no real reason.
* **Output throughput** counts only generated tokens. Counting prompt tokens too inflates the
  figure by the input/output ratio, which is how benchmarks end up incomparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One completed (or failed) request."""

    prompt_tokens: int
    output_tokens: int
    ttft_s: float | None
    total_s: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.output_tokens > 0 and self.ttft_s is not None

    @property
    def tpot_s(self) -> float | None:
        """Mean seconds per output token after the first.

        ``None`` when the request produced a single token, where TPOT is undefined rather
        than zero.
        """
        if not self.ok or self.output_tokens < 2 or self.ttft_s is None:
            return None
        return (self.total_s - self.ttft_s) / (self.output_tokens - 1)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. ``q`` in (0, 1)."""
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[max(0, index)]


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Distribution summary for one latency metric, in milliseconds."""

    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_seconds(cls, samples: list[float]) -> LatencyStats:
        if not samples:
            return cls(0, 0, 0, 0, 0, 0, 0)
        ms = [s * 1000.0 for s in samples]
        return cls(
            count=len(ms),
            mean_ms=sum(ms) / len(ms),
            p50_ms=percentile(ms, 0.50),
            p90_ms=percentile(ms, 0.90),
            p99_ms=percentile(ms, 0.99),
            min_ms=min(ms),
            max_ms=max(ms),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate outcome of one concurrency level."""

    concurrency: int
    duration_s: float
    records: tuple[RequestRecord, ...] = field(default_factory=tuple)

    @property
    def successful(self) -> tuple[RequestRecord, ...]:
        return tuple(r for r in self.records if r.ok)

    @property
    def failed(self) -> tuple[RequestRecord, ...]:
        return tuple(r for r in self.records if not r.ok)

    @property
    def error_rate(self) -> float:
        return len(self.failed) / len(self.records) if self.records else 0.0

    @property
    def ttft(self) -> LatencyStats:
        return LatencyStats.from_seconds(
            [r.ttft_s for r in self.successful if r.ttft_s is not None]
        )

    @property
    def tpot(self) -> LatencyStats:
        return LatencyStats.from_seconds(
            [t for r in self.successful if (t := r.tpot_s) is not None]
        )

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.successful)

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.successful)

    @property
    def output_throughput_tokens_s(self) -> float:
        """Generated tokens per second. Excludes prompt tokens deliberately."""
        return self.output_tokens / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def total_throughput_tokens_s(self) -> float:
        """Prompt + generated tokens per second, for comparison with engine-reported figures."""
        if self.duration_s <= 0:
            return 0.0
        return (self.output_tokens + self.prompt_tokens) / self.duration_s

    @property
    def requests_per_s(self) -> float:
        return len(self.successful) / self.duration_s if self.duration_s > 0 else 0.0

    def meets(self, *, ttft_p99_ms: float | None, tpot_p99_ms: float | None) -> bool:
        """Whether this point satisfies a latency SLA."""
        if not self.successful:
            return False
        if ttft_p99_ms is not None and self.ttft.p99_ms > ttft_p99_ms:
            return False
        return not (tpot_p99_ms is not None and self.tpot.p99_ms > tpot_p99_ms)


@dataclass(frozen=True, slots=True)
class SweepResult:
    """A full concurrency sweep, gathered from a single engine boot.

    The point of the inner/outer split in ``docs/plan.md`` §2.4: one expensive boot yields an
    entire latency-throughput curve, so the SLA knee costs almost nothing extra to find.
    """

    points: tuple[BenchmarkResult, ...]
    model: str = ""
    engine: str = ""
    engine_version: str = ""
    gpu: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def best_throughput(self) -> BenchmarkResult | None:
        usable = [p for p in self.points if p.successful]
        return max(usable, key=lambda p: p.output_throughput_tokens_s) if usable else None

    def knee(
        self, *, ttft_p99_ms: float | None, tpot_p99_ms: float | None
    ) -> BenchmarkResult | None:
        """Highest-throughput point that still satisfies the SLA."""
        ok = [p for p in self.points if p.meets(ttft_p99_ms=ttft_p99_ms, tpot_p99_ms=tpot_p99_ms)]
        return max(ok, key=lambda p: p.output_throughput_tokens_s) if ok else None

    @property
    def saturation_concurrency(self) -> int | None:
        """Concurrency beyond which throughput stops improving materially (<2%).

        The empirical analogue of the critical batch size, and a direct check on it.
        """
        usable = sorted((p for p in self.points if p.successful), key=lambda p: p.concurrency)
        if len(usable) < 2:
            return None
        best = 0.0
        for point in usable:
            throughput = point.output_throughput_tokens_s
            if throughput < best * 1.02:
                return point.concurrency
            best = max(best, throughput)
        return usable[-1].concurrency


__all__ = [
    "BenchmarkResult",
    "LatencyStats",
    "RequestRecord",
    "SweepResult",
    "percentile",
]
