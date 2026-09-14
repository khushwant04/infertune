"""Concurrency sweep: the cheap inner loop.

``docs/plan.md`` §2.4 argues that what makes configuration search expensive is *engine boots*,
not the dimensionality of the space. Weight loading, ``torch.compile`` and CUDA-graph capture
cost minutes; changing client-side concurrency costs nothing. So a single boot should yield an
entire latency-throughput curve.

This module is that inner loop. It never restarts anything — it only varies how many requests
are in flight.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.workload import SLA, WorkloadProfile
from .loadgen import EndpointConfig, run_at_concurrency
from .metrics import BenchmarkResult, SweepResult

DEFAULT_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """How to walk the concurrency ladder."""

    concurrencies: tuple[int, ...] = DEFAULT_LADDER
    requests_per_point: int = 32
    warmup_requests: int = 2
    seed: int = 0
    stop_on_sla_violation: bool = True
    """Stop climbing once the SLA is breached: higher points cannot satisfy it either, since
    latency is monotone in concurrency."""

    stop_on_saturation: bool = True
    """Stop once throughput gains fall below 2%, which is the empirical critical batch size."""

    max_error_rate: float = 0.10

    def __post_init__(self) -> None:
        if not self.concurrencies:
            raise ValueError("concurrencies must be non-empty")
        if any(c < 1 for c in self.concurrencies):
            raise ValueError("concurrencies must all be >= 1")
        if self.requests_per_point < 1:
            raise ValueError("requests_per_point must be >= 1")
        if not 0 <= self.max_error_rate <= 1:
            raise ValueError("max_error_rate must be in [0, 1]")


def sweep_concurrency(
    endpoint: EndpointConfig,
    workload: WorkloadProfile,
    config: SweepConfig | None = None,
    *,
    sla: SLA | None = None,
    model: str = "",
    engine: str = "",
    engine_version: str = "",
    gpu: str = "",
    on_point: Callable[[BenchmarkResult], None] | None = None,
) -> SweepResult:
    """Measure a latency-throughput curve without restarting the engine.

    Climbs the ladder in order and stops early when continuing cannot help: once the SLA is
    breached or throughput has saturated, higher concurrency only adds latency.
    """
    config = config or SweepConfig()
    sla = sla or SLA()
    points: list[BenchmarkResult] = []
    notes: list[str] = []
    best_throughput = 0.0

    for index, concurrency in enumerate(sorted(config.concurrencies)):
        point = run_at_concurrency(
            endpoint,
            workload,
            concurrency,
            n_requests=max(config.requests_per_point, concurrency),
            seed=config.seed + index,
            warmup_requests=config.warmup_requests if index == 0 else 0,
        )
        points.append(point)
        if on_point is not None:
            on_point(point)

        if point.error_rate > config.max_error_rate:
            notes.append(
                f"stopped at concurrency {concurrency}: error rate "
                f"{point.error_rate:.0%} exceeded {config.max_error_rate:.0%}"
            )
            break

        throughput = point.output_throughput_tokens_s
        if (
            config.stop_on_saturation
            and best_throughput > 0
            and throughput < best_throughput * 1.02
        ):
            notes.append(
                f"throughput saturated at concurrency {concurrency} "
                f"({throughput:.0f} vs best {best_throughput:.0f} tok/s); "
                "this is the empirical critical batch size"
            )
            break
        best_throughput = max(best_throughput, throughput)

        if (
            config.stop_on_sla_violation
            and not sla.is_unconstrained
            and not point.meets(ttft_p99_ms=sla.ttft_p99_ms, tpot_p99_ms=sla.tpot_p99_ms)
        ):
            notes.append(
                f"SLA breached at concurrency {concurrency} "
                f"(ttft p99 {point.ttft.p99_ms:.0f} ms, tpot p99 {point.tpot.p99_ms:.0f} ms); "
                "latency is monotone in concurrency, so higher points cannot satisfy it"
            )
            break

    return SweepResult(
        points=tuple(points),
        model=model,
        engine=engine,
        engine_version=engine_version,
        gpu=gpu,
        notes=tuple(notes),
    )


__all__ = ["DEFAULT_LADDER", "SweepConfig", "sweep_concurrency"]
