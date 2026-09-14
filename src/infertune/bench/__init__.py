"""Benchmarking: load generation, metrics, and the concurrency sweep.

Uses only the standard library, so the harness itself is testable against a local mock server
with no GPU and no engine present.
"""

from __future__ import annotations

from .harness import DEFAULT_LADDER, SweepConfig, sweep_concurrency
from .loadgen import EndpointConfig, run_at_concurrency, synthetic_prompt, wait_for_endpoint
from .metrics import BenchmarkResult, LatencyStats, RequestRecord, SweepResult, percentile

__all__ = [
    "DEFAULT_LADDER",
    "BenchmarkResult",
    "EndpointConfig",
    "LatencyStats",
    "RequestRecord",
    "SweepConfig",
    "SweepResult",
    "percentile",
    "run_at_concurrency",
    "sweep_concurrency",
    "synthetic_prompt",
    "wait_for_endpoint",
]
