"""Engine execution: the part of ``tune --execute`` that spends real boots.

:mod:`.lifecycle` starts an engine, waits for readiness, captures its log and stops its process
group. :mod:`.executor` composes that with an adapter, the concurrency sweep and the measurement
store into the ``BootAndMeasure`` callable the search core expects.
"""

from __future__ import annotations

from .executor import BootRecord, BootStatus, EngineExecutor, ExecutionConfig, stored_points
from .lifecycle import (
    EngineLaunch,
    EngineStartError,
    RunningEngine,
    free_port,
    is_healthy,
    start_engine,
)

__all__ = [
    "BootRecord",
    "BootStatus",
    "EngineExecutor",
    "EngineLaunch",
    "EngineStartError",
    "ExecutionConfig",
    "RunningEngine",
    "free_port",
    "is_healthy",
    "start_engine",
    "stored_points",
]
