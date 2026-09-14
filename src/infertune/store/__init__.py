"""Measurement persistence.

A local SQLite file, keyed by ``(gpu, model, engine, engine_version, tensor_parallel)`` so that
only comparable measurements are ever pooled for calibration.
"""

from __future__ import annotations

from .sqlite import SCHEMA_VERSION, MeasurementStore, RunKey, StoredPoint, StoredRun

__all__ = [
    "SCHEMA_VERSION",
    "MeasurementStore",
    "RunKey",
    "StoredPoint",
    "StoredRun",
]
