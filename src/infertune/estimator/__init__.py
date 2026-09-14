"""Resource estimation: memory ledger and configuration feasibility."""

from __future__ import annotations

from .calibration import CalibrationResult, calibrate, calibrated_for
from .memory import (
    EngineOverheads,
    InfeasibleConfigurationError,
    estimate_overheads,
    estimate_plan,
    max_concurrency_for_budget,
    predict_kv_for_utilization,
    recurrent_reservation_bytes,
    safe_utilization_ceiling,
    with_kv_dtype,
)
from .roofline import (
    Bound,
    Interval,
    PerfPrediction,
    RooflineCoefficients,
    critical_batch_size,
    decode_step,
    predict,
    prefill_step,
    throughput_curve,
)

__all__ = [
    "Bound",
    "CalibrationResult",
    "EngineOverheads",
    "InfeasibleConfigurationError",
    "Interval",
    "PerfPrediction",
    "RooflineCoefficients",
    "calibrate",
    "calibrated_for",
    "critical_batch_size",
    "decode_step",
    "estimate_overheads",
    "estimate_plan",
    "max_concurrency_for_budget",
    "predict",
    "predict_kv_for_utilization",
    "prefill_step",
    "recurrent_reservation_bytes",
    "safe_utilization_ceiling",
    "throughput_curve",
    "with_kv_dtype",
]
