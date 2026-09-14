"""Fit roofline coefficients from measurements.

The model has exactly two efficiency coefficients, and each is recoverable from a measurement
by inverting the relation that produced it:

* **MBU** from decode. ``tpot = bytes / (bandwidth * mbu) + overheads``, so
  ``mbu = bytes / (bandwidth * (tpot - overheads))``.
* **MFU** from prefill. ``ttft = flops / (peak * mfu) + overheads``, so
  ``mfu = flops / (peak * (ttft - overheads))``.

Two deliberate choices:

**Median, not mean.** A single evicted spot instance, a cold cache or a noisy neighbour
produces one absurd sample. The median ignores it; a mean would enshrine it.

**Only fit from measurements in the right regime.** A decode point that is compute-bound says
nothing about bandwidth utilisation. Fitting MBU from it produces a confident, wrong number, so
such points are skipped and counted as rejected rather than silently absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from ..core.dtypes import DType
from ..core.gpu import GPUProfile
from ..core.model import ModelProfile
from ..store.sqlite import RunKey, StoredRun
from .roofline import RooflineCoefficients, communication_time

MIN_PLAUSIBLE = 0.02
MAX_PLAUSIBLE = 1.0
"""A fitted coefficient outside this range means the model is wrong, not the hardware."""


@dataclass(frozen=True, slots=True)
class FitSample:
    """One coefficient recovered from one measurement."""

    value: float
    concurrency: int
    source: str

    @property
    def plausible(self) -> bool:
        return MIN_PLAUSIBLE <= self.value <= MAX_PLAUSIBLE


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Fitted coefficients plus the evidence behind them."""

    coefficients: RooflineCoefficients
    mbu_samples: tuple[FitSample, ...] = field(default_factory=tuple)
    mfu_samples: tuple[FitSample, ...] = field(default_factory=tuple)
    rejected: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_accepted(self) -> int:
        return len(self.mbu_samples) + len(self.mfu_samples)

    @property
    def is_calibrated(self) -> bool:
        return self.n_accepted > 0

    def summary(self) -> str:
        c = self.coefficients
        return (
            f"mfu={c.mfu:.3f} (n={len(self.mfu_samples)}) "
            f"mbu={c.mbu:.3f} (n={len(self.mbu_samples)}) "
            f"+/-{c.relative_uncertainty:.0%}, {len(self.rejected)} rejected"
        )


def fit_mbu_from_decode(
    model: ModelProfile,
    gpu: GPUProfile,
    *,
    tpot_s: float,
    concurrency: int,
    avg_context_tokens: int,
    kv_bytes_per_token: int,
    tensor_parallel_size: int = 1,
    fixed_step_overhead_s: float = 0.002,
) -> FitSample | None:
    """Recover MBU from one measured inter-token latency."""
    if tpot_s <= 0:
        return None
    weight_bytes = model.weight_bytes_per_gpu(tensor_parallel_size)
    kv_bytes = concurrency * avg_context_tokens * kv_bytes_per_token
    bytes_moved = weight_bytes + kv_bytes

    comm = communication_time(gpu, model, concurrency, tensor_parallel_size)
    available = tpot_s - fixed_step_overhead_s - comm
    if available <= 0:
        # Overheads alone exceed the measurement: nothing left to attribute to bandwidth.
        return None
    mbu = bytes_moved / (gpu.mem_bandwidth_bytes_s * available)
    return FitSample(value=mbu, concurrency=concurrency, source="decode")


def fit_mfu_from_prefill(
    model: ModelProfile,
    gpu: GPUProfile,
    *,
    ttft_s: float,
    prompt_tokens: int,
    tensor_parallel_size: int = 1,
    fixed_step_overhead_s: float = 0.002,
    dtype: DType | None = None,
) -> FitSample | None:
    """Recover MFU from one measured time-to-first-token.

    Note this attributes *all* of TTFT to compute, so queueing inflates the apparent FLOPs and
    depresses MFU. Fit from low-concurrency points where queueing is negligible.
    """
    if ttft_s <= 0 or prompt_tokens < 1:
        return None
    dtype = dtype or model.weight_dtype
    linear = 2.0 * model.n_params_active * prompt_tokens
    attention = 4.0 * model.n_layers * model.hidden_size * prompt_tokens * prompt_tokens
    flops = (linear + attention) / tensor_parallel_size

    comm = communication_time(gpu, model, prompt_tokens, tensor_parallel_size, dtype)
    available = ttft_s - fixed_step_overhead_s - comm
    if available <= 0:
        return None
    try:
        peak = gpu.flops(dtype)
    except ValueError:
        peak = max(gpu.dense_flops.values())
    mfu = flops / (peak * available)
    return FitSample(value=mfu, concurrency=1, source="prefill")


def calibrate(
    runs: list[StoredRun],
    model: ModelProfile,
    gpu: GPUProfile,
    *,
    prior: RooflineCoefficients | None = None,
    max_concurrency_for_mfu: int = 4,
) -> CalibrationResult:
    """Fit coefficients from stored runs for one ``(gpu, model, engine version)``.

    Falls back to the prior for whichever coefficient has no usable evidence, so a partial
    calibration is still an improvement rather than an all-or-nothing step.
    """
    prior = prior or RooflineCoefficients()
    mbu: list[FitSample] = []
    mfu: list[FitSample] = []
    rejected: list[str] = []

    for run in runs:
        tp = run.key.tensor_parallel
        per_token = run.kv_bytes_per_token
        for point in run.points:
            if point.requests_ok < 1:
                rejected.append(f"c={point.concurrency}: no successful requests")
                continue
            context = point.avg_context_tokens or (point.prompt_tokens // max(1, point.requests_ok))

            if point.tpot_p50_ms and per_token:
                sample = fit_mbu_from_decode(
                    model,
                    gpu,
                    tpot_s=point.tpot_p50_ms / 1000.0,
                    concurrency=point.concurrency,
                    avg_context_tokens=context,
                    kv_bytes_per_token=per_token,
                    tensor_parallel_size=tp,
                    fixed_step_overhead_s=prior.fixed_step_overhead_s,
                )
                if sample is None:
                    rejected.append(f"c={point.concurrency}: decode overheads exceed tpot")
                elif not sample.plausible:
                    rejected.append(
                        f"c={point.concurrency}: implausible mbu {sample.value:.2f} "
                        "(likely compute-bound, so it carries no bandwidth information)"
                    )
                else:
                    mbu.append(sample)

            if point.ttft_p50_ms and point.concurrency <= max_concurrency_for_mfu:
                prompt = point.prompt_tokens // max(1, point.requests_ok)
                sample = fit_mfu_from_prefill(
                    model,
                    gpu,
                    ttft_s=point.ttft_p50_ms / 1000.0,
                    prompt_tokens=max(1, prompt),
                    tensor_parallel_size=tp,
                    fixed_step_overhead_s=prior.fixed_step_overhead_s,
                )
                if sample is None:
                    rejected.append(f"c={point.concurrency}: prefill overheads exceed ttft")
                elif not sample.plausible:
                    rejected.append(f"c={point.concurrency}: implausible mfu {sample.value:.2f}")
                else:
                    mfu.append(sample)

    fitted = RooflineCoefficients(
        mfu=median(s.value for s in mfu) if mfu else prior.mfu,
        mbu=median(s.value for s in mbu) if mbu else prior.mbu,
        fixed_step_overhead_s=prior.fixed_step_overhead_s,
        samples=len(mbu) + len(mfu),
        source="calibrated" if (mbu or mfu) else "prior",
    )
    return CalibrationResult(
        coefficients=fitted,
        mbu_samples=tuple(mbu),
        mfu_samples=tuple(mfu),
        rejected=tuple(rejected),
    )


def calibrated_for(
    store_runs: list[StoredRun],
    key: RunKey,
    model: ModelProfile,
    gpu: GPUProfile,
    *,
    prior: RooflineCoefficients | None = None,
) -> CalibrationResult:
    """Calibrate using only runs that match ``key`` exactly.

    Engine version is part of the match on purpose: memory accounting and scheduling defaults
    change between releases, so mixing versions pollutes the fit.
    """
    matching = [r for r in store_runs if r.key == key]
    return calibrate(matching, model, gpu, prior=prior)


__all__ = [
    "MAX_PLAUSIBLE",
    "MIN_PLAUSIBLE",
    "CalibrationResult",
    "FitSample",
    "calibrate",
    "calibrated_for",
    "fit_mbu_from_decode",
    "fit_mfu_from_prefill",
]
