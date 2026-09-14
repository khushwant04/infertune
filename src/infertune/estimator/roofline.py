"""Analytical performance model: prefill, decode, and where the wall is.

Deliberately a two-coefficient physical model rather than a learned surrogate. A model with
interpretable coefficients can be calibrated from a handful of measurements and can explain
itself; a black-box surrogate needs hundreds of samples and cannot say *why*.

The two regimes behave completely differently:

* **Prefill is compute-bound.** It processes many tokens at once, so arithmetic dominates.
* **Decode is memory-bandwidth-bound.** It reads all the weights to produce one token per
  sequence, so bytes moved dominate until the batch is large enough to amortise them.

The crossover is the critical batch size, ``B* = (FLOPS x MFU) / (bandwidth x MBU)``, and it
is the single most decision-relevant number here: below it, added concurrency is nearly free;
above it, concurrency buys latency rather than throughput.

Tensor parallelism is charged **both** a bandwidth and a per-call latency term. Two
all-reduces per layer means a 36-layer model synchronises 72 times per decode step, so on PCIe
the fixed cost per call can dominate the transfer itself — which is why the same model on the
same VRAM behaves differently on NVLink and PCIe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from ..core.dtypes import DType
from ..core.gpu import GPUProfile
from ..core.model import ModelProfile
from ..core.plan import ResourcePlan


class Bound(Enum):
    """What limits a step."""

    MEMORY = "memory"
    COMPUTE = "compute"
    COMMUNICATION = "communication"


@dataclass(frozen=True, slots=True)
class RooflineCoefficients:
    """Efficiency factors, the only fitted quantities in the model.

    ``mfu`` and ``mbu`` are the fraction of peak compute and peak bandwidth actually achieved.
    Priors are deliberately mid-range; :mod:`infertune.estimator.calibration` refits them from
    measurements, and ``samples`` records how much evidence stands behind them so predictions
    can be reported with an honest interval instead of false precision.
    """

    mfu: float = 0.5
    mbu: float = 0.8
    fixed_step_overhead_s: float = 0.002
    """Per-step scheduling, sampling and kernel-launch cost that scales with neither."""

    samples: int = 0
    source: str = "prior"

    def __post_init__(self) -> None:
        if not 0 < self.mfu <= 1:
            raise ValueError(f"mfu must be in (0, 1], got {self.mfu}")
        if not 0 < self.mbu <= 1:
            raise ValueError(f"mbu must be in (0, 1], got {self.mbu}")
        if self.fixed_step_overhead_s < 0:
            raise ValueError("fixed_step_overhead_s must be >= 0")
        if self.samples < 0:
            raise ValueError("samples must be >= 0")

    @property
    def relative_uncertainty(self) -> float:
        """Fractional width to report around a prediction.

        Wide when uncalibrated and narrowing as evidence accumulates, floored at 10% because
        an analytical model over a real serving stack is never better than that.
        """
        if self.samples <= 0:
            return 0.50
        return max(0.10, 0.50 / math.sqrt(1 + self.samples))


@dataclass(frozen=True, slots=True)
class Interval:
    """A prediction with an honest range attached."""

    value: float
    low: float
    high: float

    @property
    def width_fraction(self) -> float:
        return (self.high - self.low) / self.value if self.value else float("inf")


def _interval(value: float, uncertainty: float) -> Interval:
    return Interval(value=value, low=value * (1 - uncertainty), high=value * (1 + uncertainty))


@dataclass(frozen=True, slots=True)
class StepBreakdown:
    """Time for one forward step, decomposed so the binding term is visible."""

    t_memory_s: float
    t_compute_s: float
    t_communication_s: float
    t_fixed_s: float
    bytes_moved: int
    flops: float

    @property
    def bound(self) -> Bound:
        if self.t_communication_s > max(self.t_memory_s, self.t_compute_s):
            return Bound.COMMUNICATION
        return Bound.COMPUTE if self.t_compute_s > self.t_memory_s else Bound.MEMORY

    @property
    def total_s(self) -> float:
        # Compute and memory overlap; communication and fixed costs serialise.
        return max(self.t_memory_s, self.t_compute_s) + self.t_communication_s + self.t_fixed_s


def _achievable_flops(gpu: GPUProfile, dtype: DType, mfu: float) -> float:
    try:
        peak = gpu.flops(dtype)
    except ValueError:
        # Fall back to the widest dtype the device does support, which is what an engine
        # would end up using anyway.
        peak = max(gpu.dense_flops.values())
    return peak * mfu


def communication_time(
    gpu: GPUProfile,
    model: ModelProfile,
    tokens_in_step: int,
    tensor_parallel_size: int,
    dtype: DType = DType.BF16,
) -> float:
    """All-reduce cost for one step under tensor parallelism.

    Two all-reduces per layer, each moving ``tokens x hidden`` elements. Ring all-reduce moves
    ``2(n-1)/n`` of the payload, and every call also pays a fixed latency — the term that makes
    TP over PCIe expensive even when the payload is small.
    """
    if tensor_parallel_size <= 1:
        return 0.0
    link = gpu.interconnect
    if link.bytes_s <= 0:
        return 0.0

    calls = 2 * model.n_layers
    payload = tokens_in_step * model.hidden_size * (dtype.bits // 8)
    ring_factor = 2 * (tensor_parallel_size - 1) / tensor_parallel_size
    transfer_s = calls * ring_factor * payload / link.bytes_s
    latency_s = calls * link.latency_s
    return transfer_s + latency_s


def decode_step(
    model: ModelProfile,
    gpu: GPUProfile,
    *,
    batch: int,
    avg_context_tokens: int,
    kv_bytes_per_token: int,
    tensor_parallel_size: int = 1,
    coefficients: RooflineCoefficients | None = None,
    dtype: DType | None = None,
) -> StepBreakdown:
    """One decode step: every sequence in the batch emits one token.

    Bytes moved is the whole point. Weights are read once per step regardless of batch size,
    which is why decode throughput improves almost for free until the batch is large enough
    that reading the KV cache and doing arithmetic start to matter.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if avg_context_tokens < 0:
        raise ValueError("avg_context_tokens must be >= 0")
    coef = coefficients or RooflineCoefficients()
    dtype = dtype or model.weight_dtype

    weight_bytes = model.weight_bytes_per_gpu(tensor_parallel_size)
    kv_bytes = batch * avg_context_tokens * kv_bytes_per_token
    bytes_moved = weight_bytes + kv_bytes
    t_memory = bytes_moved / (gpu.mem_bandwidth_bytes_s * coef.mbu)

    # One token per sequence: 2 FLOPs per active parameter per token.
    flops = 2.0 * model.n_params_active * batch / tensor_parallel_size
    t_compute = flops / _achievable_flops(gpu, dtype, coef.mfu)

    t_comm = communication_time(gpu, model, batch, tensor_parallel_size, dtype)
    return StepBreakdown(
        t_memory_s=t_memory,
        t_compute_s=t_compute,
        t_communication_s=t_comm,
        t_fixed_s=coef.fixed_step_overhead_s,
        bytes_moved=bytes_moved,
        flops=flops,
    )


def prefill_step(
    model: ModelProfile,
    gpu: GPUProfile,
    *,
    prompt_tokens: int,
    tensor_parallel_size: int = 1,
    coefficients: RooflineCoefficients | None = None,
    dtype: DType | None = None,
) -> StepBreakdown:
    """Prefill for one prompt, including the quadratic attention term.

    ``2 * P * N`` dominates at short context, but attention scales as ``N^2`` and takes over
    at long context — which is why TTFT stops being linear in prompt length.
    """
    if prompt_tokens < 1:
        raise ValueError(f"prompt_tokens must be >= 1, got {prompt_tokens}")
    coef = coefficients or RooflineCoefficients()
    dtype = dtype or model.weight_dtype

    linear_flops = 2.0 * model.n_params_active * prompt_tokens
    attention_flops = 4.0 * model.n_layers * model.hidden_size * prompt_tokens * prompt_tokens
    flops = (linear_flops + attention_flops) / tensor_parallel_size
    t_compute = flops / _achievable_flops(gpu, dtype, coef.mfu)

    # Weights are streamed once; activations dominate reads at long context.
    bytes_moved = model.weight_bytes_per_gpu(tensor_parallel_size)
    t_memory = bytes_moved / (gpu.mem_bandwidth_bytes_s * coef.mbu)

    t_comm = communication_time(gpu, model, prompt_tokens, tensor_parallel_size, dtype)
    return StepBreakdown(
        t_memory_s=t_memory,
        t_compute_s=t_compute,
        t_communication_s=t_comm,
        t_fixed_s=coef.fixed_step_overhead_s,
        bytes_moved=bytes_moved,
        flops=flops,
    )


@dataclass(frozen=True, slots=True)
class PerfPrediction:
    """Predicted serving performance, with intervals and the binding term named."""

    ttft_s: Interval
    tpot_s: Interval
    output_throughput_tokens_s: Interval
    decode_bound: Bound
    prefill_bound: Bound
    critical_batch_size: float
    batch: int
    coefficients: RooflineCoefficients

    @property
    def is_bandwidth_bound(self) -> bool:
        return self.decode_bound is Bound.MEMORY

    @property
    def headroom_to_critical_batch(self) -> float:
        """How much concurrency remains before compute becomes the wall."""
        return self.critical_batch_size / self.batch if self.batch else float("inf")


def critical_batch_size(
    gpu: GPUProfile,
    dtype: DType = DType.BF16,
    coefficients: RooflineCoefficients | None = None,
) -> float:
    """Batch size at which decode stops being bandwidth-bound.

    Derived by equating the two roofline terms with weights dominating bytes moved, which is
    the regime that matters at realistic context lengths.
    """
    coef = coefficients or RooflineCoefficients()
    return _achievable_flops(gpu, dtype, coef.mfu) / (gpu.mem_bandwidth_bytes_s * coef.mbu)


def predict(
    model: ModelProfile,
    gpu: GPUProfile,
    plan: ResourcePlan,
    *,
    batch: int,
    prompt_tokens: int,
    avg_context_tokens: int | None = None,
    coefficients: RooflineCoefficients | None = None,
) -> PerfPrediction:
    """Predict TTFT, inter-token latency and output throughput for a configuration."""
    coef = coefficients or RooflineCoefficients()
    tp = plan.parallelism.tensor
    context = avg_context_tokens if avg_context_tokens is not None else prompt_tokens

    prefill = prefill_step(
        model, gpu, prompt_tokens=prompt_tokens, tensor_parallel_size=tp, coefficients=coef
    )
    decode = decode_step(
        model,
        gpu,
        batch=batch,
        avg_context_tokens=context,
        kv_bytes_per_token=plan.kv_bytes_per_token,
        tensor_parallel_size=tp,
        coefficients=coef,
    )

    u = coef.relative_uncertainty
    tpot = decode.total_s
    return PerfPrediction(
        ttft_s=_interval(prefill.total_s, u),
        tpot_s=_interval(tpot, u),
        output_throughput_tokens_s=_interval(batch / tpot if tpot > 0 else 0.0, u),
        decode_bound=decode.bound,
        prefill_bound=prefill.bound,
        critical_batch_size=critical_batch_size(gpu, model.weight_dtype, coef),
        batch=batch,
        coefficients=coef,
    )


def throughput_curve(
    model: ModelProfile,
    gpu: GPUProfile,
    plan: ResourcePlan,
    *,
    batches: tuple[int, ...],
    prompt_tokens: int = 1024,
    avg_context_tokens: int | None = None,
    coefficients: RooflineCoefficients | None = None,
) -> tuple[PerfPrediction, ...]:
    """Predicted throughput/latency across a concurrency sweep.

    The analytical counterpart to the benchmark harness's inner loop, and cheap enough to
    prune candidates before any engine is booted.
    """
    return tuple(
        predict(
            model,
            gpu,
            plan,
            batch=b,
            prompt_tokens=prompt_tokens,
            avg_context_tokens=avg_context_tokens,
            coefficients=coefficients,
        )
        for b in batches
    )


def with_coefficients(
    prediction: PerfPrediction, coefficients: RooflineCoefficients
) -> PerfPrediction:
    """Restate a prediction under different coefficients, keeping its shape."""
    return replace(prediction, coefficients=coefficients)


__all__ = [
    "Bound",
    "Interval",
    "PerfPrediction",
    "RooflineCoefficients",
    "StepBreakdown",
    "communication_time",
    "critical_batch_size",
    "decode_step",
    "predict",
    "prefill_step",
    "throughput_curve",
    "with_coefficients",
]
