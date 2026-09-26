#!/usr/bin/env python
"""Score InferTune's throughput prediction against measured curves.

This is the M3 acceptance measurement: *predicted output throughput within +/-20% on
held-out configurations* (``docs/plan.md`` §9).

Run this **on a GPU host with vLLM installed**. For each configuration it:

1. builds a :class:`ResourcePlan` and compiles it to engine flags via the vLLM adapter,
2. boots ``vllm serve`` with exactly those flags,
3. sweeps client-side concurrency against the running server (one boot, whole curve),
4. records the curve in the measurement store,
5. tears the server down.

Scoring is **leave-one-out cross-validation** over configurations rather than a single
train/test split. With only a handful of affordable boots a single split would make the
result depend on which configuration happened to land in the test set; LOO uses every
configuration as held-out exactly once, and the calibration it is scored against has
genuinely never seen it.

The uncalibrated prior is scored alongside the calibrated fit. Without that contrast a
passing number proves nothing about calibration: if the prior already predicts within
20%, the fitting machinery is unexercised and the criterion is measuring the roofline
model alone.

Usage::

    python scripts/validate_throughput.py --model Qwen/Qwen3-8B
    python scripts/validate_throughput.py --model Qwen/Qwen3-4B --max-tp 1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from infertune.adapters import VLLMAdapter
from infertune.bench import (
    BenchmarkResult,
    EndpointConfig,
    SweepConfig,
    sweep_concurrency,
    wait_for_endpoint,
)
from infertune.core import LogNormal, WorkloadProfile
from infertune.core.gpu import GPUProfile, Interconnect
from infertune.core.model import ModelProfile
from infertune.core.plan import Parallelism, ResourcePlan
from infertune.estimator import (
    InfeasibleConfigurationError,
    RooflineCoefficients,
    calibrate,
    estimate_plan,
    predict,
)
from infertune.hardware import nvml, specdb
from infertune.models import analyze
from infertune.store import MeasurementStore, RunKey, StoredPoint, StoredRun

CRITERION_PCT = 20.0


@dataclass(frozen=True, slots=True)
class Score:
    """One held-out point, predicted two ways."""

    config: str
    concurrency: int
    measured_tps: float
    prior_tps: float
    calibrated_tps: float
    note: str

    @staticmethod
    def _error(predicted: float, measured: float) -> float:
        return abs(predicted - measured) / measured * 100.0

    @property
    def prior_error_pct(self) -> float:
        return self._error(self.prior_tps, self.measured_tps)

    @property
    def calibrated_error_pct(self) -> float:
        return self._error(self.calibrated_tps, self.measured_tps)


@dataclass(frozen=True, slots=True)
class Config:
    """One engine boot's worth of configuration."""

    tensor_parallel: int
    max_num_seqs: int
    max_model_len: int

    @property
    def label(self) -> str:
        return f"tp={self.tensor_parallel} seqs={self.max_num_seqs} len={self.max_model_len}"


DEFAULT_CONFIGS = (
    Config(tensor_parallel=1, max_num_seqs=32, max_model_len=4096),
    Config(tensor_parallel=1, max_num_seqs=128, max_model_len=4096),
    Config(tensor_parallel=2, max_num_seqs=32, max_model_len=4096),
    Config(tensor_parallel=2, max_num_seqs=128, max_model_len=8192),
)


def resolve_gpu(key: str | None, *, count: int) -> GPUProfile:
    """Prefer a live measurement; fall back to the spec database."""
    if key is None:
        try:
            return nvml.discover(count=count)
        except Exception as exc:
            print(f"NVML unavailable ({exc}); using the spec database")
            key = "a10"
    gpu = specdb.load(key)
    if count > 1:
        gpu = replace(gpu, count=count, interconnect=Interconnect.PCIE_GEN4_X16)
    return gpu


def build_plan(
    profile: ModelProfile, gpu: GPUProfile, workload: WorkloadProfile, config: Config
) -> ResourcePlan:
    return estimate_plan(
        profile,
        gpu,
        workload,
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
        parallelism=Parallelism(tensor=config.tensor_parallel),
        samples=2000,
    )


@contextlib.contextmanager
def serve(command: list[str], port: int, log_path: Path, timeout_s: float):  # type: ignore[no-untyped-def]
    """Boot a vLLM server for the duration of the block, and always reap it.

    The server is started in its own process group so that teardown kills the engine's
    worker processes too. A leaked tensor-parallel worker holds VRAM, which would make
    the *next* configuration in the sweep fail to allocate for reasons that look like a
    memory-model error.
    """
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [*command, "--port", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        endpoint = EndpointConfig(base_url=f"http://127.0.0.1:{port}")
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                tail = log_path.read_text().splitlines()[-15:]
                raise RuntimeError("server exited during boot:\n  " + "\n  ".join(tail))
            if wait_for_endpoint(endpoint, timeout_s=5.0, interval_s=1.0):
                break
        else:
            raise RuntimeError(f"server did not come up within {timeout_s:.0f}s")
        yield endpoint
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=30)
        # VRAM is not returned synchronously with process exit.
        time.sleep(10)


def _to_stored_point(point: BenchmarkResult) -> StoredPoint:
    """Convert a measured point for persistence.

    ``avg_context_tokens`` is the mean KV depth *during decode*, which is what the decode
    term is sensitive to. A sequence starts at ``prompt`` and ends at
    ``prompt + output``, so the time-average is ``prompt + output/2`` — using the prompt
    length alone would understate the cache read per step and bias the fitted bandwidth
    utilisation upward.
    """
    n_ok = len(point.successful)
    avg_context: int | None = None
    if n_ok:
        avg_context = int(point.prompt_tokens / n_ok + (point.output_tokens / n_ok) / 2)
    return StoredPoint(
        concurrency=point.concurrency,
        duration_s=point.duration_s,
        requests_ok=n_ok,
        requests_failed=len(point.failed),
        prompt_tokens=point.prompt_tokens,
        output_tokens=point.output_tokens,
        output_tps=point.output_throughput_tokens_s,
        ttft_p50_ms=point.ttft.p50_ms or None,
        ttft_p99_ms=point.ttft.p99_ms or None,
        tpot_p50_ms=point.tpot.p50_ms or None,
        tpot_p99_ms=point.tpot.p99_ms or None,
        avg_context_tokens=avg_context,
    )


def measure(
    model: str,
    profile: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    config: Config,
    adapter: VLLMAdapter,
    *,
    concurrencies: tuple[int, ...],
    requests: int,
    port: int,
    log_dir: Path,
    timeout_s: float,
) -> StoredRun | None:
    """Boot once, sweep concurrency, and return the recorded run."""
    print(f"\n=== {config.label} ===")
    gpu_for_config = replace(
        gpu,
        count=config.tensor_parallel,
        interconnect=(
            gpu.interconnect
            if config.tensor_parallel > 1 and gpu.interconnect is not Interconnect.NONE
            else (Interconnect.PCIE_GEN4_X16 if config.tensor_parallel > 1 else Interconnect.NONE)
        ),
    )
    try:
        plan = build_plan(profile, gpu_for_config, workload, config)
    except (InfeasibleConfigurationError, ValueError) as exc:
        print(f"  skipped, infeasible: {exc}")
        return None

    spec = adapter.compile(
        plan,
        model,
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
        total_vram_bytes=gpu_for_config.vram_bytes,
    )
    if not spec.is_runnable:
        print("  skipped: " + "; ".join(d.message for d in spec.errors))
        return None

    command = spec.command(executable="vllm")
    print(f"  {' '.join(command[1:])}")
    log_path = log_dir / f"server_tp{config.tensor_parallel}_s{config.max_num_seqs}.log"

    ladder = tuple(c for c in concurrencies if c <= config.max_num_seqs)
    with serve(command, port, log_path, timeout_s) as endpoint:
        facts = adapter.parse_startup_log(log_path.read_text())
        result = sweep_concurrency(
            replace(endpoint, model=model),
            workload,
            SweepConfig(
                concurrencies=ladder,
                requests_per_point=requests,
                stop_on_saturation=False,
                stop_on_sla_violation=False,
                context_limit=config.max_model_len,
            ),
            model=model,
            engine="vllm",
            engine_version=adapter.capabilities().version or "unknown",
            gpu=gpu.name,
            on_point=lambda p: print(
                f"    c={p.concurrency:<4d} {p.output_throughput_tokens_s:8.1f} tok/s  "
                f"ttft p50 {p.ttft.p50_ms:7.1f}ms  tpot p50 {p.tpot.p50_ms:6.2f}ms  "
                f"ok={len(p.successful)}/{len(p.records)}"
            ),
        )

    points = tuple(_to_stored_point(p) for p in result.points)
    return StoredRun(
        key=RunKey(
            gpu=gpu.name,
            model=model,
            engine="vllm",
            engine_version=adapter.capabilities().version or "unknown",
            tensor_parallel=config.tensor_parallel,
        ),
        gpu_count=config.tensor_parallel,
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
        kv_bytes_per_token=plan.kv_bytes_per_token,
        predicted_kv_bytes=plan.kv_budget_bytes,
        actual_kv_bytes=facts.kv_cache_bytes,
        weight_bytes=facts.weight_bytes,
        points=points,
        extra={"max_model_len": config.max_model_len},
    )


def score(
    runs: list[tuple[Config, StoredRun]],
    profile: ModelProfile,
    gpu: GPUProfile,
    workload: WorkloadProfile,
    *,
    prompt_tokens: int,
) -> list[Score]:
    """Leave-one-out cross-validation of predicted vs measured throughput."""
    rows: list[Score] = []
    for index, (config, held_out) in enumerate(runs):
        training = [r for j, (_, r) in enumerate(runs) if j != index]
        gpu_for_config = replace(
            gpu,
            count=config.tensor_parallel,
            interconnect=(
                Interconnect.PCIE_GEN4_X16 if config.tensor_parallel > 1 else Interconnect.NONE
            ),
        )
        plan = build_plan(profile, gpu_for_config, workload, config)

        fitted = RooflineCoefficients()
        note = "prior only (no other run had usable evidence)"
        if training:
            result = calibrate(training, profile, gpu_for_config)
            if result.is_calibrated:
                fitted = result.coefficients
                note = f"fitted from {len(training)} other run(s), n={result.n_accepted}"

        for point in held_out.points:
            if point.requests_ok < 1 or point.output_tps <= 0:
                continue
            context = point.avg_context_tokens or prompt_tokens
            base = predict(
                profile,
                gpu_for_config,
                plan,
                batch=point.concurrency,
                prompt_tokens=prompt_tokens,
                avg_context_tokens=context,
            )
            tuned = predict(
                profile,
                gpu_for_config,
                plan,
                batch=point.concurrency,
                prompt_tokens=prompt_tokens,
                avg_context_tokens=context,
                coefficients=fitted,
            )
            rows.append(
                Score(
                    config=config.label,
                    concurrency=point.concurrency,
                    measured_tps=point.output_tps,
                    prior_tps=base.output_throughput_tokens_s.value,
                    calibrated_tps=tuned.output_throughput_tokens_s.value,
                    note=note,
                )
            )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--gpu", default=None, help="specdb key; omit to use NVML")
    ap.add_argument("--max-tp", type=int, default=2)
    ap.add_argument("--concurrencies", default="1,2,4,8,16,32,64")
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--store", default="infertune-m3.db")
    ap.add_argument("--log-dir", default="m3-logs")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    concurrencies = tuple(int(c) for c in args.concurrencies.split(","))

    gpu = resolve_gpu(args.gpu, count=args.max_tp)
    print(f"GPU          : {gpu.name} x{gpu.count}, link {gpu.interconnect} ({gpu.source})")
    for note in gpu.notes:
        print(f"  note: {note}")

    adapter = VLLMAdapter()
    print(f"engine       : vllm {adapter.capabilities().version}")
    profile = analyze(args.model)
    print(f"model        : {args.model}")

    workload = WorkloadProfile(
        input_tokens=LogNormal.from_median_p95(1024, 2048),
        output_tokens=LogNormal.from_median_p95(256, 512),
    )
    configs = [c for c in DEFAULT_CONFIGS if c.tensor_parallel <= args.max_tp]

    store = MeasurementStore(args.store)
    measured: list[tuple[Config, StoredRun]] = []
    for config in configs:
        try:
            run = measure(
                args.model,
                profile,
                gpu,
                workload,
                config,
                adapter,
                concurrencies=concurrencies,
                requests=args.requests,
                port=args.port,
                log_dir=log_dir,
                timeout_s=args.timeout,
            )
        except RuntimeError as exc:
            print(f"  FAILED: {exc}")
            continue
        if run is None or not run.points:
            continue
        store.record(run)
        measured.append((config, run))

    if len(measured) < 2:
        print("\nneed at least two successful configurations to hold one out")
        return 1

    rows = score(measured, profile, gpu, workload, prompt_tokens=1024)
    if not rows:
        print("\nno comparable points")
        return 1

    print("\n" + "=" * 100)
    print(
        f"{'configuration':30s} {'conc':>5s} {'measured':>10s} "
        f"{'prior':>10s} {'err':>7s} {'calibrated':>11s} {'err':>7s}"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row.config:30s} "
            f"{row.concurrency:>5d} "
            f"{row.measured_tps:>10.1f} "
            f"{row.prior_tps:>10.1f} "
            f"{row.prior_error_pct:>6.1f}% "
            f"{row.calibrated_tps:>11.1f} "
            f"{row.calibrated_error_pct:>6.1f}%"
        )

    prior_errors = [r.prior_error_pct for r in rows]
    cal_errors = [r.calibrated_error_pct for r in rows]
    worst = max(cal_errors)
    print("-" * 100)
    print(
        f"prior      : median {statistics.median(prior_errors):5.1f}%  "
        f"worst {max(prior_errors):5.1f}%"
    )
    print(f"calibrated : median {statistics.median(cal_errors):5.1f}%  worst {worst:5.1f}%")
    within = sum(1 for e in cal_errors if e <= CRITERION_PCT)
    print(
        f"{within}/{len(cal_errors)} held-out points within +/-{CRITERION_PCT:.0f}%  "
        f"-> {'PASS' if worst <= CRITERION_PCT else 'FAIL'} (M3 criterion)"
    )

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in rows], indent=2))
        print(f"raw results -> {args.json}")
    return 0 if worst <= CRITERION_PCT else 1


if __name__ == "__main__":
    sys.exit(main())
