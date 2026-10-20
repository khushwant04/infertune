"""The production boot-and-measure callback for :func:`infertune.search.search`.

The search core was written against an injected ``BootAndMeasure`` so that its strategy could be
validated on a synthetic objective. This module is the real implementation of that callable:

1. compile the candidate's plan through the engine's adapter (against the *installed* schema);
2. start the engine on a fresh port and wait for readiness, capturing its log;
3. parse the engine's own startup facts — the KV pool it actually allocated;
4. sweep client-side concurrency on the one boot;
5. stop the engine's whole process group;
6. record the run with full provenance, so later searches can calibrate from it.

A boot that fails is not an exception to the search. It returns an empty sweep, which the search
already treats as unusable, and the reason is kept on a :class:`BootRecord` so it can be reported.
Raising instead would throw away every measurement taken so far because one candidate OOMed.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from .. import __version__
from ..adapters.base import FrameworkAdapter, LaunchSpec, StartupFacts
from ..bench.harness import SweepConfig, sweep_concurrency
from ..bench.loadgen import EndpointConfig
from ..bench.metrics import SweepResult
from ..core.gpu import GPUProfile
from ..core.workload import SLA, WorkloadProfile
from ..estimator.memory import safe_utilization_ceiling
from ..search.space import Candidate, ScoredCandidate
from ..store.sqlite import MeasurementStore, RunKey, StoredPoint, StoredRun
from .lifecycle import EngineLaunch, EngineStartError, free_port, start_engine

BootStatus = Literal["measured", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class BootRecord:
    """What happened when one candidate was booted."""

    candidate: Candidate
    status: BootStatus
    spec: LaunchSpec | None = None
    """The compiled launch, without the per-boot host and port."""

    command: tuple[str, ...] = ()
    reason: str = ""
    log_path: Path | None = None
    log_tail: str = ""
    startup: StartupFacts | None = None
    ready_after_s: float | None = None
    elapsed_s: float = 0.0
    run_id: int | None = None
    predicted_kv_bytes: int | None = None

    @property
    def kv_error(self) -> float | None:
        """Relative error of the predicted KV pool against what the engine allocated."""
        actual = self.startup.kv_cache_bytes if self.startup else None
        if not actual or not self.predicted_kv_bytes:
            return None
        return abs(self.predicted_kv_bytes - actual) / actual


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """How candidates are booted and measured."""

    model_id: str
    """The repository id passed to the engine, and the model name requests are sent to."""

    log_dir: Path = Path("infertune-runs")
    host: str = "127.0.0.1"
    port: int | None = None
    """Fixed port for every boot; ``None`` picks a fresh free port per boot."""

    executable: str | None = None
    """Replaces the adapter's launcher prefix, e.g. ``/opt/venv/bin/vllm``."""

    env: Mapping[str, str] | None = None
    ready_timeout_s: float = 900.0
    poll_interval_s: float = 2.0
    sweep: SweepConfig = field(default_factory=SweepConfig)
    clip_ladder_to_max_seqs: bool = True
    """Concurrency above the engine's ``max_num_seqs`` only queues client-side, so it measures
    the scheduler's queue rather than the GPU. Clipping keeps the sweep honest and short."""


def stored_points(sweep: SweepResult) -> tuple[StoredPoint, ...]:
    """Convert a sweep into storable points, including the context length calibration needs.

    ``avg_context_tokens`` is the average KV context during decode: the prompt plus half the
    output, since each request's context grows linearly while it generates.
    """
    points: list[StoredPoint] = []
    for p in sweep.points:
        ok = len(p.successful)
        context = (
            round((p.prompt_tokens + p.output_tokens / 2) / ok) if ok and p.prompt_tokens else None
        )
        points.append(
            StoredPoint(
                concurrency=p.concurrency,
                duration_s=p.duration_s,
                requests_ok=ok,
                requests_failed=len(p.failed),
                prompt_tokens=p.prompt_tokens,
                output_tokens=p.output_tokens,
                output_tps=p.output_throughput_tokens_s,
                ttft_p50_ms=p.ttft.p50_ms or None,
                ttft_p99_ms=p.ttft.p99_ms or None,
                tpot_p50_ms=p.tpot.p50_ms or None,
                tpot_p99_ms=p.tpot.p99_ms or None,
                avg_context_tokens=context,
            )
        )
    return tuple(points)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "run"


class EngineExecutor:
    """Boot candidates for real. An instance is a ``BootAndMeasure`` callable.

    Args:
        adapter: Adapter for the installed engine. Its schema must be introspected, not
            recorded, so that the compiled flags match what will actually run.
        gpu: The hardware the engine will run on.
        workload: Request length distribution for the concurrency sweep.
        config: Launch and sweep settings.
        sla: Used to stop each sweep once latency breaches it.
        store: Where measured runs are recorded. ``None`` skips recording.
        on_event: Progress callback, one human-readable line at a time.
    """

    def __init__(
        self,
        adapter: FrameworkAdapter,
        gpu: GPUProfile,
        workload: WorkloadProfile,
        config: ExecutionConfig,
        *,
        sla: SLA | None = None,
        store: MeasurementStore | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.adapter = adapter
        self.gpu = gpu
        self.workload = workload
        self.config = config
        self.sla = sla or SLA()
        self.store = store
        self.on_event = on_event
        self.records: list[BootRecord] = []
        self.engine_version = adapter.capabilities().version
        self._started = time.strftime("%Y%m%d-%H%M%S")

    # ----------------------------------------------------------------- compilation

    def compile(self, scored: ScoredCandidate) -> LaunchSpec:
        """The launch spec for a candidate, as the installed engine will receive it."""
        candidate = scored.candidate
        ceiling = safe_utilization_ceiling(
            self.gpu.vram_bytes, max(0, self.gpu.vram_bytes - self.gpu.vram_usable_bytes)
        )
        return self.adapter.compile(
            scored.plan,
            self.config.model_id,
            max_num_seqs=candidate.max_num_seqs,
            max_model_len=candidate.max_model_len,
            total_vram_bytes=self.gpu.vram_bytes,
            max_utilization=ceiling,
            enforce_eager=candidate.enforce_eager,
        )

    def command(self, spec: LaunchSpec, port: int) -> tuple[str, ...]:
        argv = self.adapter.command(spec, self.config.executable)
        return (*argv, "--host", self.config.host, "--port", str(port))

    def ladder(self, candidate: Candidate) -> SweepConfig:
        sweep = self.config.sweep
        if not self.config.clip_ladder_to_max_seqs:
            return sweep
        clipped = tuple(c for c in sweep.concurrencies if c <= candidate.max_num_seqs)
        return replace(sweep, concurrencies=clipped or (candidate.max_num_seqs,))

    # ----------------------------------------------------------------- execution

    def __call__(self, scored: ScoredCandidate) -> SweepResult:
        candidate = scored.candidate
        index = len(self.records) + 1
        started = time.monotonic()
        self._emit(f"[{index}] {candidate.label()}: compiling")

        spec = self.compile(scored)
        if not spec.is_runnable:
            reason = "; ".join(d.message for d in spec.errors)
            return self._finish(
                BootRecord(
                    candidate=candidate,
                    status="skipped",
                    reason=f"not runnable on {self.adapter.name} {self.engine_version}: {reason}",
                    predicted_kv_bytes=scored.plan.kv_budget_bytes,
                    elapsed_s=time.monotonic() - started,
                )
            )

        port = self.config.port or free_port(self.config.host)
        command = self.command(spec, port)
        log_name = f"{index:02d}-{_slug(candidate.label())}.log"
        log_path = self.config.log_dir / self._started / log_name
        launch = EngineLaunch(
            command=command,
            base_url=f"http://{self.config.host}:{port}",
            log_path=log_path,
            env=self.config.env,
            ready_timeout_s=self.config.ready_timeout_s,
            poll_interval_s=self.config.poll_interval_s,
        )

        self._emit(f"[{index}] booting: {' '.join(command)}")
        try:
            engine = start_engine(launch)
        except EngineStartError as exc:
            facts = self.adapter.parse_startup_log(_read(log_path))
            return self._finish(
                BootRecord(
                    candidate=candidate,
                    status="failed",
                    spec=spec,
                    command=command,
                    reason=str(exc),
                    log_path=log_path,
                    log_tail=exc.log_tail,
                    startup=facts,
                    predicted_kv_bytes=scored.plan.kv_budget_bytes,
                    elapsed_s=time.monotonic() - started,
                )
            )

        with engine:
            self._emit(f"[{index}] ready after {engine.ready_after_s or 0:.0f}s; sweeping")
            sweep = sweep_concurrency(
                EndpointConfig(base_url=launch.base_url, model=self.config.model_id),
                self.workload,
                self.ladder(candidate),
                sla=self.sla,
                model=self.config.model_id,
                engine=self.adapter.name,
                engine_version=self.engine_version,
                gpu=self.gpu.name,
            )
            facts = self.adapter.parse_startup_log(engine.log_text())

        run_id = self._record(scored, spec, command, facts, sweep, log_path)
        best = sweep.best_throughput
        record = BootRecord(
            candidate=candidate,
            status="measured" if best is not None else "failed",
            spec=spec,
            command=command,
            reason="" if best is not None else "no request succeeded during the sweep",
            log_path=log_path,
            startup=facts,
            ready_after_s=engine.ready_after_s,
            elapsed_s=time.monotonic() - started,
            run_id=run_id,
            predicted_kv_bytes=scored.plan.kv_budget_bytes,
        )
        self._finish(record)
        return sweep

    def _finish(self, record: BootRecord) -> SweepResult:
        self.records.append(record)
        if record.status != "measured":
            self._emit(f"[{len(self.records)}] {record.status}: {record.reason}")
        return SweepResult(
            points=(),
            model=self.config.model_id,
            engine=self.adapter.name,
            engine_version=self.engine_version,
            gpu=self.gpu.name,
            notes=(record.reason,) if record.reason else (),
        )

    def _record(
        self,
        scored: ScoredCandidate,
        spec: LaunchSpec,
        command: tuple[str, ...],
        facts: StartupFacts,
        sweep: SweepResult,
        log_path: Path,
    ) -> int | None:
        if self.store is None or not sweep.points:
            return None
        candidate = scored.candidate
        plan = scored.plan
        return self.store.record(
            StoredRun(
                key=RunKey(
                    gpu=self.gpu.name,
                    model=self.config.model_id,
                    engine=self.adapter.name,
                    engine_version=self.engine_version,
                    tensor_parallel=candidate.tensor_parallel,
                ),
                gpu_count=self.gpu.count,
                max_num_seqs=candidate.max_num_seqs,
                max_model_len=candidate.max_model_len,
                kv_dtype=str(candidate.kv_dtype),
                weight_bytes=plan.weight_bytes_per_gpu,
                kv_bytes_per_token=plan.kv_bytes_per_token,
                predicted_kv_bytes=plan.kv_budget_bytes,
                actual_kv_bytes=facts.kv_cache_bytes,
                notes=" | ".join(sweep.notes),
                extra={
                    "source": "tune --execute",
                    "infertune_version": __version__,
                    "gpu_source": self.gpu.source,
                    "vram_bytes": self.gpu.vram_bytes,
                    "enforce_eager": candidate.enforce_eager,
                    "command": list(command),
                    "flags": [list(pair) for pair in spec.args],
                    "log": str(log_path),
                    "predicted_throughput": scored.predicted_throughput,
                    "engine_kv_tokens": facts.kv_cache_tokens,
                },
                points=stored_points(sweep),
            )
        )

    def _emit(self, message: str) -> None:
        if self.on_event is not None:
            self.on_event(message)


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


__all__ = [
    "BootRecord",
    "BootStatus",
    "EngineExecutor",
    "ExecutionConfig",
    "stored_points",
]
