"""Engine execution: lifecycle, the boot-and-measure executor, and ``tune --execute``.

Every test here starts a real subprocess (``tests/fake_engine.py``) that behaves like ``vllm
serve``: it replays a recorded vLLM startup log and serves the streaming completions API. That
exercises the parts a synthetic objective cannot — process groups, readiness polling, log
capture, early-exit detection, and the store records a real boot produces — without a GPU.
"""

from __future__ import annotations

import os
import shlex
import time
import traceback
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from infertune.adapters import ParamSchema, VLLMAdapter, load_schema
from infertune.bench import SweepConfig
from infertune.bench.metrics import BenchmarkResult, RequestRecord, SweepResult
from infertune.core.workload import Constant, WorkloadProfile
from infertune.estimator import calibrate_by_tp, calibrated_for
from infertune.hardware import specdb
from infertune.runner import (
    EngineExecutor,
    EngineLaunch,
    EngineStartError,
    ExecutionConfig,
    free_port,
    is_healthy,
    start_engine,
    stored_points,
)
from infertune.search import search
from infertune.store import MeasurementStore, RunKey, StoredPoint, StoredRun
from tests import fake_engine
from tests.test_estimator import LLAMA_31_8B

SCHEMA_FILE = Path(__file__).resolve().parent / "fixtures" / "vllm" / "engine_args_0.19.1.json"
GPU = specdb.load("h100-sxm")
WORKLOAD = WorkloadProfile(input_tokens=Constant(16), output_tokens=Constant(8))
EXECUTABLE = shlex.join([*fake_engine.command()[:3]])


@pytest.fixture
def schema() -> ParamSchema:
    return load_schema(SCHEMA_FILE)


def _launch(tmp_path: Path, timeout: float = 30.0, **env: str) -> EngineLaunch:
    port = free_port()
    return EngineLaunch(
        command=tuple(fake_engine.command("serve", "m", "--port", str(port))),
        base_url=f"http://127.0.0.1:{port}",
        log_path=tmp_path / "engine.log",
        env=fake_engine.env(**env),
        ready_timeout_s=timeout,
        poll_interval_s=0.05,
        stop_timeout_s=5.0,
    )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0; treat it as gone once its state says so.
    stat = Path(f"/proc/{pid}/stat")
    return not (stat.exists() and stat.read_text().split()[2] == "Z")


# --------------------------------------------------------------------------- lifecycle


class TestLifecycle:
    def test_starts_serves_and_stops(self, tmp_path: Path) -> None:
        launch = _launch(tmp_path)
        with start_engine(launch) as engine:
            assert engine.is_running()
            assert engine.ready_after_s is not None
            assert is_healthy(launch.base_url)
            log = engine.log_text()
        assert not engine.is_running()
        assert not is_healthy(launch.base_url)
        assert log.startswith("$ ")
        assert "Available KV cache memory" in log

    def test_early_exit_is_reported_without_waiting_out_the_timeout(self, tmp_path: Path) -> None:
        launch = _launch(tmp_path, timeout=60.0, FAKE_ENGINE_OOM_ABOVE_SEQS="1")
        started = time.monotonic()
        with pytest.raises(EngineStartError) as info:
            start_engine(launch)
        assert time.monotonic() - started < 30
        assert info.value.exit_code == 1
        assert "exited with code 1" in str(info.value)
        assert "CUDA out of memory" in info.value.log_tail

    def test_stalled_boot_times_out_and_is_stopped(self, tmp_path: Path) -> None:
        launch = _launch(tmp_path, timeout=1.0, FAKE_ENGINE_HANG="1")
        with pytest.raises(EngineStartError, match="not ready after"):
            start_engine(launch)

    def test_stop_reaps_the_whole_process_group(self, tmp_path: Path) -> None:
        pidfile = tmp_path / "child.pid"
        launch = _launch(tmp_path, FAKE_ENGINE_CHILD_PIDFILE=str(pidfile))
        with start_engine(launch):
            child = int(pidfile.read_text())
            assert _alive(child)
        deadline = time.monotonic() + 5
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(child), "an engine worker outlived the engine"

    def test_missing_executable_is_a_start_error(self, tmp_path: Path) -> None:
        launch = EngineLaunch(
            command=("infertune-no-such-engine", "serve"),
            base_url="http://127.0.0.1:1",
            log_path=tmp_path / "x.log",
        )
        with pytest.raises(EngineStartError, match="could not start"):
            start_engine(launch)

    def test_launch_validates_its_inputs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="command"):
            EngineLaunch(command=(), base_url="http://x", log_path=tmp_path / "l")
        with pytest.raises(ValueError, match="ready_timeout_s"):
            EngineLaunch(
                command=("x",), base_url="http://x", log_path=tmp_path / "l", ready_timeout_s=0
            )


# --------------------------------------------------------------------------- executor


def _executor(
    tmp_path: Path, schema: ParamSchema, store: MeasurementStore | None, **env: str
) -> EngineExecutor:
    return EngineExecutor(
        VLLMAdapter(schema),
        GPU,
        WORKLOAD,
        ExecutionConfig(
            model_id=LLAMA_31_8B.model_id,
            log_dir=tmp_path / "logs",
            executable=EXECUTABLE,
            env=fake_engine.env(**env),
            ready_timeout_s=30.0,
            poll_interval_s=0.05,
            sweep=SweepConfig(concurrencies=(1, 2, 4), requests_per_point=4, warmup_requests=0),
        ),
        store=store,
    )


class TestExecutor:
    def test_search_boots_measures_and_records(self, tmp_path: Path, schema: ParamSchema) -> None:
        store = MeasurementStore(tmp_path / "m.db")
        executor = _executor(tmp_path, schema, store)
        result = search(LLAMA_31_8B, GPU, WORKLOAD, executor, boot_budget=2, prune_samples=50)

        assert result.boots_used == 2
        assert result.best is not None
        assert [r.status for r in executor.records] == ["measured", "measured"]

        runs = store.runs()
        assert len(runs) == 2
        run = runs[0]
        assert run.key.is_complete
        assert run.key.engine == "vllm"
        assert run.key.engine_version == "0.19.1"
        assert run.key.gpu == GPU.name
        assert run.kv_bytes_per_token
        assert run.predicted_kv_bytes
        # Parsed from the engine's own startup log, not predicted.
        assert run.actual_kv_bytes is not None
        assert run.extra["source"] == "tune --execute"
        assert run.points
        assert all(p.avg_context_tokens for p in run.points)

        record = executor.records[0]
        assert record.log_path is not None
        assert record.log_path.is_file()
        assert record.spec is not None
        assert "--host" not in record.spec.command_line()
        assert "--port" in record.command
        assert record.kv_error is not None

    def test_failed_boot_is_recorded_and_the_search_continues(
        self, tmp_path: Path, schema: ParamSchema
    ) -> None:
        store = MeasurementStore(tmp_path / "m.db")
        executor = _executor(tmp_path, schema, store, FAKE_ENGINE_OOM_ABOVE_SEQS="64")
        result = search(LLAMA_31_8B, GPU, WORKLOAD, executor, boot_budget=6, prune_samples=50)

        statuses = {r.candidate.max_num_seqs: r.status for r in executor.records}
        assert any(s == "failed" for s in statuses.values()), statuses
        assert any(s == "measured" for s in statuses.values()), statuses
        assert all(
            status == ("failed" if seqs > 64 else "measured") for seqs, status in statuses.items()
        )
        failed = next(r for r in executor.records if r.status == "failed")
        assert "CUDA out of memory" in failed.log_tail
        assert result.best is not None
        assert result.best.scored.candidate.max_num_seqs <= 64
        # Only measured boots are stored; a failed boot has no curve to calibrate from.
        assert len(store.runs()) == sum(1 for r in executor.records if r.status == "measured")

    def test_ladder_is_clipped_to_max_num_seqs(self, tmp_path: Path, schema: ParamSchema) -> None:
        executor = _executor(tmp_path, schema, None)
        from infertune.search import Candidate

        assert executor.ladder(Candidate(max_num_seqs=2)).concurrencies == (1, 2)
        assert executor.ladder(Candidate(max_num_seqs=256)).concurrencies == (1, 2, 4)


def test_stored_points_carry_decode_context() -> None:
    records = tuple(
        RequestRecord(prompt_tokens=100, output_tokens=20, ttft_s=0.05, total_s=0.25)
        for _ in range(4)
    )
    sweep = SweepResult(points=(BenchmarkResult(concurrency=4, duration_s=1.0, records=records),))
    (point,) = stored_points(sweep)
    assert point.requests_ok == 4
    assert point.avg_context_tokens == 110  # prompt + half the output
    assert point.tpot_p50_ms is not None


# --------------------------------------------------------------------------- provenance


def _run(key: RunKey) -> StoredRun:
    return StoredRun(
        key=key,
        kv_bytes_per_token=131_072,
        points=(
            StoredPoint(
                concurrency=8,
                duration_s=10.0,
                requests_ok=32,
                requests_failed=0,
                prompt_tokens=32 * 1024,
                output_tokens=32 * 256,
                output_tps=800.0,
                ttft_p50_ms=120.0,
                tpot_p50_ms=12.0,
                avg_context_tokens=1152,
            ),
        ),
    )


class TestProvenance:
    KEY = RunKey(gpu=GPU.name, model="m", engine="vllm", engine_version="0.19.1")

    def test_unknown_labels_are_incomplete(self) -> None:
        assert self.KEY.is_complete
        assert not RunKey(gpu="unknown", model="m", engine="vllm", engine_version="1").is_complete
        assert not RunKey(gpu="g", model="m", engine="vllm", engine_version="").is_complete

    def test_incomplete_key_never_calibrates(self) -> None:
        unknown = RunKey(gpu="unknown", model="m", engine="vllm", engine_version="unknown")
        result = calibrated_for([_run(unknown)], unknown, LLAMA_31_8B, GPU)
        assert not result.is_calibrated
        assert "incomplete provenance" in result.rejected[0]

    def test_each_tp_degree_is_fitted_only_from_its_own_runs(self) -> None:
        results = calibrate_by_tp([_run(self.KEY)], self.KEY, (1, 2), LLAMA_31_8B, GPU)
        assert results[1].is_calibrated
        assert not results[2].is_calibrated


# --------------------------------------------------------------------------- CLI

runner = CliRunner()


def _failure(result: Result) -> str:
    """Output plus the traceback of an unexpected exception, for a readable assertion."""
    if result.exc_info is None or isinstance(result.exception, SystemExit):
        return result.stdout
    return result.stdout + "".join(traceback.format_exception(*result.exc_info))


def test_tune_execute_without_a_gpu_starts_nothing() -> None:
    from infertune.cli import app

    result = runner.invoke(app, ["tune", "--model", "unused", "--execute"])
    assert result.exit_code == 2, _failure(result)
    assert "no engine was started" in result.stdout
    assert "dry run" not in result.stdout


def test_tune_execute_without_the_engine_starts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from infertune import adapters
    from infertune.cli import app

    def unavailable(engine: str, schema: object = None) -> object:
        raise RuntimeError(f"{engine} is not installed")

    monkeypatch.setattr(adapters, "get_adapter", unavailable)
    result = runner.invoke(app, ["tune", "--model", "m", "--gpu", "a10", "--execute"])
    assert result.exit_code == 2
    assert "not usable here" in result.stdout
    assert "no engine was started" in result.stdout


def test_tune_execute_end_to_end(
    tmp_path: Path, schema: ParamSchema, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full command: prune, boot each survivor, measure, record, pick and print a winner."""
    from infertune import adapters, models
    from infertune.cli import app

    monkeypatch.setattr(adapters, "get_adapter", lambda engine, schema_=None: VLLMAdapter(schema))
    monkeypatch.setattr(models, "analyze", lambda model: LLAMA_31_8B)
    # ~40 ms per token is what an 8B bf16 model costs on an A10's 600 GB/s, so the second run
    # can calibrate from these measurements instead of rejecting them as implausible.
    for key, value in fake_engine.env(FAKE_ENGINE_TPOT_S="0.04").items():
        monkeypatch.setenv(key, value)

    store = tmp_path / "runs.db"
    args = [
        "tune",
        "--model",
        LLAMA_31_8B.model_id,
        # A10 with the default workload: few candidates and little KV headroom, so the
        # analytic pruning that precedes the boots stays fast.
        "--gpu",
        "a10",
        "--boots",
        "2",
        "--execute",
        "--store",
        str(store),
        "--log-dir",
        str(tmp_path / "logs"),
        "--executable",
        EXECUTABLE,
        "--concurrencies",
        "1,2",
        "--requests",
        "4",
        "--output-median",
        "4",
        "--output-p95",
        "8",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout
    assert "winner" in result.stdout
    assert "tests.fake_engine serve" in result.stdout
    assert len(MeasurementStore(store).runs()) == 2

    # A second run calibrates from the first one's records.
    again = runner.invoke(app, [a for a in args if a != "--execute"])
    assert again.exit_code == 0, again.stdout
    assert "calibrated tp=1" in again.stdout
