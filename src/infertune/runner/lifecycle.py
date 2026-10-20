"""Engine process lifecycle: launch, wait for readiness, capture logs, stop.

Three failure modes matter, and each must be reported as what it is rather than as a slow or
empty benchmark:

* **The engine exits during startup.** An OOM in weight loading or CUDA-graph capture kills the
  process within seconds to minutes. Polling only the health endpoint would wait out the full
  readiness timeout for a server that no longer exists, so the process is polled as well.
* **The engine never becomes ready.** A hung download or a deadlocked worker. Bounded by a
  timeout, after which the process group is stopped.
* **The engine outlives its caller.** Tensor-parallel engines fork worker processes. Signalling
  only the parent leaves orphans holding GPU memory, which then makes the *next* boot OOM. Every
  engine therefore starts in its own session and is stopped as a process group.

The engine's combined stdout/stderr goes to a file, not a pipe: a pipe nobody drains fills its
buffer and blocks the engine, and the log is the ground truth the memory ledger is scored
against, so it has to survive the run.
"""

from __future__ import annotations

import http.client
import os
import shlex
import signal
import socket
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from urllib.parse import urlparse

LOG_TAIL_LINES = 40


class EngineStartError(RuntimeError):
    """The engine did not become ready. Carries the end of its log for diagnosis."""

    def __init__(self, message: str, *, log_tail: str = "", exit_code: int | None = None) -> None:
        super().__init__(message)
        self.log_tail = log_tail
        self.exit_code = exit_code


@dataclass(frozen=True, slots=True)
class EngineLaunch:
    """Everything needed to start one engine and know when it is ready."""

    command: tuple[str, ...]
    base_url: str
    log_path: Path
    env: Mapping[str, str] | None = None
    """Extra environment variables, layered over the current environment."""

    ready_timeout_s: float = 900.0
    """Weight loading, ``torch.compile`` and graph capture take minutes on a cold cache."""

    poll_interval_s: float = 2.0
    health_path: str = "/health"
    stop_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command must be non-empty")
        if self.ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be > 0")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")


def free_port(host: str = "127.0.0.1") -> int:
    """A port that was free a moment ago.

    There is an unavoidable race between closing the probe socket and the engine binding it,
    but each boot uses a fresh port, so a collision costs one failed boot rather than a
    benchmark against the wrong server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


def tail(path: Path, lines: int = LOG_TAIL_LINES) -> str:
    """Last ``lines`` lines of a log file, or an empty string when it does not exist."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def is_healthy(base_url: str, path: str = "/health", timeout_s: float = 5.0) -> bool:
    """True when the server answers ``path`` with HTTP 200.

    Stricter than "any response": SGLang answers 503 on ``/health`` while it is still warming
    up, and a benchmark started then would measure the warm-up.
    """
    parsed = urlparse(base_url)
    secure = parsed.scheme == "https"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if secure else 80)
    cls = http.client.HTTPSConnection if secure else http.client.HTTPConnection
    try:
        conn = cls(host, port, timeout=timeout_s)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            response.read()
            return response.status == 200
        finally:
            conn.close()
    except (OSError, http.client.HTTPException):
        return False


class RunningEngine:
    """A started engine process. Use as a context manager so it is always stopped."""

    def __init__(self, launch: EngineLaunch, process: subprocess.Popen[bytes]) -> None:
        self.launch = launch
        self.process = process
        self.started_at = time.monotonic()
        self.ready_after_s: float | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    def is_running(self) -> bool:
        return self.process.poll() is None

    def log_text(self) -> str:
        try:
            return self.launch.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def wait_until_ready(self) -> float:
        """Block until healthy. Returns seconds taken; raises if the engine dies or stalls."""
        launch = self.launch
        deadline = self.started_at + launch.ready_timeout_s
        while True:
            code = self.process.poll()
            if code is not None:
                raise EngineStartError(
                    f"engine exited with code {code} before becoming ready "
                    f"(log: {launch.log_path})",
                    log_tail=tail(launch.log_path),
                    exit_code=code,
                )
            if is_healthy(launch.base_url, launch.health_path):
                self.ready_after_s = time.monotonic() - self.started_at
                return self.ready_after_s
            if time.monotonic() >= deadline:
                self.stop()
                raise EngineStartError(
                    f"engine not ready after {launch.ready_timeout_s:.0f}s; stopped it "
                    f"(log: {launch.log_path})",
                    log_tail=tail(launch.log_path),
                )
            time.sleep(launch.poll_interval_s)

    def stop(self) -> int | None:
        """Stop the whole process group: SIGTERM, then SIGKILL after the grace period."""
        if self.process.poll() is None:
            self._signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=self.launch.stop_timeout_s)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGKILL)
                self.process.wait(timeout=self.launch.stop_timeout_s)
        # Workers forked by the engine may outlive the parent; sweep the group regardless.
        self._signal(signal.SIGKILL)
        return self.process.returncode

    def _signal(self, sig: signal.Signals) -> None:
        try:
            if hasattr(os, "killpg"):
                os.killpg(self.process.pid, sig)
            else:  # pragma: no cover - non-POSIX
                self.process.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            pass

    def __enter__(self) -> RunningEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()


def start_engine(launch: EngineLaunch) -> RunningEngine:
    """Start an engine and return once it is ready to serve.

    The caller owns the returned engine and must stop it; the context-manager form does so even
    when the benchmark raises.
    """
    launch.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **(launch.env or {})}
    with launch.log_path.open("wb") as log:
        log.write(f"$ {shlex.join(launch.command)}\n".encode())
        log.flush()
        try:
            process = subprocess.Popen(
                list(launch.command),
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
        except OSError as exc:
            raise EngineStartError(
                f"could not start {launch.command[0]!r}: {exc}", log_tail=tail(launch.log_path)
            ) from exc
    engine = RunningEngine(launch, process)
    try:
        engine.wait_until_ready()
    except BaseException:
        engine.stop()
        raise
    return engine


__all__ = [
    "EngineLaunch",
    "EngineStartError",
    "RunningEngine",
    "free_port",
    "is_healthy",
    "start_engine",
    "tail",
]
