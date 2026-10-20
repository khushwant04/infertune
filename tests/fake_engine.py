"""A stand-in for ``vllm serve`` that runs as a real subprocess.

Run as ``python -m tests.fake_engine`` with the repository root on ``PYTHONPATH`` (see
:func:`command` and :func:`env`). Accepts any command line, as the real engine would, and
reads only ``--host``, ``--port`` and ``--max-num-seqs``. It replays a recorded vLLM startup
log, so the adapter's log parser sees real engine output, then serves the OpenAI-compatible
streaming API from :mod:`tests.mock_engine` until it is signalled.

Behaviour is steered through environment variables, because the launcher passes the environment
through untouched:

* ``FAKE_ENGINE_OOM_ABOVE_SEQS`` — exit 1 with a CUDA OOM message when ``--max-num-seqs`` is
  larger, imitating a boot that fails during graph capture.
* ``FAKE_ENGINE_HANG`` — never start serving, imitating a stalled boot.
* ``FAKE_ENGINE_TTFT_S`` / ``FAKE_ENGINE_TPOT_S`` — first-token and inter-token delays, so a
  test can produce latencies that are physically plausible for the model it claims to serve.
* ``FAKE_ENGINE_CHILD_PIDFILE`` — fork a long-lived child and write its pid here, imitating the
  worker processes a tensor-parallel engine spawns.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from tests.mock_engine import MockEngine

HERE = Path(__file__).resolve().parent
STARTUP_LOG = HERE / "fixtures" / "vllm" / "startup_qwen3-0.6b_0.19.1.log"
ROOT = HERE.parent


def command(*args: str) -> list[str]:
    """Argv that runs this module; pair it with :func:`env`."""
    return [sys.executable, "-m", "tests.fake_engine", *args]


def env(**overrides: str) -> dict[str, str]:
    """Environment the fake engine needs: the repository root importable, plus overrides."""
    path = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
    return {"PYTHONPATH": path, **overrides}


def _arg(argv: list[str], flag: str, default: str) -> str:
    return argv[argv.index(flag) + 1] if flag in argv else default


def main() -> int:
    argv = sys.argv[1:]
    host = _arg(argv, "--host", "127.0.0.1")
    port = int(_arg(argv, "--port", "8000"))
    seqs = int(_arg(argv, "--max-num-seqs", "256"))

    print(STARTUP_LOG.read_text(), flush=True)

    pidfile = os.environ.get("FAKE_ENGINE_CHILD_PIDFILE")
    if pidfile:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        Path(pidfile).write_text(str(child.pid))

    limit = os.environ.get("FAKE_ENGINE_OOM_ABOVE_SEQS")
    if limit is not None and seqs > int(limit):
        print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", flush=True)
        return 1

    if os.environ.get("FAKE_ENGINE_HANG"):
        time.sleep(300)
        return 0

    ttft = float(os.environ.get("FAKE_ENGINE_TTFT_S", "0.002"))
    tpot = float(os.environ.get("FAKE_ENGINE_TPOT_S", "0.0005"))
    with MockEngine(ttft_s=ttft, tpot_s=tpot, host=host, port=port):
        print(f"INFO: Uvicorn running on http://{host}:{port}", flush=True)
        while True:
            time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
