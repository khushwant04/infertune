"""A minimal OpenAI-compatible streaming server for testing the load generator.

Standard library only. Emits SSE chunks with *controllable* delays, which is what makes the
timing logic verifiable: if the server waits 80 ms before the first chunk and 20 ms between
subsequent ones, then a correct client must report TTFT ~= 80 ms and TPOT ~= 20 ms. Without
this, the harness could only be tested against a real GPU.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MockEngine:
    """Controllable fake engine.

    Args:
        ttft_s: delay before the first token.
        tpot_s: delay between subsequent tokens.
        fail_above_concurrency: return HTTP 503 when more than this many requests are in
            flight, to exercise the harness's error-rate handling.
        status: force a status code for every request.
        reported_prompt_tokens: value to report in the usage chunk when the client asks for
            usage. Chosen by the test to be distinguishable from the client's own estimate.
    """

    def __init__(
        self,
        *,
        ttft_s: float = 0.02,
        tpot_s: float = 0.005,
        fail_above_concurrency: int | None = None,
        status: int = 200,
        reported_prompt_tokens: int = 4242,
    ) -> None:
        self.ttft_s = ttft_s
        self.tpot_s = tpot_s
        self.fail_above_concurrency = fail_above_concurrency
        self.status = status
        self.reported_prompt_tokens = reported_prompt_tokens
        self.requests: list[dict[str, Any]] = []
        self.max_observed_concurrency = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self._server is not None, "server not started"
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{int(port)}"

    def __enter__(self) -> MockEngine:
        engine = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")

                with engine._lock:
                    engine._in_flight += 1
                    engine.max_observed_concurrency = max(
                        engine.max_observed_concurrency, engine._in_flight
                    )
                    engine.requests.append(payload)
                    over = (
                        engine.fail_above_concurrency is not None
                        and engine._in_flight > engine.fail_above_concurrency
                    )
                try:
                    if engine.status != 200 or over:
                        code = 503 if over else engine.status
                        body = b'{"error":"overloaded"}'
                        self.send_response(code)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return

                    n = int(payload.get("max_tokens", 4))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    time.sleep(engine.ttft_s)
                    for i in range(n):
                        if i:
                            time.sleep(engine.tpot_s)
                        chunk = json.dumps({"choices": [{"text": f" t{i}", "index": 0}]}).encode()
                        frame = b"data: " + chunk + b"\n\n"
                        self.wfile.write(f"{len(frame):X}\r\n".encode() + frame + b"\r\n")
                        self.wfile.flush()
                    if (payload.get("stream_options") or {}).get("include_usage"):
                        # Real engines send a final chunk with no choices, carrying exact
                        # token accounting. The value is deliberately not derivable from
                        # the request so tests can prove the client reads it.
                        usage = json.dumps(
                            {
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": engine.reported_prompt_tokens,
                                    "completion_tokens": n,
                                    "total_tokens": engine.reported_prompt_tokens + n,
                                },
                            }
                        ).encode()
                        frame = b"data: " + usage + b"\n\n"
                        self.wfile.write(f"{len(frame):X}\r\n".encode() + frame + b"\r\n")
                        self.wfile.flush()
                    done = b"data: [DONE]\n\n"
                    self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                finally:
                    with engine._lock:
                        engine._in_flight -= 1

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


__all__ = ["MockEngine"]
