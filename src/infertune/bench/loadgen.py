"""Load generator for OpenAI-compatible streaming endpoints.

Threads over ``http.client`` rather than an async HTTP dependency. At the concurrencies that
matter here (single digits to a few hundred) threads are entirely adequate, they keep the
dependency surface at zero, and — the deciding factor — the whole generator is testable
against a ``http.server`` mock, so the harness is verifiable without a GPU.

Streaming is mandatory, not an option: time-to-first-token cannot be measured from a buffered
response, and TTFT is half the point.
"""

from __future__ import annotations

import http.client
import json
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..core.workload import Distribution, WorkloadProfile
from .metrics import BenchmarkResult, RequestRecord

_PROMPT_WORDS: tuple[str, ...] = (
    "the",
    "of",
    "and",
    "to",
    "in",
    "a",
    "is",
    "that",
    "it",
    "for",
    "was",
    "on",
    "are",
    "as",
    "with",
    "his",
    "they",
    "at",
    "be",
    "this",
    "have",
    "from",
    "or",
    "one",
    "had",
    "by",
    "word",
    "but",
    "not",
    "what",
    "all",
    "were",
    "we",
    "when",
    "your",
    "can",
    "said",
    "there",
    "use",
    "an",
    "each",
    "which",
    "she",
    "do",
    "how",
    "their",
    "if",
    "will",
    "up",
    "other",
    "about",
    "out",
    "many",
    "then",
    "them",
    "these",
    "so",
    "some",
    "her",
    "would",
    "make",
    "like",
    "him",
    "into",
    "time",
    "has",
    "look",
    "two",
    "more",
    "write",
    "go",
    "see",
    "number",
    "no",
    "way",
    "could",
    "people",
    "my",
    "than",
    "first",
    "water",
    "been",
    "call",
    "who",
    "oil",
    "its",
    "now",
    "find",
    "long",
    "down",
    "day",
    "did",
    "get",
    "come",
    "made",
    "may",
    "part",
    "over",
    "new",
    "sound",
)
"""Short, common words. See :func:`synthetic_prompt` for why this is not ``word<int>``."""


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """Where and how to send requests."""

    base_url: str = "http://127.0.0.1:8000"
    model: str = ""
    path: str = "/v1/completions"
    api_key: str = ""
    timeout_s: float = 600.0

    def target(self) -> tuple[str, int, bool]:
        parsed = urlparse(self.base_url)
        secure = parsed.scheme == "https"
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if secure else 80)
        return host, port, secure


def synthetic_prompt(n_tokens: int, rng: random.Random) -> str:
    """A prompt of approximately ``n_tokens`` tokens.

    Words are varied so that prefix caching does not silently turn a prefill benchmark into a
    cache-hit benchmark — which would make TTFT look far better than it is.

    The pool is short, common, lower-case words. That matters more than it looks: an earlier
    version emitted ``word<random-int>``, which BPE tokenizers split into roughly 2.7 tokens
    apiece, so a request built for 1500 tokens actually carried about 4100. Requests then
    overflowed ``max_model_len`` and were rejected outright, and every recorded prompt length
    was understated by the same factor. Common words with a leading space are a single token
    in practice, so the requested count and the real count stay close. The exact figure is
    still read back from the server's usage report rather than trusted from here.
    """
    if n_tokens < 1:
        raise ValueError("n_tokens must be >= 1")
    return " ".join(rng.choice(_PROMPT_WORDS) for _ in range(n_tokens))


def _sample_int(dist: Distribution, rng: random.Random, floor: int = 1) -> int:
    return max(floor, int(dist.sample(rng)))


# Words are approximately one token each, but not exactly, so leave room rather than
# targeting the limit precisely.
_CONTEXT_SAFETY_TOKENS = 64


def _fit_to_context(n_in: int, n_out: int, limit: int | None) -> tuple[int, int]:
    """Truncate a sampled (prompt, output) pair so the engine will accept it.

    The prompt is shortened before the output length: output length is what the decode
    measurement is *about*, so trimming it first would bias the very quantity being measured.
    """
    if limit is None:
        return n_in, n_out
    budget = max(2, limit - _CONTEXT_SAFETY_TOKENS)
    n_out = max(1, min(n_out, budget - 1))
    n_in = max(1, min(n_in, budget - n_out))
    return n_in, n_out


@dataclass
class _Counters:
    """Shared request counter, so workers cannot collectively overshoot the target."""

    started: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, limit: int) -> bool:
        with self.lock:
            if self.started >= limit:
                return False
            self.started += 1
            return True


def _stream_one(
    endpoint: EndpointConfig,
    prompt: str,
    max_tokens: int,
    prompt_tokens: int,
) -> RequestRecord:
    """Issue one streaming completion and time the token stream."""
    host, port, secure = endpoint.target()
    body = json.dumps(
        {
            "model": endpoint.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
            # Ask the server to report exact token counts in a final chunk. The
            # client-side estimate is only approximate, and a wrong prompt length feeds
            # straight into the calibration fit as a wrong cache-read volume.
            "stream_options": {"include_usage": True},
        }
    )
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"

    cls = http.client.HTTPSConnection if secure else http.client.HTTPConnection
    conn = cls(host, port, timeout=endpoint.timeout_s)
    started = time.perf_counter()
    ttft: float | None = None
    tokens = 0
    reported_prompt: int | None = None
    reported_output: int | None = None
    try:
        conn.request("POST", endpoint.path, body=body, headers=headers)
        response = conn.getresponse()
        if response.status >= 400:
            detail = response.read(200).decode("utf-8", "replace")
            return RequestRecord(
                prompt_tokens=prompt_tokens,
                output_tokens=0,
                ttft_s=None,
                total_s=time.perf_counter() - started,
                error=f"HTTP {response.status}: {detail}",
            )
        for raw in response:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            # The usage chunk carries no choices, so it must be read before the
            # empty-text guard below discards it.
            if isinstance(usage := chunk.get("usage"), dict):
                if isinstance(value := usage.get("prompt_tokens"), int):
                    reported_prompt = value
                if isinstance(value := usage.get("completion_tokens"), int):
                    reported_output = value
            text = ""
            for choice in chunk.get("choices", ()):
                text += choice.get("text") or (choice.get("delta") or {}).get("content") or ""
            if not text:
                continue
            tokens += 1
            if ttft is None:
                ttft = time.perf_counter() - started
        return RequestRecord(
            prompt_tokens=reported_prompt if reported_prompt is not None else prompt_tokens,
            output_tokens=reported_output if reported_output is not None else tokens,
            ttft_s=ttft,
            total_s=time.perf_counter() - started,
        )
    except (OSError, http.client.HTTPException) as exc:
        return RequestRecord(
            prompt_tokens=prompt_tokens,
            output_tokens=tokens,
            ttft_s=ttft,
            total_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        conn.close()


def run_at_concurrency(
    endpoint: EndpointConfig,
    workload: WorkloadProfile,
    concurrency: int,
    *,
    n_requests: int | None = None,
    duration_s: float | None = None,
    seed: int = 0,
    warmup_requests: int = 0,
    context_limit: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> BenchmarkResult:
    """Drive ``concurrency`` in-flight requests and measure the result.

    Closed-loop by design: exactly ``concurrency`` requests are in flight at all times, each
    worker starting a new one as soon as its previous finishes. That isolates the engine's
    behaviour at a known batch size, which is what the roofline model predicts. An open-loop
    arrival process measures queueing as much as the engine.

    ``context_limit`` is the engine's ``max_model_len``. Sampled lengths are truncated to fit
    within it, because a workload distribution knows nothing about the context window it is
    being replayed against: its tail routinely exceeds the limit, and the engine rejects those
    requests rather than truncating them. Unclamped, the resulting error rate aborts the sweep
    and the failure looks like engine instability rather than a client-side bug.

    Either ``n_requests`` or ``duration_s`` must be given.
    """
    if context_limit is not None and context_limit < 2:
        raise ValueError(f"context_limit must be >= 2, got {context_limit}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if (n_requests is None) == (duration_s is None):
        raise ValueError("pass exactly one of n_requests or duration_s")
    if n_requests is not None and n_requests < 1:
        raise ValueError("n_requests must be >= 1")

    rng_master = random.Random(seed)
    total = n_requests if n_requests is not None else 1_000_000
    counters = _Counters()
    records: list[RequestRecord] = []
    records_lock = threading.Lock()
    deadline = None if duration_s is None else time.perf_counter() + duration_s

    if warmup_requests > 0:
        warm_rng = random.Random(seed - 1)
        for _ in range(warmup_requests):
            n_in, n_out = _fit_to_context(
                _sample_int(workload.input_tokens, warm_rng), 8, context_limit
            )
            _stream_one(endpoint, synthetic_prompt(n_in, warm_rng), n_out, n_in)

    def worker(worker_id: int) -> None:
        rng = random.Random(seed * 1000 + worker_id)
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                return
            if not counters.take(total):
                return
            n_in, n_out = _fit_to_context(
                _sample_int(workload.input_tokens, rng),
                _sample_int(workload.output_tokens, rng),
                context_limit,
            )
            record = _stream_one(endpoint, synthetic_prompt(n_in, rng), n_out, n_in)
            with records_lock:
                records.append(record)
                if progress is not None:
                    progress(len(records))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(concurrency)))
    elapsed = time.perf_counter() - started

    _ = rng_master  # reserved for future arrival-process modelling
    return BenchmarkResult(concurrency=concurrency, duration_s=elapsed, records=tuple(records))


def wait_for_endpoint(
    endpoint: EndpointConfig, *, timeout_s: float = 600.0, interval_s: float = 2.0
) -> bool:
    """Poll until the server answers, so benchmarks do not start mid-load."""
    host, port, secure = endpoint.target()
    cls = http.client.HTTPSConnection if secure else http.client.HTTPConnection
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            conn = cls(host, port, timeout=5.0)
            conn.request("GET", "/health")
            response = conn.getresponse()
            response.read()
            conn.close()
            if response.status < 500:
                return True
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(interval_s)
    return False


__all__ = [
    "EndpointConfig",
    "run_at_concurrency",
    "synthetic_prompt",
    "wait_for_endpoint",
]
