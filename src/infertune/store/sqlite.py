"""Measurement store.

Every measurement is keyed by ``(gpu, model, engine, engine_version, config)``. The
engine version is part of the key deliberately: vLLM's memory accounting and defaults change
between releases, so a measurement from one version is not evidence about another. Storing
them in one undifferentiated pool is how a calibration set quietly becomes wrong.

SQLite via the standard library, so there is nothing to install and the database is a single
file that can be copied between machines.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    gpu               TEXT    NOT NULL,
    gpu_count         INTEGER NOT NULL DEFAULT 1,
    model             TEXT    NOT NULL,
    engine            TEXT    NOT NULL,
    engine_version    TEXT    NOT NULL,
    tensor_parallel   INTEGER NOT NULL DEFAULT 1,
    max_num_seqs      INTEGER,
    max_model_len     INTEGER,
    kv_dtype          TEXT,
    weight_bytes      INTEGER,
    kv_bytes_per_token INTEGER,
    predicted_kv_bytes INTEGER,
    actual_kv_bytes   INTEGER,
    notes             TEXT    NOT NULL DEFAULT '',
    extra             TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS points (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    concurrency     INTEGER NOT NULL,
    duration_s      REAL    NOT NULL,
    requests_ok     INTEGER NOT NULL,
    requests_failed INTEGER NOT NULL,
    prompt_tokens   INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    ttft_p50_ms     REAL,
    ttft_p99_ms     REAL,
    tpot_p50_ms     REAL,
    tpot_p99_ms     REAL,
    output_tps      REAL    NOT NULL,
    avg_context_tokens INTEGER
);

CREATE INDEX IF NOT EXISTS idx_runs_key
    ON runs(gpu, model, engine, engine_version, tensor_parallel);
CREATE INDEX IF NOT EXISTS idx_points_run ON points(run_id);
"""


@dataclass(frozen=True, slots=True)
class RunKey:
    """What makes measurements comparable.

    Engine version is included because it changes behaviour, not merely cosmetics.
    """

    gpu: str
    model: str
    engine: str
    engine_version: str
    tensor_parallel: int = 1


@dataclass(frozen=True, slots=True)
class StoredPoint:
    """One concurrency level's measured outcome."""

    concurrency: int
    duration_s: float
    requests_ok: int
    requests_failed: int
    prompt_tokens: int
    output_tokens: int
    output_tps: float
    ttft_p50_ms: float | None = None
    ttft_p99_ms: float | None = None
    tpot_p50_ms: float | None = None
    tpot_p99_ms: float | None = None
    avg_context_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StoredRun:
    """A boot and its sweep."""

    key: RunKey
    gpu_count: int = 1
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    kv_dtype: str | None = None
    weight_bytes: int | None = None
    kv_bytes_per_token: int | None = None
    predicted_kv_bytes: int | None = None
    actual_kv_bytes: int | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    points: tuple[StoredPoint, ...] = field(default_factory=tuple)
    run_id: int | None = None

    @property
    def kv_error_fraction(self) -> float | None:
        """Relative KV prediction error, when both figures are present."""
        if not self.predicted_kv_bytes or not self.actual_kv_bytes:
            return None
        return abs(self.predicted_kv_bytes - self.actual_kv_bytes) / self.actual_kv_bytes


class MeasurementStore:
    """A local SQLite database of benchmark measurements."""

    def __init__(self, path: str | Path = "infertune.db") -> None:
        self.path = Path(path)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute("SELECT version FROM schema_info").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
            elif row["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.path}: schema version {row['version']} != expected "
                    f"{SCHEMA_VERSION}; migrate or use a fresh database"
                )

    def record(self, run: StoredRun) -> int:
        """Persist a run and its points. Returns the run id."""
        with self._connect() as conn, closing(conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO runs (gpu, gpu_count, model, engine, engine_version,
                                  tensor_parallel, max_num_seqs, max_model_len, kv_dtype,
                                  weight_bytes, kv_bytes_per_token, predicted_kv_bytes,
                                  actual_kv_bytes, notes, extra)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run.key.gpu,
                    run.gpu_count,
                    run.key.model,
                    run.key.engine,
                    run.key.engine_version,
                    run.key.tensor_parallel,
                    run.max_num_seqs,
                    run.max_model_len,
                    run.kv_dtype,
                    run.weight_bytes,
                    run.kv_bytes_per_token,
                    run.predicted_kv_bytes,
                    run.actual_kv_bytes,
                    run.notes,
                    json.dumps(run.extra, sort_keys=True),
                ),
            )
            run_id = int(cur.lastrowid or 0)
            cur.executemany(
                """
                INSERT INTO points (run_id, concurrency, duration_s, requests_ok,
                                    requests_failed, prompt_tokens, output_tokens,
                                    ttft_p50_ms, ttft_p99_ms, tpot_p50_ms, tpot_p99_ms,
                                    output_tps, avg_context_tokens)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        run_id,
                        p.concurrency,
                        p.duration_s,
                        p.requests_ok,
                        p.requests_failed,
                        p.prompt_tokens,
                        p.output_tokens,
                        p.ttft_p50_ms,
                        p.ttft_p99_ms,
                        p.tpot_p50_ms,
                        p.tpot_p99_ms,
                        p.output_tps,
                        p.avg_context_tokens,
                    )
                    for p in run.points
                ],
            )
            return run_id

    def runs(
        self, key: RunKey | None = None, *, engine_version: str | None = None
    ) -> list[StoredRun]:
        """Fetch runs, optionally narrowed to a comparable set."""
        sql = "SELECT * FROM runs"
        params: list[Any] = []
        clauses: list[str] = []
        if key is not None:
            clauses += [
                "gpu = ?",
                "model = ?",
                "engine = ?",
                "engine_version = ?",
                "tensor_parallel = ?",
            ]
            params += [key.gpu, key.model, key.engine, key.engine_version, key.tensor_parallel]
        elif engine_version is not None:
            clauses.append("engine_version = ?")
            params.append(engine_version)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            out: list[StoredRun] = []
            for row in rows:
                pts = conn.execute(
                    "SELECT * FROM points WHERE run_id = ? ORDER BY concurrency", (row["id"],)
                ).fetchall()
                out.append(
                    StoredRun(
                        key=RunKey(
                            gpu=row["gpu"],
                            model=row["model"],
                            engine=row["engine"],
                            engine_version=row["engine_version"],
                            tensor_parallel=row["tensor_parallel"],
                        ),
                        gpu_count=row["gpu_count"],
                        max_num_seqs=row["max_num_seqs"],
                        max_model_len=row["max_model_len"],
                        kv_dtype=row["kv_dtype"],
                        weight_bytes=row["weight_bytes"],
                        kv_bytes_per_token=row["kv_bytes_per_token"],
                        predicted_kv_bytes=row["predicted_kv_bytes"],
                        actual_kv_bytes=row["actual_kv_bytes"],
                        notes=row["notes"],
                        extra=json.loads(row["extra"]),
                        points=tuple(
                            StoredPoint(
                                concurrency=p["concurrency"],
                                duration_s=p["duration_s"],
                                requests_ok=p["requests_ok"],
                                requests_failed=p["requests_failed"],
                                prompt_tokens=p["prompt_tokens"],
                                output_tokens=p["output_tokens"],
                                output_tps=p["output_tps"],
                                ttft_p50_ms=p["ttft_p50_ms"],
                                ttft_p99_ms=p["ttft_p99_ms"],
                                tpot_p50_ms=p["tpot_p50_ms"],
                                tpot_p99_ms=p["tpot_p99_ms"],
                                avg_context_tokens=p["avg_context_tokens"],
                            )
                            for p in pts
                        ),
                        run_id=row["id"],
                    )
                )
            return out

    def count(self) -> tuple[int, int]:
        """(runs, points) currently stored."""
        with self._connect() as conn:
            r = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
            p = conn.execute("SELECT COUNT(*) AS n FROM points").fetchone()["n"]
            return int(r), int(p)

    def to_json(self) -> str:
        return json.dumps([asdict(r) for r in self.runs()], indent=2, default=str)


__all__ = [
    "SCHEMA_VERSION",
    "MeasurementStore",
    "RunKey",
    "StoredPoint",
    "StoredRun",
]
