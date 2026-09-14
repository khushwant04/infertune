"""Exact weight measurement from safetensors metadata, without downloading weights.

A safetensors file begins with an 8-byte little-endian header length followed by a JSON
header describing every tensor: dtype, shape, and byte offsets. That header is fetchable
with an HTTP range request, so the *exact* resident weight size of any model on the Hub is
obtainable in about a megabyte of traffic
(`HF metadata parsing <https://huggingface.co/docs/safetensors/en/metadata_parsing>`_).

This matters because ``n_params * bytes_per_dtype`` is wrong for precisely the checkpoints
people most need help sizing:

* **Quantized checkpoints** leave embeddings, norms, routers and often ``lm_head`` at full
  precision, so the nominal bit width overstates compression.
* **Tied embeddings** are the big one. Qwen3-0.6B declares ``tie_word_embeddings: true``
  *and* ships both ``lm_head.weight`` and ``model.embed_tokens.weight``. Summing every
  tensor therefore double-counts one 151936x1024 matrix — a **26.1% overestimate** on that
  model. Only one copy is resident at inference time.
* **MoE checkpoints** need total parameters for memory but active parameters for compute.

Measuring, rather than deriving, removes all three sources of error at once.
"""

from __future__ import annotations

import json
import math
import os
import struct
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
_HEADER_LEN_SIZE = 8
_MAX_HEADER_BYTES = 128 * 1024 * 1024
_TIMEOUT_S = 30.0

# safetensors dtype -> bits
_DTYPE_BITS: dict[str, int] = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
}


class ModelMetadataError(RuntimeError):
    """Raised when checkpoint metadata cannot be read or understood."""


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    n_bytes: int

    @property
    def n_elements(self) -> int:
        return math.prod(self.shape) if self.shape else 1


@dataclass(frozen=True, slots=True)
class WeightMeasurement:
    """Measured resident weight size for a checkpoint."""

    total_bytes: int
    """Resident bytes after deduplicating tied embeddings."""

    raw_bytes: int
    """Sum of all tensor bytes as stored on disk, before deduplication."""

    n_params: int
    dtype_bytes: dict[str, int] = field(default_factory=dict)
    tied_dedup_bytes: int = 0
    n_tensors: int = 0
    n_shards: int = 1
    source: str = "safetensors-header"
    exact_params: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dominant_dtype(self) -> str:
        if not self.dtype_bytes:
            return "unknown"
        return max(self.dtype_bytes.items(), key=lambda kv: kv[1])[0]

    @property
    def bytes_per_param(self) -> float:
        return self.total_bytes / self.n_params if self.n_params else 0.0


def _fetch_range(url: str, start: int, end: int, *, token: str | None = None) -> bytes:
    """Fetch ``[start, end]`` inclusive via an HTTP range request."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ModelMetadataError(
                f"access to {url} is restricted (HTTP {exc.code}). Set HF_TOKEN for gated "
                "repositories."
            ) from exc
        if exc.code == 404:
            raise ModelMetadataError(f"not found: {url}") from exc
        raise ModelMetadataError(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise ModelMetadataError(f"network error fetching {url}: {exc.reason}") from exc


def _fetch_all(url: str, *, token: str | None = None) -> bytes:
    """Fetch a whole file.

    Used for shard indexes, which must never be truncated: DeepSeek-V3's index is ~8 MB
    across 91,991 tensors, and a range-capped read yields invalid JSON that would silently
    fall back to the single-file path.
    """
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ModelMetadataError(
                f"access to {url} is restricted (HTTP {exc.code}). Set HF_TOKEN for gated "
                "repositories."
            ) from exc
        if exc.code == 404:
            raise ModelMetadataError(f"not found: {url}") from exc
        raise ModelMetadataError(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise ModelMetadataError(f"network error fetching {url}: {exc.reason}") from exc


def _read_local_range(path: Path, start: int, end: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(start)
        return handle.read(end - start + 1)


def _read_bytes(location: str, start: int, end: int, *, token: str | None = None) -> bytes:
    if location.startswith(("http://", "https://")):
        return _fetch_range(location, start, end, token=token)
    path = Path(location)
    if not path.is_file():
        raise ModelMetadataError(f"no such file: {location}")
    return _read_local_range(path, start, end)


def parse_header(raw: bytes) -> dict[str, Any]:
    """Parse a safetensors JSON header from its raw bytes, dropping ``__metadata__``."""
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelMetadataError(f"malformed safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise ModelMetadataError("safetensors header must be a JSON object")
    header.pop("__metadata__", None)
    return header


def read_safetensors_header(location: str, *, token: str | None = None) -> dict[str, Any]:
    """Read only the JSON header of a safetensors file.

    Two small reads: eight bytes for the header length, then the header itself.
    """
    length_bytes = _read_bytes(location, 0, _HEADER_LEN_SIZE - 1, token=token)
    if len(length_bytes) < _HEADER_LEN_SIZE:
        raise ModelMetadataError(f"{location}: truncated safetensors header length")
    (header_len,) = struct.unpack("<Q", length_bytes)
    if not 0 < header_len <= _MAX_HEADER_BYTES:
        raise ModelMetadataError(f"{location}: implausible safetensors header length {header_len}")
    raw = _read_bytes(location, _HEADER_LEN_SIZE, _HEADER_LEN_SIZE + header_len - 1, token=token)
    return parse_header(raw)


def tensors_from_header(header: dict[str, Any]) -> list[TensorInfo]:
    """Convert a parsed header into :class:`TensorInfo` records.

    Byte counts come from ``data_offsets`` rather than ``dtype x shape``, so they are exact
    even for dtypes this module does not recognise.
    """
    tensors: list[TensorInfo] = []
    for name, info in header.items():
        if not isinstance(info, dict):
            continue
        offsets = info.get("data_offsets")
        shape = tuple(int(d) for d in info.get("shape", ()))
        dtype = str(info.get("dtype", "unknown"))
        if isinstance(offsets, list) and len(offsets) == 2:
            n_bytes = int(offsets[1]) - int(offsets[0])
        else:
            bits = _DTYPE_BITS.get(dtype)
            if bits is None:
                raise ModelMetadataError(
                    f"tensor {name!r} has neither data_offsets nor a known dtype ({dtype})"
                )
            n_bytes = -(-(math.prod(shape) if shape else 1) * bits // 8)
        if n_bytes < 0:
            raise ModelMetadataError(f"tensor {name!r} has negative size")
        tensors.append(TensorInfo(name=name, dtype=dtype, shape=shape, n_bytes=n_bytes))
    if not tensors:
        raise ModelMetadataError("safetensors header contained no tensors")
    return tensors


def _is_lm_head(name: str) -> bool:
    return name in ("lm_head.weight", "output.weight") or name.endswith(".lm_head.weight")


def _summarise(
    tensors: list[TensorInfo],
    *,
    tied_embeddings: bool,
    n_shards: int,
    source: str,
) -> WeightMeasurement:
    raw_bytes = sum(t.n_bytes for t in tensors)
    n_params = sum(t.n_elements for t in tensors)
    dtype_bytes: dict[str, int] = {}
    for tensor in tensors:
        dtype_bytes[tensor.dtype] = dtype_bytes.get(tensor.dtype, 0) + tensor.n_bytes

    dedup = 0
    warnings: list[str] = []
    if tied_embeddings:
        head = next((t for t in tensors if _is_lm_head(t.name)), None)
        if head is not None:
            dedup = head.n_bytes
            n_params -= head.n_elements
            dtype_bytes[head.dtype] = dtype_bytes.get(head.dtype, 0) - head.n_bytes
            warnings.append(
                f"tie_word_embeddings is set and the checkpoint also stores {head.name!r}; "
                f"deduplicated {dedup:,} bytes ({dedup / raw_bytes * 100:.1f}% of the "
                "checkpoint) because only one copy is resident at inference time"
            )

    return WeightMeasurement(
        total_bytes=raw_bytes - dedup,
        raw_bytes=raw_bytes,
        n_params=n_params,
        dtype_bytes={k: v for k, v in dtype_bytes.items() if v > 0},
        tied_dedup_bytes=dedup,
        n_tensors=len(tensors),
        n_shards=n_shards,
        source=source,
        exact_params=True,
        warnings=tuple(warnings),
    )


def measure_local(directory: str | Path, *, tied_embeddings: bool = False) -> WeightMeasurement:
    """Measure a checkpoint directory on disk."""
    path = Path(directory)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
        tensors: list[TensorInfo] = []
        for shard in shards:
            tensors.extend(tensors_from_header(read_safetensors_header(str(path / shard))))
        return _summarise(
            tensors,
            tied_embeddings=tied_embeddings,
            n_shards=len(shards),
            source="safetensors-header (local, sharded)",
        )

    single = path / "model.safetensors"
    if not single.is_file():
        raise ModelMetadataError(f"{path}: no model.safetensors or index found")
    tensors = tensors_from_header(read_safetensors_header(str(single)))
    return _summarise(
        tensors,
        tied_embeddings=tied_embeddings,
        n_shards=1,
        source="safetensors-header (local)",
    )


def measure_hub(
    repo_id: str,
    *,
    revision: str = "main",
    tied_embeddings: bool = False,
    token: str | None = None,
    max_workers: int = 16,
) -> WeightMeasurement:
    """Measure a checkpoint on the Hugging Face Hub using range requests only.

    Every shard header is read and its ``data_offsets`` summed. Headers are fetched
    concurrently, so even DeepSeek-V3's 163 shards complete in seconds.

    **The index's ``metadata.total_size`` is deliberately not trusted.** For
    deepseek-ai/DeepSeek-V3 it reports 1369 GB against an actual 713 GB of tensor data — a
    1.92x overstatement, because the field is computed as if every tensor were 16-bit and
    the checkpoint is fp8. Any figure derived from it would miss the ±1% target by nearly
    100%. It is still read, and a disagreement is reported as a diagnostic.
    """
    token = token or os.environ.get("HF_TOKEN")
    base = f"{HF_ENDPOINT}/{repo_id}/resolve/{revision}"

    index: dict[str, Any] | None = None
    try:
        index = json.loads(_fetch_all(f"{base}/model.safetensors.index.json", token=token))
    except ModelMetadataError as exc:
        if "restricted" in str(exc):
            raise
    except json.JSONDecodeError as exc:
        raise ModelMetadataError(
            f"{repo_id}: shard index is not valid JSON ({exc}); refusing to fall back to a "
            "single-file read, which would silently measure the wrong thing"
        ) from exc

    if index is None:
        tensors = tensors_from_header(
            read_safetensors_header(f"{base}/model.safetensors", token=token)
        )
        return _summarise(
            tensors, tied_embeddings=tied_embeddings, n_shards=1, source="safetensors-header"
        )

    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise ModelMetadataError(f"{repo_id}: index has no weight_map")
    shards = sorted(set(weight_map.values()))

    def read_shard(shard: str) -> list[TensorInfo]:
        return tensors_from_header(read_safetensors_header(f"{base}/{shard}", token=token))

    all_tensors: list[TensorInfo] = []
    if len(shards) == 1:
        all_tensors = read_shard(shards[0])
    else:
        workers = max(1, min(max_workers, len(shards)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for shard_tensors in pool.map(read_shard, shards):
                all_tensors.extend(shard_tensors)

    measurement = _summarise(
        all_tensors,
        tied_embeddings=tied_embeddings,
        n_shards=len(shards),
        source="safetensors-header (sharded)",
    )

    # Cross-check the index's own claim and report disagreement. This is a valuable signal in
    # its own right: several published checkpoints ship a total_size computed as if all
    # tensors were 16-bit, which is wrong by ~2x for fp8 weights.
    claimed = index.get("metadata", {}).get("total_size")
    if isinstance(claimed, int) and claimed > 0 and measurement.raw_bytes > 0:
        ratio = claimed / measurement.raw_bytes
        if abs(ratio - 1.0) > 0.01:
            return replace(
                measurement,
                warnings=(
                    *measurement.warnings,
                    f"index metadata.total_size claims {claimed:,} bytes but the tensor "
                    f"headers sum to {measurement.raw_bytes:,} ({ratio:.2f}x); using the "
                    "measured value, since total_size is frequently computed as if every "
                    "tensor were 16-bit",
                ),
            )
    return measurement
