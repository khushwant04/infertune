"""M1 acceptance criteria that require network access.

Excluded from the default run and from CI, which must stay hermetic. Run explicitly:

.. code-block:: shell

    pytest -m network --run-network

The criterion under test is docs/plan.md §9 M1: **weight bytes within ±1% of ground truth**
across checkpoints spanning dense, MoE, MLA, hybrid, and AWQ/fp8/mxfp4 quantization.

Ground truth is the **actual size of the checkpoint files** over HTTP, obtained via ``HEAD``
requests and independent of the header-parsing code under test: for each shard,
``Content-Length`` minus that shard's framing (an 8-byte length prefix plus the JSON header).

Note what is deliberately *not* used as ground truth: the index's ``metadata.total_size``.
For deepseek-ai/DeepSeek-V3 it claims 1369 GB against 713 GB of real tensor data, because it
is computed as if every tensor were 16-bit while the checkpoint is fp8. Validating against it
would both be circular and enshrine a 1.92x error.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

from infertune.core.dtypes import DType
from infertune.models.hf import fetch_config, text_config
from infertune.models.safetensors import HF_ENDPOINT, measure_hub

pytestmark = pytest.mark.network

# (repo, expected architecture family) — chosen to span the cases where naive sizing fails.
CHECKPOINTS = [
    ("Qwen/Qwen3-0.6B", "dense, tied embeddings"),
    ("Qwen/Qwen3-1.7B", "dense, tied embeddings"),
    ("Qwen/Qwen3-4B", "dense"),
    ("Qwen/Qwen3-8B", "dense, sharded"),
    ("Qwen/Qwen3-14B", "dense, sharded"),
    ("Qwen/Qwen2.5-7B-Instruct-AWQ", "awq 4-bit"),
    ("Qwen/Qwen2.5-14B-Instruct-AWQ", "awq 4-bit"),
    ("neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8", "fp8 compressed-tensors"),
    ("openai/gpt-oss-20b", "mxfp4 MoE, interleaved attention"),
    ("Qwen/Qwen3-30B-A3B", "MoE, many shards"),
    ("nvidia/Nemotron-H-8B-Base-8K", "attention/Mamba hybrid"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "dense"),
    ("HuggingFaceTB/SmolLM2-135M", "tiny dense"),
    ("HuggingFaceTB/SmolLM2-1.7B", "dense"),
    ("allenai/OLMo-2-1124-7B", "dense"),
    ("microsoft/phi-4", "dense"),
    ("tiiuae/Falcon3-7B-Instruct", "dense"),
    ("Qwen/Qwen3-0.6B-Base", "dense, tied embeddings"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "dense, tied embeddings"),
    ("google/flan-t5-base", "encoder-decoder"),
]

TOLERANCE = 0.01


def _file_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
    if not length:
        raise RuntimeError(f"no Content-Length for {url}")
    return int(length)


def _framing_bytes(url: str) -> int:
    """Bytes a safetensors file spends on its length prefix and JSON header."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(request, timeout=30) as response:
        (header_len,) = struct.unpack("<Q", response.read(8))
    return 8 + int(header_len)


def _ground_truth_bytes(repo: str) -> tuple[int, str]:
    """Tensor payload size from actual file sizes, independent of the code under test."""
    base = f"{HF_ENDPOINT}/{repo}/resolve/main"
    try:
        with urllib.request.urlopen(f"{base}/model.safetensors.index.json", timeout=30) as r:
            index = json.loads(r.read())
        shards = sorted(set(index["weight_map"].values()))
    except Exception:
        shards = ["model.safetensors"]

    total = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(_file_size, [f"{base}/{s}" for s in shards]))
        framings = list(pool.map(_framing_bytes, [f"{base}/{s}" for s in shards]))
    total = sum(sizes) - sum(framings)
    return total, f"HTTP file sizes across {len(shards)} shard(s), less framing"


@pytest.mark.parametrize(("repo", "family"), CHECKPOINTS, ids=[c[0] for c in CHECKPOINTS])
def test_weight_bytes_within_one_percent(repo: str, family: str) -> None:
    try:
        config = text_config(fetch_config(repo))
        tied = bool(config.get("tie_word_embeddings"))
        measurement = measure_hub(repo, tied_embeddings=tied)
        truth, truth_source = _ground_truth_bytes(repo)
    except Exception as exc:
        pytest.skip(f"{repo} unavailable: {exc}")

    # measure_hub reports *resident* bytes, so add the tied duplicate back to compare
    # against on-disk ground truth.
    on_disk = measurement.total_bytes + measurement.tied_dedup_bytes
    error = abs(on_disk - truth) / truth

    assert error <= TOLERANCE, (
        f"{repo} ({family}): measured {on_disk:,} B vs {truth_source} {truth:,} B "
        f"= {error:.3%} error, exceeds {TOLERANCE:.0%}"
    )


@pytest.mark.parametrize("repo", ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"])
def test_tied_embedding_dedup_reduces_resident_bytes(repo: str) -> None:
    """Resident memory must be below on-disk size when a tied lm_head is shipped."""
    try:
        config = text_config(fetch_config(repo))
        if not config.get("tie_word_embeddings"):
            pytest.skip(f"{repo} does not tie embeddings")
        measurement = measure_hub(repo, tied_embeddings=True)
    except Exception as exc:
        pytest.skip(f"{repo} unavailable: {exc}")

    if measurement.tied_dedup_bytes == 0:
        pytest.skip(f"{repo} omits lm_head, so there is nothing to deduplicate")

    expected = config["vocab_size"] * config["hidden_size"] * 2
    assert measurement.tied_dedup_bytes == pytest.approx(expected, rel=0.02)
    assert measurement.total_bytes < measurement.raw_bytes


def test_deepseek_v3_mla_is_not_sized_with_the_gqa_formula() -> None:
    """End-to-end guard on the 57x MLA correction, against the live config."""
    from infertune.core.cache import AttentionKind, AttentionSpec, MLASpec
    from infertune.models.hf import build_cache_spec

    try:
        config = text_config(fetch_config("deepseek-ai/DeepSeek-V3"))
    except Exception as exc:
        pytest.skip(f"config unavailable: {exc}")

    spec, _ = build_cache_spec(config)
    assert isinstance(spec, MLASpec)

    naive = AttentionSpec(
        kind=AttentionKind.GQA,
        n_kv_heads=config["num_key_value_heads"],
        head_dim=config["v_head_dim"],
        n_kv_layers=config["num_hidden_layers"],
    ).kv_bytes_per_token(DType.BF16)
    actual = spec.marginal_bytes_per_token(DType.BF16)
    assert naive / actual > 50
