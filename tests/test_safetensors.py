"""Weight measurement from safetensors metadata.

Hermetic: headers are synthesised in-memory or written to tmp files, so no network.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from infertune.models.safetensors import (
    ModelMetadataError,
    measure_local,
    parse_header,
    read_safetensors_header,
    tensors_from_header,
)


def make_header(tensors: dict[str, tuple[str, list[int]]]) -> dict[str, Any]:
    """Build a safetensors header with contiguous data_offsets."""
    bits = {"BF16": 16, "F16": 16, "F32": 32, "F8_E4M3": 8, "I32": 32, "U8": 8}
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        elements = 1
        for dim in shape:
            elements *= dim
        n_bytes = elements * bits[dtype] // 8
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n_bytes]}
        offset += n_bytes
    return header


def write_safetensors(path: Path, header: dict[str, Any]) -> None:
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0")


def test_parse_header_strips_metadata() -> None:
    header = parse_header(json.dumps({"__metadata__": {"format": "pt"}, "a": {}}).encode())
    assert "__metadata__" not in header
    assert "a" in header


def test_parse_header_rejects_malformed_json() -> None:
    with pytest.raises(ModelMetadataError, match="malformed"):
        parse_header(b"{not json")


def test_parse_header_rejects_non_object() -> None:
    with pytest.raises(ModelMetadataError, match="must be a JSON object"):
        parse_header(b"[1, 2, 3]")


def test_bytes_come_from_data_offsets_not_dtype_maths() -> None:
    """Offsets are authoritative, so unknown dtypes still measure exactly."""
    header = {"w": {"dtype": "SOME_FUTURE_DTYPE", "shape": [10, 10], "data_offsets": [0, 700]}}
    tensors = tensors_from_header(header)
    assert tensors[0].n_bytes == 700
    assert tensors[0].n_elements == 100


def test_unknown_dtype_without_offsets_refuses() -> None:
    with pytest.raises(ModelMetadataError, match="neither data_offsets nor a known dtype"):
        tensors_from_header({"w": {"dtype": "MYSTERY", "shape": [4]}})


def test_empty_header_refuses() -> None:
    with pytest.raises(ModelMetadataError, match="no tensors"):
        tensors_from_header({})


def test_reads_only_the_header(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    header = make_header({"a.weight": ("BF16", [4, 4])})
    raw = json.dumps(header).encode()
    # Append a large body to prove it is never read.
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\xff" * (4 * 1024 * 1024))
    parsed = read_safetensors_header(str(path))
    assert set(parsed) == {"a.weight"}


def test_implausible_header_length_refuses(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 60) + b"{}")
    with pytest.raises(ModelMetadataError, match="implausible"):
        read_safetensors_header(str(path))


def test_missing_file_refuses() -> None:
    with pytest.raises(ModelMetadataError, match="no such file"):
        read_safetensors_header("/nonexistent/model.safetensors")


class TestTiedEmbeddingDedup:
    """The 26% error.

    Qwen3-0.6B declares ``tie_word_embeddings: true`` *and* ships both ``lm_head.weight``
    and ``model.embed_tokens.weight``. Only one copy is resident at inference time, so
    summing every tensor overstates resident weights by a whole embedding matrix.
    """

    def build(self, tmp_path: Path) -> Path:
        # Qwen3-0.6B geometry: vocab 151936 x hidden 1024, bf16.
        header = make_header(
            {
                "lm_head.weight": ("BF16", [151936, 1024]),
                "model.embed_tokens.weight": ("BF16", [151936, 1024]),
                "model.layers.0.self_attn.q_proj.weight": ("BF16", [2048, 1024]),
            }
        )
        write_safetensors(tmp_path / "model.safetensors", header)
        return tmp_path

    def test_dedup_removes_exactly_one_embedding_matrix(self, tmp_path: Path) -> None:
        directory = self.build(tmp_path)
        tied = measure_local(directory, tied_embeddings=True)
        untied = measure_local(directory, tied_embeddings=False)

        embedding_bytes = 151936 * 1024 * 2
        assert untied.total_bytes - tied.total_bytes == embedding_bytes
        assert tied.tied_dedup_bytes == embedding_bytes
        assert tied.raw_bytes == untied.total_bytes

    def test_dedup_error_would_exceed_25_percent(self, tmp_path: Path) -> None:
        directory = self.build(tmp_path)
        tied = measure_local(directory, tied_embeddings=True)
        overstatement = tied.raw_bytes / tied.total_bytes
        assert overstatement > 1.25, "this is the error the dedup exists to remove"

    def test_dedup_also_corrects_the_parameter_count(self, tmp_path: Path) -> None:
        directory = self.build(tmp_path)
        tied = measure_local(directory, tied_embeddings=True)
        untied = measure_local(directory, tied_embeddings=False)
        assert untied.n_params - tied.n_params == 151936 * 1024

    def test_dedup_is_explained_in_warnings(self, tmp_path: Path) -> None:
        tied = measure_local(self.build(tmp_path), tied_embeddings=True)
        assert any("tie_word_embeddings" in w for w in tied.warnings)

    def test_no_dedup_when_lm_head_is_absent(self, tmp_path: Path) -> None:
        """Most tied checkpoints simply omit lm_head; then nothing must be subtracted."""
        header = make_header({"model.embed_tokens.weight": ("BF16", [1000, 64])})
        write_safetensors(tmp_path / "model.safetensors", header)
        measurement = measure_local(tmp_path, tied_embeddings=True)
        assert measurement.tied_dedup_bytes == 0
        assert measurement.total_bytes == measurement.raw_bytes
        assert not measurement.warnings


def test_sharded_local_measurement(tmp_path: Path) -> None:
    for index, name in enumerate(["a.weight", "b.weight"], start=1):
        write_safetensors(
            tmp_path / f"model-0000{index}-of-00002.safetensors",
            make_header({name: ("BF16", [100, 100])}),
        )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 40_000},
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                },
            }
        )
    )
    measurement = measure_local(tmp_path)
    assert measurement.n_shards == 2
    assert measurement.total_bytes == 2 * 100 * 100 * 2
    assert measurement.n_tensors == 2


def test_mixed_precision_dtype_histogram(tmp_path: Path) -> None:
    """Quantized checkpoints keep some tensors wide; the histogram must show it."""
    write_safetensors(
        tmp_path / "model.safetensors",
        make_header(
            {
                "model.embed_tokens.weight": ("BF16", [1000, 64]),
                "model.layers.0.mlp.down_proj.weight": ("F8_E4M3", [4096, 4096]),
            }
        ),
    )
    measurement = measure_local(tmp_path)
    assert set(measurement.dtype_bytes) == {"BF16", "F8_E4M3"}
    assert measurement.dominant_dtype == "F8_E4M3"


def test_no_checkpoint_refuses(tmp_path: Path) -> None:
    with pytest.raises(ModelMetadataError, match=r"no model\.safetensors"):
        measure_local(tmp_path)
