"""Model metadata analysis.

Unlike :mod:`infertune.core`, this layer performs I/O (HTTP range requests against the
Hugging Face Hub, or local file reads). It still requires no GPU and no model download.
"""

from __future__ import annotations

from .hf import (
    UnknownArchitectureError,
    analyze,
    build_cache_spec,
    build_moe_spec,
    build_quant_spec,
    fetch_config,
    profile_from_config,
    text_config,
)
from .safetensors import (
    ModelMetadataError,
    TensorInfo,
    WeightMeasurement,
    measure_hub,
    measure_local,
    parse_header,
    read_safetensors_header,
    tensors_from_header,
)

__all__ = [
    "ModelMetadataError",
    "TensorInfo",
    "UnknownArchitectureError",
    "WeightMeasurement",
    "analyze",
    "build_cache_spec",
    "build_moe_spec",
    "build_quant_spec",
    "fetch_config",
    "measure_hub",
    "measure_local",
    "parse_header",
    "profile_from_config",
    "read_safetensors_header",
    "tensors_from_header",
    "text_config",
]
