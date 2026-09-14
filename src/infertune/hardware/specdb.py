"""GPU specification database for the capacity-planning path.

Lets the profiler answer "what would I need to serve this model?" without the hardware in
hand — which is the highest-value question, and the one that is lost if detection is
mandatory. Stored as JSON rather than YAML so that loading needs nothing beyond the
standard library.

Every profile built here is tagged ``source="specdb"``, and reports must say so: a spec
sheet cannot know about co-tenants, ECC settings, MIG partitioning, or a display attached
to the card.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from ..core.gpu import GPUProfile, Interconnect
from ..core.units import gib

_DATA_FILE = "gpus.json"


class UnknownGPUError(KeyError):
    """Raised when a GPU key is not in the database."""


@lru_cache(maxsize=1)
def _load() -> dict[str, Any]:
    raw = resources.files(__package__).joinpath(_DATA_FILE).read_text(encoding="utf-8")
    data = json.loads(raw)
    gpus = data.get("gpus")
    if not isinstance(gpus, dict) or not gpus:
        raise RuntimeError(f"{_DATA_FILE} contains no gpus")
    return gpus


def available() -> tuple[str, ...]:
    """Sorted GPU keys known to the database."""
    return tuple(sorted(_load()))


def _normalise(key: str) -> str:
    return key.strip().lower().replace("_", "-").replace(" ", "-")


def _resolve_key(key: str) -> str:
    gpus = _load()
    wanted = _normalise(key)
    if wanted in gpus:
        return wanted

    # Accept full product names, e.g. "NVIDIA RTX 4090".
    for candidate, spec in gpus.items():
        if _normalise(str(spec.get("name", ""))) == wanted:
            return candidate

    matches = [c for c in gpus if wanted in c]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise UnknownGPUError(f"{key!r} is ambiguous; matches {sorted(matches)}")
    raise UnknownGPUError(f"unknown GPU {key!r}; known keys: {', '.join(available())}")


def load(key: str, *, count: int = 1) -> GPUProfile:
    """Build a :class:`GPUProfile` from the database.

    Args:
        key: A database key (``"h100-sxm"``) or full product name (``"NVIDIA H100 SXM"``).
        count: Number of identical devices available.
    """
    spec = _load()[_resolve_key(key)]
    vram = gib(float(spec["vram_gib"]))
    usable = round(vram * float(spec["usable_fraction"]))
    capability = tuple(int(v) for v in spec["compute_capability"])
    if len(capability) != 2:
        raise RuntimeError(f"{key}: compute_capability must have two components")

    interconnect = Interconnect.NONE
    if count > 1:
        label = str(spec.get("interconnect", "none"))
        interconnect = next(
            (member for member in Interconnect if member.label == label), Interconnect.NONE
        )
        if interconnect is Interconnect.NONE:
            raise RuntimeError(f"{key}: unknown interconnect {label!r}")

    return GPUProfile(
        name=str(spec["name"]),
        vram_bytes=vram,
        vram_usable_bytes=usable,
        compute_capability=(capability[0], capability[1]),
        sm_count=int(spec["sm_count"]),
        mem_bandwidth_bytes_s=float(spec["mem_bandwidth_gb_s"]) * 1e9,
        dense_flops={str(k): float(v) for k, v in spec["dense_flops"].items()},
        count=count,
        interconnect=interconnect,
        source="specdb",
        notes=(
            "figures from a specification sheet; a live device may expose less memory due "
            "to ECC, MIG partitioning, an attached display, or co-tenants",
        ),
    )


__all__ = ["UnknownGPUError", "available", "load"]
