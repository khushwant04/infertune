"""Byte and token unit handling.

Confusing decimal (GB) and binary (GiB) units is the single most common source of
memory-arithmetic bugs in this domain, and it matters here: a "24 GB" GPU exposes
about 22.35 GiB, and a profiler that loses that 1.65 GiB will happily recommend a
configuration that OOMs on boot.

Rules enforced by convention throughout `infertune`:

* Every stored quantity is an ``int`` count of **bytes**. Never a float, never a
  pre-scaled "GB" value.
* Every field name carries its unit: ``*_bytes``, ``*_tokens``, ``*_bytes_s``.
* Decimal units are used only where a vendor spec sheet uses them (VRAM capacity,
  memory bandwidth). Everything computed is binary.
"""

from __future__ import annotations

import re
from typing import Final

# Decimal (SI) — what vendor spec sheets quote.
KB: Final = 1_000
MB: Final = 1_000_000
GB: Final = 1_000_000_000
TB: Final = 1_000_000_000_000

# Binary (IEC) — what allocators actually deal in.
KIB: Final = 1024
MIB: Final = 1024**2
GIB: Final = 1024**3
TIB: Final = 1024**4

_SUFFIXES: Final[dict[str, int]] = {
    "b": 1,
    "kb": KB,
    "mb": MB,
    "gb": GB,
    "tb": TB,
    "kib": KIB,
    "mib": MIB,
    "gib": GIB,
    "tib": TIB,
}

_SIZE_RE: Final = re.compile(
    r"^\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<suffix>[a-zA-Z]*)\s*$",
)


def gib(value: float) -> int:
    """Binary gigabytes -> bytes."""
    return round(value * GIB)


def mib(value: float) -> int:
    """Binary megabytes -> bytes."""
    return round(value * MIB)


def gb(value: float) -> int:
    """Decimal gigabytes -> bytes. Use for bandwidth and disk, *not* for VRAM."""
    return round(value * GB)


def vram_nameplate(advertised_gb: float) -> int:
    """Convert an advertised VRAM capacity to bytes, treating it as **binary**.

    GPU VRAM is quoted in binary units even though the marketing says "GB", because
    memory is manufactured in powers of two. This trips people up in the opposite
    direction from disk capacity:

    * A "24 GB" RTX 4090 reports 24564 MiB — about 23.99 GiB, not 22.35 GiB.
    * An "80 GB" H100 offers roughly 85.5 *decimal* GB, i.e. ~80 GiB.

    Reading VRAM as decimal understates capacity by 7.4%, which on a 24 GB card is
    1.65 GiB — enough KV cache for well over ten thousand tokens. The error is
    conservative, so it does not cause OOM; it silently leaves throughput on the table
    and pushes the recommender toward unnecessary mitigations like fp8 KV.

    Prefer a measured figure from NVML when a device is present. This helper exists for
    the spec-database path, where only the advertised number is known.
    """
    if advertised_gb <= 0:
        raise ValueError(f"advertised_gb must be > 0, got {advertised_gb}")
    return gib(advertised_gb)


def to_gib(n_bytes: int) -> float:
    """Bytes -> binary gigabytes, for display only."""
    return n_bytes / GIB


def to_gb(n_bytes: int) -> float:
    """Bytes -> decimal gigabytes, for display only."""
    return n_bytes / GB


def parse_size(text: str) -> int:
    """Parse a human-written size into an exact byte count.

    Accepts both unit families, and treats a bare number as bytes::

        >>> parse_size("3.77GiB")
        4047689318
        >>> parse_size("24 GB")
        24000000000
        >>> parse_size("2048")
        2048

    Raises:
        ValueError: on unparseable input, an unknown suffix, or a negative size.
    """
    match = _SIZE_RE.match(text)
    if match is None:
        raise ValueError(f"cannot parse size: {text!r}")

    raw_value = float(match.group("value"))
    suffix = match.group("suffix").lower() or "b"

    if suffix not in _SUFFIXES:
        known = ", ".join(sorted(_SUFFIXES))
        raise ValueError(f"unknown size suffix {suffix!r} in {text!r}; expected one of: {known}")
    if raw_value < 0:
        raise ValueError(f"size must be non-negative, got {text!r}")

    return round(raw_value * _SUFFIXES[suffix])


def fmt_bytes(n_bytes: int, *, binary: bool = True, precision: int = 2) -> str:
    """Format a byte count for display.

    Uses binary units by default, because that is what allocation failures speak.
    """
    if n_bytes < 0:
        return "-" + fmt_bytes(-n_bytes, binary=binary, precision=precision)

    if binary:
        units = (("TiB", TIB), ("GiB", GIB), ("MiB", MIB), ("KiB", KIB))
    else:
        units = (("TB", TB), ("GB", GB), ("MB", MB), ("KB", KB))

    for name, scale in units:
        if n_bytes >= scale:
            return f"{n_bytes / scale:.{precision}f} {name}"
    return f"{n_bytes} B"


def fmt_tokens(n_tokens: int) -> str:
    """Format a token count with thousands separators."""
    return f"{n_tokens:,}"
