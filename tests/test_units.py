"""Unit-handling tests, focused on the decimal/binary trap."""

from __future__ import annotations

import pytest

from infertune.core.units import (
    GB,
    GIB,
    MIB,
    fmt_bytes,
    fmt_tokens,
    gb,
    gib,
    parse_size,
    to_gb,
    to_gib,
)


def test_a_24gb_gpu_is_2235_gib() -> None:
    """The gap that sinks naive memory arithmetic.

    A "24 GB" card exposes ~22.35 GiB. Losing this 1.65 GiB is enough to turn a
    working recommendation into a boot-time OOM.
    """
    vram = gb(24)
    assert vram == 24_000_000_000
    assert to_gib(vram) == pytest.approx(22.35, abs=0.01)
    assert to_gib(vram) < 24.0


def test_decimal_and_binary_do_not_coincide() -> None:
    assert gib(1) == 1_073_741_824
    assert gb(1) == 1_000_000_000
    assert gib(1) > gb(1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0),
        ("2048", 2048),
        ("1b", 1),
        ("1KiB", 1024),
        ("1 MiB", MIB),
        ("1GiB", GIB),
        ("24GB", 24 * GB),
        ("24 gb", 24 * GB),
        ("3.77GiB", round(3.77 * GIB)),
        ("  16MiB  ", 16 * MIB),
    ],
)
def test_parse_size(text: str, expected: int) -> None:
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "12 lightyears", "1.2.3GiB", "-5GiB", "GiB"])
def test_parse_size_rejects_bad_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_size(text)


def test_parse_size_error_names_valid_suffixes() -> None:
    with pytest.raises(ValueError, match="gib"):
        parse_size("5 quatloos")


def test_parse_size_roundtrips_through_gib() -> None:
    assert parse_size("3.77GiB") == gib(3.77)


def test_fmt_bytes_defaults_to_binary() -> None:
    assert fmt_bytes(GIB) == "1.00 GiB"
    assert fmt_bytes(gb(24)) == "22.35 GiB"
    assert fmt_bytes(gb(24), binary=False) == "24.00 GB"


def test_fmt_bytes_small_and_negative() -> None:
    assert fmt_bytes(512) == "512 B"
    assert fmt_bytes(-GIB) == "-1.00 GiB"


def test_to_gb_and_to_gib_are_display_only() -> None:
    assert to_gb(gb(24)) == pytest.approx(24.0)
    assert to_gib(gib(8)) == pytest.approx(8.0)


def test_fmt_tokens_uses_separators() -> None:
    assert fmt_tokens(30900) == "30,900"
