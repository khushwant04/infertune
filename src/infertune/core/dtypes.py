"""Numeric data types, sized in bits.

Bits rather than bytes, because sub-byte quantization is real: int4 is 0.5 bytes per
element, and carrying that as a float invites rounding drift through the memory ledger.
All arithmetic stays integral by working in bits and dividing once, at the end.
"""

from __future__ import annotations

from enum import Enum


class DType(Enum):
    """A numeric type, with its storage width in bits."""

    FP32 = ("fp32", 32)
    FP16 = ("fp16", 16)
    BF16 = ("bf16", 16)
    FP8_E4M3 = ("fp8_e4m3", 8)
    FP8_E5M2 = ("fp8_e5m2", 8)
    INT8 = ("int8", 8)
    FP4 = ("fp4", 4)
    NVFP4 = ("nvfp4", 4)
    INT4 = ("int4", 4)

    def __init__(self, label: str, bits: int) -> None:
        self.label = label
        self.bits = bits

    @property
    def is_float8(self) -> bool:
        return self in (DType.FP8_E4M3, DType.FP8_E5M2)

    @property
    def is_sub_byte(self) -> bool:
        return self.bits < 8

    @classmethod
    def parse(cls, text: str) -> DType:
        """Look up a dtype by label, case-insensitively."""
        wanted = text.strip().lower()
        for member in cls:
            if member.label == wanted:
                return member
        known = ", ".join(m.label for m in cls)
        raise ValueError(f"unknown dtype {text!r}; expected one of: {known}")

    def __str__(self) -> str:
        return self.label


def bytes_for(count: int, dtype: DType) -> int:
    """Storage for ``count`` elements of ``dtype``, in whole bytes.

    Rounds up, since a partial byte still occupies a byte.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    total_bits = count * dtype.bits
    return -(-total_bits // 8)  # ceiling division
