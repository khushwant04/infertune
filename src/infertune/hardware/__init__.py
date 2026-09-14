"""Hardware discovery.

Two interchangeable sources of a :class:`~infertune.core.gpu.GPUProfile`:

* :mod:`infertune.hardware.specdb` — a JSON specification database, needing no hardware.
* :mod:`infertune.hardware.nvml` — live inspection, requiring the optional ``nvml`` extra.
"""

from __future__ import annotations

from . import nvml, specdb
from .specdb import UnknownGPUError

__all__ = ["UnknownGPUError", "nvml", "specdb"]
