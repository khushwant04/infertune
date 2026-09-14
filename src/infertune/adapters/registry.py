"""Engine registry.

Exists so that callers ask for "an adapter for engine X" rather than importing a specific
class. Without it, every consumer — the CLI, the search loop, the validation script — would
name engines explicitly, and adding a third engine would mean editing all of them. With it,
a new engine is one entry here.

This is what makes the M4 acceptance criterion meaningful: *"SGLang support adds zero changes
outside ``adapters/``"*.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import FrameworkAdapter, ParamSchema
from .sglang import SGLangAdapter
from .vllm import VLLMAdapter

_FACTORIES: dict[str, Callable[[ParamSchema | None], FrameworkAdapter]] = {
    "vllm": lambda schema: VLLMAdapter(schema),
    "sglang": lambda schema: SGLangAdapter(schema),
}


class UnknownEngineError(KeyError):
    """Raised when no adapter is registered for an engine name."""


def available_engines() -> tuple[str, ...]:
    """Engine names with a registered adapter."""
    return tuple(sorted(_FACTORIES))


def get_adapter(engine: str, schema: ParamSchema | None = None) -> FrameworkAdapter:
    """Build an adapter by engine name.

    Args:
        engine: Registered engine name, case-insensitive.
        schema: Optional recorded schema. When omitted the adapter introspects the installed
            engine, which is the normal path on a serving host.
    """
    key = engine.strip().lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        raise UnknownEngineError(
            f"no adapter for {engine!r}; available: {', '.join(available_engines())}"
        )
    return factory(schema)


__all__ = ["UnknownEngineError", "available_engines", "get_adapter"]
