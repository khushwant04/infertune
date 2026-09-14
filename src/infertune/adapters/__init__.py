"""Framework adapters.

Adapters translate a framework-independent :class:`~infertune.core.plan.ResourcePlan` into a
specific engine's invocation, resolving semantic roles against the **installed** engine's own
parameter schema rather than a table maintained here.
"""

from __future__ import annotations

from .base import (
    Capabilities,
    Diagnostic,
    FrameworkAdapter,
    LaunchSpec,
    ParamRole,
    ParamSchema,
    ParamSpec,
    Severity,
    StartupFacts,
)
from .registry import UnknownEngineError, available_engines, get_adapter
from .sglang import SGLangAdapter, SGLangSchemaError
from .vllm import VLLMAdapter, VLLMSchemaError, introspect_installed, load_schema

__all__ = [
    "Capabilities",
    "Diagnostic",
    "FrameworkAdapter",
    "LaunchSpec",
    "ParamRole",
    "ParamSchema",
    "ParamSpec",
    "SGLangAdapter",
    "SGLangSchemaError",
    "Severity",
    "StartupFacts",
    "UnknownEngineError",
    "VLLMAdapter",
    "VLLMSchemaError",
    "available_engines",
    "get_adapter",
    "introspect_installed",
    "load_schema",
]
