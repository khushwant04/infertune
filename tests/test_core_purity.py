"""Enforce that `infertune.core` stays stdlib-only.

This is the M0 acceptance criterion, made mechanical. Stated as a design commitment it
would erode on the first convenient ``import torch``; as a test it cannot.

Checked by AST inspection rather than by importing, so the test is meaningful even in an
environment where the forbidden packages happen to be installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).resolve().parent.parent / "src" / "infertune" / "core"

ALLOWED_FIRST_PARTY = {"infertune"}


def _core_modules() -> list[Path]:
    modules = sorted(CORE_DIR.glob("*.py"))
    assert modules, f"no modules found under {CORE_DIR}"
    return modules


def _top_level_imports(source: str) -> set[str]:
    """Top-level package names imported by a module, ignoring relative imports."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


@pytest.mark.parametrize("module", _core_modules(), ids=lambda p: p.name)
def test_core_imports_only_stdlib(module: Path) -> None:
    imported = _top_level_imports(module.read_text())
    disallowed = imported - sys.stdlib_module_names - ALLOWED_FIRST_PARTY
    assert not disallowed, (
        f"{module.name} imports non-stdlib package(s) {sorted(disallowed)}. "
        "infertune.core must stay dependency-free so the analytical engine runs on "
        "CPU-only CI and without a GPU present; move this code to a layer above core."
    )


def test_core_never_imports_torch_or_heavy_deps() -> None:
    """Explicit check for the packages this constraint exists to keep out."""
    forbidden = {"torch", "pydantic", "vllm", "sglang", "pynvml", "transformers", "numpy"}
    offenders: dict[str, set[str]] = {}
    for module in _core_modules():
        hits = _top_level_imports(module.read_text()) & forbidden
        if hits:
            offenders[module.name] = hits
    assert not offenders, f"forbidden imports in infertune.core: {offenders}"


def test_core_is_importable_without_cli_dependencies() -> None:
    """Importing core must not transitively pull in typer/rich."""
    for name in list(sys.modules):
        if name.startswith(("infertune", "typer", "rich")):
            del sys.modules[name]

    import infertune.core  # noqa: F401

    leaked = [m for m in sys.modules if m.startswith(("typer", "rich"))]
    assert not leaked, f"importing infertune.core pulled in CLI dependencies: {leaked}"
