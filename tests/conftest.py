"""Test configuration.

Network-dependent tests are opt-in so the default suite and CI stay hermetic and fast.
"""

from __future__ import annotations

import os

import pytest

# Pin the rendering width before rich is imported. Rich sizes output to the terminal, so
# assertions on rendered CLI output otherwise pass on a wide developer terminal and fail in
# CI's 80-column environment. Prefer introspecting the app over asserting on rendered text,
# but where output is checked, make the width deterministic.
os.environ.setdefault("COLUMNS", "200")
os.environ.setdefault("TERM", "dumb")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that require network access (Hugging Face Hub)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
