"""Configuration search.

Structured around the fact that engine boots, not candidate count, are the expensive
resource: enumerate freely, prune analytically, and spend a small boot budget on the
survivors. See ``docs/plan.md`` §2.4 and §8.
"""

from __future__ import annotations

from .engine import BootAndMeasure, Evaluation, SearchResult, grid_search, search
from .space import (
    Candidate,
    PruneReport,
    ScoredCandidate,
    enumerate_candidates,
    pareto_front,
    prune,
    score,
    score_all,
)

__all__ = [
    "BootAndMeasure",
    "Candidate",
    "Evaluation",
    "PruneReport",
    "ScoredCandidate",
    "SearchResult",
    "enumerate_candidates",
    "grid_search",
    "pareto_front",
    "prune",
    "score",
    "score_all",
    "search",
]
