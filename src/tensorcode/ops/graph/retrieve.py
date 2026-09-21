"""Retrieval over an explicit graph collection using supplied relevance."""

from __future__ import annotations

from dataclasses import dataclass

from ..base import Operation
from .decode import JSONDecoder
from .representation import Graph
from .score import require_score, require_semantics_identity


@dataclass(frozen=True)
class ScoredGraph:
    value: Graph
    score: float


class Retrieve(Operation):
    """Rank only the configured graph values; never constructs a candidate."""

    def __init__(self, items, relevance, *, semantics: str, limit: int | None = None):
        self.items = tuple(items)
        if not all(isinstance(item, Graph) for item in self.items):
            raise TypeError("Retrieve items must be Graph values")
        if not callable(relevance):
            raise TypeError("Retrieve requires a relevance callable")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer or None")
        self.relevance = relevance
        self.semantics = require_semantics_identity(semantics)
        self.limit = limit

    def forward(self, value, *, context=None) -> tuple[ScoredGraph, ...]:
        if not isinstance(value, Graph):
            raise TypeError("Retrieve expects a Graph query")
        context = context or {}
        scored = tuple(
            ScoredGraph(item, require_score(self.relevance(value, item, context)))
            for item in self.items
        )
        ranked = tuple(sorted(scored, key=lambda item: item.score, reverse=True))
        return ranked if self.limit is None else ranked[: self.limit]

    def configuration(self) -> dict[str, object]:
        decode = JSONDecoder()
        return {
            "operation": "graph.retrieve",
            "semantics": self.semantics,
            "limit": self.limit,
            "items": [decode.forward(item) for item in self.items],
        }
