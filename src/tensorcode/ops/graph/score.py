"""Graph scoring with caller-supplied, explicitly identified semantics."""

from __future__ import annotations

import math
from numbers import Real

from ..base import Operation
from .representation import Graph


def require_semantics_identity(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("An explicit nonempty semantics identity is required")
    return value


def require_score(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Supplied graph semantics must return a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Supplied graph semantics must return a finite score")
    return result


class Score(Operation):
    """Apply a supplied scoring function without assigning it a built-in meaning."""

    def __init__(self, function, *, semantics: str):
        if not callable(function):
            raise TypeError("Score requires a callable")
        self.function = function
        self.semantics = require_semantics_identity(semantics)

    def forward(self, value, *, context=None) -> float:
        if not isinstance(value, Graph):
            raise TypeError("Score expects a Graph")
        return require_score(self.function(value, context or {}))

    def configuration(self) -> dict[str, str]:
        return {"operation": "graph.score", "semantics": self.semantics}
