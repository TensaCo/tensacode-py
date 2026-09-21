"""Graph choice using caller-supplied, explicitly identified utility semantics."""

from __future__ import annotations

from dataclasses import dataclass

from ..base import Operation
from .representation import Graph
from .retrieve import ScoredGraph
from .score import require_score, require_semantics_identity


@dataclass(frozen=True)
class ChoiceInput:
    objective: Graph
    options: tuple[Graph, ...]

    def __post_init__(self) -> None:
        options = tuple(self.options)
        if not isinstance(self.objective, Graph):
            raise TypeError("Choice objective must be a Graph")
        if not options or not all(isinstance(option, Graph) for option in options):
            raise ValueError("Choice options must contain at least one Graph")
        object.__setattr__(self, "options", options)


@dataclass(frozen=True)
class Decision:
    value: Graph
    scores: tuple[ScoredGraph, ...]
    semantics: str


class Decide(Operation):
    """Score every supplied option and select the first maximum."""

    def __init__(self, utility, *, semantics: str):
        if not callable(utility):
            raise TypeError("Decide requires a utility callable")
        self.utility = utility
        self.semantics = require_semantics_identity(semantics)

    def forward(self, value, *, context=None) -> Decision:
        if not isinstance(value, ChoiceInput):
            raise TypeError("Decide expects ChoiceInput")
        context = context or {}
        scores = tuple(
            ScoredGraph(
                option,
                require_score(self.utility(value.objective, option, context)),
            )
            for option in value.options
        )
        selected = max(scores, key=lambda item: item.score)
        return Decision(selected.value, scores, self.semantics)

    def configuration(self) -> dict[str, str]:
        return {"operation": "graph.decide", "semantics": self.semantics}
