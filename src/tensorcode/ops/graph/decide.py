"""Reserved symbolic operation contracts; execution is not implemented."""
from __future__ import annotations

from ._symbolic import SymbolicOperation
from .representation import Graph
from dataclasses import dataclass


@dataclass(frozen=True)
class ChoiceInput:
    """Explicit objective and alternatives; this record supplies no policy."""

    objective: Graph
    options: tuple[Graph, ...]

    def __post_init__(self):
        options = tuple(self.options)
        if not isinstance(self.objective, Graph):
            raise TypeError("Choice objective must be a Graph")
        if not options or not all(isinstance(option, Graph) for option in options):
            raise ValueError("Choice options must contain at least one Graph")
        object.__setattr__(self, "options", options)


class Decide(SymbolicOperation):
    """Select among explicit symbolic alternatives for an objective.

    Reserved API: calling this operation raises NotImplementedError.
    """

    def forward(self, value: ChoiceInput, *, context=None) -> Graph:
        """Reserved symbolic API; always raises NotImplementedError."""
        self._unimplemented()
