"""Reserved symbolic operation contracts; execution is not implemented."""
from __future__ import annotations

from ._symbolic import SymbolicOperation
from .representation import Graph


class Score(SymbolicOperation):
    """Assess symbolic structure under an explicitly configured objective.

    Reserved API: calling this operation raises NotImplementedError.
    """

    def forward(self, value: Graph, *, context=None) -> float:
        """Reserved symbolic API; always raises NotImplementedError."""
        self._unimplemented()
