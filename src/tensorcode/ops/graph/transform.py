"""Reserved symbolic operation contracts; execution is not implemented."""
from __future__ import annotations

from ._symbolic import SymbolicOperation
from .representation import Graph


class Transform(SymbolicOperation):
    """Revise symbolic structure in light of explicitly supplied context.

    Reserved API: calling this operation raises NotImplementedError.
    """

    def forward(self, value: Graph, *, context=None) -> Graph:
        """Reserved symbolic API; always raises NotImplementedError."""
        self._unimplemented()
