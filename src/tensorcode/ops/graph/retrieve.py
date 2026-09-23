"""Reserved symbolic operation contracts; execution is not implemented."""
from __future__ import annotations

from ._symbolic import SymbolicOperation
from .representation import Graph


class Retrieve(SymbolicOperation):
    """Retrieve symbolic evidence relevant to a graph query.

    Reserved API: calling this operation raises NotImplementedError.
    """

    def forward(self, value: Graph, *, context=None) -> tuple[Graph, ...]:
        """Reserved symbolic API; always raises NotImplementedError."""
        self._unimplemented()
