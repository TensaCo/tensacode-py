"""Reserved symbolic operation contracts; execution is not implemented."""
from __future__ import annotations

from ._symbolic import SymbolicOperation
from .representation import Graph


class Decode(SymbolicOperation):
    """Realize symbolic structure in an output representation.

    Reserved API: calling this operation raises NotImplementedError.
    """

    def forward(self, value: Graph, *, context=None) -> object:
        self._unimplemented()


class TextDecode(SymbolicOperation):
    """Realize symbolic content as text without inventing unsupported relationships.

    Reserved API: calling this operation raises NotImplementedError.
    """

    def forward(self, value: Graph, *, context=None) -> str:
        self._unimplemented()
