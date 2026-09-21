"""Reserved symbolic operation contracts; execution is not implemented."""
from __future__ import annotations

from ._symbolic import SymbolicOperation
from .representation import Graph


class Encode(SymbolicOperation):
    """Encode input evidence into source-grounded symbolic structure.

    Reserved API: calling this operation raises NotImplementedError.
    """

    def forward(self, value: object, *, context=None) -> Graph:
        self._unimplemented()


class TextEncode(SymbolicOperation):
    """Interpret text as symbolic relationships while preserving its evidence and uncertainty.

    Reserved API: calling this operation raises NotImplementedError.
    """

    def forward(self, value: str, *, context=None) -> Graph:
        self._unimplemented()
