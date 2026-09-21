"""Shared declaration for symbolic operations that have no implementation yet."""
from ..base import Operation


class SymbolicOperation(Operation):
    """A callable API declaration, never a callback or neural fallback."""

    def configuration(self) -> dict[str, str]:
        return {
            "operation": f"graph.{type(self).__name__}",
            "implementation": "unimplemented",
        }

    def _unimplemented(self):
        raise NotImplementedError(
            f"graph.{type(self).__name__} symbolic semantics are not implemented. "
            "Graph values can store explicitly supplied structure; "
            "they do not infer meaning from inputs."
        )
