"""Shared declaration for symbolic operations that have no implementation yet."""
from ..base import Operation
from ..._internal.operation_config import validated_config


class SymbolicOperation(Operation):
    """A callable API declaration, never a callback or neural fallback."""

    def __init__(self, config=None):
        self.config = validated_config(config, ())

    @classmethod
    def from_foundation(cls, *args, **kwargs):
        cls()._unimplemented()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls()._unimplemented()

    def save_pretrained(self, *args, **kwargs):
        self._unimplemented()

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
