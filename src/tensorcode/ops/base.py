"""The common callable boundary; no global model or implicit fallback."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
from ..tracing import invoke


class Operation(ABC):
    # Opt in only when replay cannot perform external effects.
    replayable = False

    def __call__(self, value: Any, *, context: Mapping | None = None) -> Any:
        return invoke(self, value, context, self.forward)

    @abstractmethod
    def forward(self, value: Any, *, context: Mapping | None = None) -> Any:
        """Implement a transformation; callers use the instance, not forward."""
        raise NotImplementedError
