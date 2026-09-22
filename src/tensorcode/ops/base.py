"""The common callable boundary; no global model or implicit fallback."""
from abc import ABC, abstractmethod
import asyncio
from collections.abc import Mapping
from typing import Any
from tensorcode._internal.tracing import invoke


class Operation(ABC):
    # Opt in only when replay cannot perform external effects.
    replayable = False

    def __call__(self, value: Any, *, context: Mapping | None = None) -> Any:
        return invoke(self, value, context, self.forward)

    async def acall(self, value: Any, *, context: Mapping | None = None) -> Any:
        """Explicit asynchronous invocation with the same tracing boundary."""
        from tensorcode._internal.tracing import invoke_async
        return await invoke_async(self, value, context, self.aforward)

    async def aforward(self, value: Any, *, context: Mapping | None = None) -> Any:
        """Thread fallback; override for a native asynchronous backend."""
        return await asyncio.to_thread(self.forward, value, context=context)

    @abstractmethod
    def forward(self, value: Any, *, context: Mapping | None = None) -> Any:
        """Implement a transformation; callers use the instance, not forward."""
        raise NotImplementedError
