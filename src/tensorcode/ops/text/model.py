"""Provider-neutral model requests used by all message operations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .messages import Message


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[Message, ...]
    instructions: str | None = None
    response_schema: Mapping[str, Any] | None = None
    schema_name: str | None = None

    def __post_init__(self):
        messages = tuple(self.messages)
        if not messages or not all(isinstance(message, Message) for message in messages):
            raise TypeError("ModelRequest.messages must be a nonempty Message sequence")
        object.__setattr__(self, "messages", messages)
        if self.instructions is not None and not isinstance(self.instructions, str):
            raise TypeError("ModelRequest.instructions must be a string or None")


@dataclass(frozen=True)
class ModelOutput:
    text: str | None = None
    structured: Mapping[str, Any] | None = None
    provider_metadata: Mapping[str, Any] | None = None

    def __post_init__(self):
        if self.text is None and self.structured is None:
            raise ValueError("ModelOutput requires text or structured output")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("ModelOutput.text must be a string or None")
        if self.structured is not None and not isinstance(self.structured, Mapping):
            raise TypeError("ModelOutput.structured must be a mapping or None")


@runtime_checkable
class Model(Protocol):
    """Synchronous provider-neutral model: ``complete(ModelRequest) -> ModelOutput``."""
    def complete(self, request: ModelRequest) -> ModelOutput:
        ...


@runtime_checkable
class AsyncModel(Protocol):
    """Asynchronous model: ``await acomplete(ModelRequest) -> ModelOutput``."""
    async def acomplete(self, request: ModelRequest) -> ModelOutput:
        ...


@runtime_checkable
class BatchModel(Protocol):
    """Optional fused transport: one ``ModelOutput`` per request, in order."""
    def complete_batch(self, requests: Sequence[ModelRequest]) -> Sequence[ModelOutput]:
        ...


@runtime_checkable
class QuestionModel(Protocol):
    """Answer several named requests about identical messages in one exchange."""

    def complete_questions(self, requests: Mapping[str, ModelRequest]) -> Mapping[str, ModelOutput]:
        ...
