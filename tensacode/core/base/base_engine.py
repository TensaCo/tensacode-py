from abc import abstractmethod
from typing import Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import traceback

from tensacode.internal.utils.language import Language
from tensacode.internal.protocols.tagged_object import HasID
from tensacode.core.base.ops.ops import BaseOp


class BaseEngine(HasID, HasRegistry, BaseModel):

    class Context(BaseModel):
        mode: Literal["command", "notes", "feedback", "info"] = Field(default="info")
        callstack: list[str] = Field(
            default_factory=lambda: [
                frame.name for frame in traceback.extract_stack()[:-2]
            ]
        )
        data: Any

    class LogStatement(BaseModel):
        mode: TODO
        timestamp: datetime = Field(default_factory=datetime.now)

    tensacode_version: str
    render_language: Language = "python"
    config: dict[str, Any]

    ops: dict[str, BaseOp]

    def execute(self, op: BaseOp, input: Any, **kwargs: Any) -> Any:
        return op.execute(self, input, **kwargs)

    stack: list[Context]

    @cached_property
    def current_context(self) -> Context:
        d = {}
        for context in self.stack:
            d.update(context.data)
        return d

    def command(self, content: Any, **kwargs):
        self.stack.append(self.LogStatement(mode="command", data=content, **kwargs))

    def notes(self, content: Any, **kwargs):
        self.stack.append(self.LogStatement(mode="notes", data=content, **kwargs))

    def feedback(self, content: Any, **kwargs):
        self.stack.append(self.LogStatement(mode="feedback", data=content, **kwargs))

    def info(self, content: Any, **kwargs):
        self.stack.append(self.LogStatement(mode="info", data=content, **kwargs))
