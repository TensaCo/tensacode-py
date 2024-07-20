import traceback
from typing import Any, Literal, ClassVar
from pydantic import BaseModel, Field
from datetime import datetime
from typing import TypedDict
from contextlib import contextmanager
from tensacode.internal.utils.registry import Registry, HasRegistry


class LogEntry(BaseModel):
    type: Literal["command", "notes", "feedback", "info"] = Field(default="info")
    weight: float = Field(default=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    frames: list[str] = Field(
        default_factory=lambda: [frame.name for frame in traceback.extract_stack()[:-2]]
    )
    content: Any


class BaseLog(BaseModel):

    items: list[LogEntry]

    def command(self, content: Any, **kwargs):
        self.items.append(LogEntry(type="command", content=content, **kwargs))

    def notes(self, content: Any, **kwargs):
        self.items.append(LogEntry(type="notes", content=content, **kwargs))

    def feedback(self, content: Any, **kwargs):
        self.items.append(LogEntry(type="feedback", content=content, **kwargs))

    def info(self, content: Any, **kwargs):
        self.items.append(LogEntry(type="info", content=content, **kwargs))

    def __getitem__(self, index: int) -> LogEntry:
        return self.items[index]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __str__(self):
        return "\n\n".join(str(message) for message in self.items)

    @contextmanager
    def subcontext(self):
        """
        Context manager for creating a subcontext.
        Usage:
            Creates a copy of the current context, allows modifications to the copy,
            and then discards the copy when exiting the context manager.

        Examples:
            >>> context = BaseContext()
            >>> context.command("Original command")
            >>> with context.subcontext() as subctx:
            ...     subctx.command("Subcontext command")
            ...     assert len(subctx) == 2
            ...     assert len(context) == 1
            >>> assert len(context) == 1
            >>> assert context[-1].content == "Original command"

            # Ensure that modifying the subcontext doesn't affect the original
            >>> context = BaseContext()
            >>> context.command("First command")
            >>> with context.subcontext() as subctx:
            ...     subctx.command("Second command")
            ...     subctx.items[0].content = "Modified first command"
            >>> assert context[0].content == "First command"
            >>> assert len(context) == 1
        """
        copy = self.copy()
        yield copy
        self.items.pop()
