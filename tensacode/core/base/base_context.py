from typing import Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class ContextItem(BaseModel):
    type: Literal["command", "response", "error", "info"] = Field(default="info")
    weight: float = Field(default=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseContext(BaseModel, HasRegistry):

    items: list[ContextItem]
    _registry: ClassVar[Registry["BaseContext"]] = Registry()

    @property
    def consolidated_items(self):
        consolidated = {}
        for item in self.items:
            consolidated.update(item.dict())
        return consolidated

    def __log(
        self,
        message_type: str,
        content: Any,
        context: Any = {},
        weight: float = 1.0,
        **kwargs,
    ):
        if not context:
            import traceback

            stack = traceback.extract_stack()
            context = [
                frame.name for frame in stack[:-2]
            ]  # Exclude the current function

        self.items.append(
            ContextItem(
                type=message_type,
                content=content,
                context=context,
                weight=weight,
                **kwargs,
            )
        )

    def command(self, content: Any, **kwargs):
        self.__log("command", content, **kwargs)

    def response(self, content: Any, **kwargs):
        self.__log("response", content, **kwargs)

    def error(self, content: Any, **kwargs):
        self.__log("error", content, **kwargs)

    def info(self, content: Any, **kwargs):
        self.__log("info", content, **kwargs)

    def __getitem__(self, index: int) -> ContextItem:
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
