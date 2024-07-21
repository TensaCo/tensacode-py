from abc import abstractmethod
from typing import Any, Literal, TypedDict, ClassVar
from functools import cached_property, contextmanager
from pydantic import BaseModel, Field
from datetime import datetime
from functools import reduce
import traceback
import inspect
from functools import wraps
from tensacode.internal.utils.consts import VERSION
from tensacode.internal.utils.language import Language
from tensacode.internal.protocols.tagged_object import HasID
from tensacode.core.base.ops.ops import BaseOp
from tensacode.internal.utils.functional import cached_with_key
from contextlib import contextmanager
from tensacode.internal.utils.python_str import render_function_call


def generate_callstack(skip_frames=1):
    return [
        {
            "fn_name": frame.function,
            "params": {
                **dict(
                    zip(
                        inspect.getargvalues(frame[0]).args,
                        inspect.getargvalues(frame[0]).locals.values(),
                    )
                ),
                **{
                    k: v
                    for k, v in inspect.getargvalues(frame[0]).locals.items()
                    if k not in inspect.getargvalues(frame[0]).args
                },
            },
        }
        for frame in inspect.stack()[skip_frames:]
    ]


class Event(BaseModel, ABC):
    type: ClassVar[str]
    timestamp: datetime = Field(default_factory=datetime.now)
    callstack: list[dict] = Field(default_factory=generate_callstack)


class Command(Event):
    type: ClassVar[str] = "command"
    content: str
    importance: float


class Message(Event):
    type: ClassVar[str] = "message"
    content: str


class Feedback(Event):
    type: ClassVar[str] = "feedback"
    content: str
    reward: float


class Info(Event):
    type: ClassVar[str] = "info"
    content: str


class EngineContext(BaseModel):
    events: list[Event] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


class BaseEngine(HasID, BaseModel):

    tensacode_version: ClassVar[str] = VERSION
    render_language: Language = "python"
    ops_registry: dict[str, BaseOp] = Field(default_factory=dict)

    ### Context Management ###

    context_stack: list[EngineContext] = Field(default_factory=list)

    @cached_with_key(
        # recursively hash the context stack up to the index that way even if
        # upper contexts change, the cache remains the same for lower contexts
        lambda self, index: reduce(
            lambda acc, context: acc + hash(context.data),
            self.context_stack[:index],
            0,
        )
    )
    def cumulative_context(self, index=-1) -> EngineContext:
        """
        Returns the cumulative context up to the specified index in the context stack.

        This method combines all the contexts in the stack up to the given index,
        merging their data and concatenating their events.

        Args:
            index (int): The index up to which to accumulate context. Defaults to -1 (last context).

        Returns:
            EngineContext: A new EngineContext object containing the merged data and events.

        Raises:
            IndexError: If the provided index is out of range.

        Examples:
            >>> from tensacode.core.base.base_engine import BaseEngine, EngineContext, Event
            >>> engine = BaseEngine()
            >>> engine.context_stack = [
            ...     EngineContext(data={"a": 1}, events=[Event(type="message", content="First")]),
            ...     EngineContext(data={"b": 2}, events=[Event(type="message", content="Second")]),
            ...     EngineContext(data={"c": 3}, events=[Event(type="message", content="Third")])
            ... ]
            >>> cumulative = engine.cumulative_context(1)
            >>> cumulative.data
            {'a': 1, 'b': 2}
            >>> [event.content for event in cumulative.events]
            ['First', 'Second']
            >>> cumulative = engine.cumulative_context()  # Default to last context
            >>> cumulative.data
            {'a': 1, 'b': 2, 'c': 3}
            >>> [event.content for event in cumulative.events]
            ['First', 'Second', 'Third']
            >>> engine.cumulative_context(0)
            EngineContext(events=[], data={})
            >>> engine.cumulative_context(3)
            Traceback (most recent call last):
                ...
            IndexError: Index 3 out of range
        """
        # Convert negative index to positive index
        if index < 0:
            index = len(self.context_stack) + index

        # Check if index is out of range
        if index > len(self.context_stack) - 1:
            raise IndexError(f"Index {index} out of range")

        cumulative_data = {}
        cumulative_events = []
        current_index = 0

        while current_index <= index:
            context = self.context_stack[current_index]
            cumulative_data.update(context.data)
            cumulative_events.extend(context.events)
            current_index += 1

        return EngineContext(data=cumulative_data, events=cumulative_events)

    @contextmanager
    def subcontext(self, **overrides):
        new_context = self.context_start(**overrides)
        try:
            yield self
        finally:
            old_context = self.context_stop()
            assert (
                new_context == old_context
            ), "Context mismatch. Did you modify `engine.context_stack` while the `engine.subcontext` context manager was active?"

    def context_start(self, **overrides):
        new_context = EngineContext(logs=[], **overrides)
        self.context_stack.append(new_context)
        return new_context

    def context_stop(self):
        old_context = self.context_stack.pop()
        return old_context

    #### Logging ####

    def command(self, content: Any, **kwargs):
        self.context_stack[-1].events.append(Command(content=content, **kwargs))

    def notes(self, content: Any, **kwargs):
        self.context_stack[-1].events.append(Message(content=content, **kwargs))

    def feedback(self, content: Any, **kwargs):
        self.context_stack[-1].events.append(Feedback(content=content, **kwargs))

    def info(self, content: Any, **kwargs):
        self.context_stack[-1].events.append(Info(content=content, **kwargs))

    ### Execution ###

    def execute(self, op: BaseOp, *args, **kwargs: Any) -> Any:
        return op.execute(self, *args, **kwargs)

    def reward(self, reward: float): ...

    def train(self): ...

    def eval(self): ...

    ### Decorators ###

    def trace(self):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                with self.subcontext():
                    args_dict = {n: v for n, v in zip(fn.__annotations__.keys(), args)}
                    fn_call = render_function_call(fn, args=args, kwargs=kwargs)
                    self.info(fn_call)
                    result = fn(*args, **kwargs)
                    self.info(f"{fn.__name__} => {result}")
                    return result

            return wrapper

        return decorator
