from __future__ import annotations
import datetime
from typing import Any, Optional, TypeVar, Union
from numbers import Number

from pydantic import BaseModel, Field

from tensacode._internal.code2str import render_invocation


K, V = TypeVar("K"), TypeVar("V")
nested_dict = dict[K, "nested_dict[K, V]"] | dict[K, V]


def make_union(types):
    if len(types) == 1:
        return types[0]
    else:
        return Union[types[0], make_union(types[1:])]


class Statement(BaseModel):
    content: Any
    metadata: dict[str, Any] = Field(factory=lambda: {})


class Event(Statement):
    timestamp: datetime = Field(default_factory=datetime.now)


class Message(Event):
    pass


class Action(Event):
    events: list[Event] = Field(factory=lambda: [])

    @property
    def child_actions(self) -> list[Action]:
        return [child for child in self.events if isinstance(child, Action)]

    description: Optional[str] = None

    def __repr__(self) -> str:
        if self.description:
            return self.description
        else:
            return super().__repr__()


class Invocation(BaseModel):
    fn: callable
    args: list[Any]
    kwargs: dict[str, Any]
    result: Any

    def __repr__(self) -> str:
        return render_invocation(
            self.fn, args=self.args, kwargs=self.kwargs, result=self.result
        )


class Example(Statement):
    inputs: Any
    action: Optional[Action]
    output: Any
