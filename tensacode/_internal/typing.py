from __future__ import annotations
from typing import Any, TypeVar, Union
from numbers import Number

from pydantic import BaseModel, Field


K, V = TypeVar("K"), TypeVar("V")
nested_dict = dict[K, "nested_dict[K, V]"] | dict[K, V]

Predicate = TypeVar("Predicate", bound=callable[..., bool])


def make_union(types):
    if len(types) == 1:
        return types[0]
    else:
        return Union[types[0], make_union(types[1:])]

from typing import Protocol, runtime_checkable


@runtime_checkable
class Predicate(Protocol):
    def __call__(self, *args, **kwargs) -> bool:
        pass


class Message(BaseModel):
    content: Any
    metadata: dict[str, Any] = Field(factory=lambda: {})

class Action(Message):
    children: list[Message] = Field(factory=lambda: [])