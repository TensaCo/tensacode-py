from __future__ import annotations
from typing import Any, Generator, Literal, Self

import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    Union,
    Optional,
    TypeVar,
    Generic,
    Type,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    Set,
    MutableSet,
    ByteString,
    MutableSequence,
)
from types import (
    NoneType,
    ModuleType,
    FunctionType,
    MethodType,
    GeneratorType,
    AsyncGeneratorType,
)
from types import FunctionType, MethodType, ModuleType
from abc import ABC, abstractmethod
from collections import namedtuple
import dataclasses
import typing
import attrs

import typingx
import attr
import pydantic

from attr import AttrsInstance
from _typeshed import DataclassInstance

K = str | int | None
T = TypeVar("T")
pyobject = object


ARG_IDENTIFIER = (
    str | int
)  # in ts, there is only ints, there may exist languages with only strings


class enc(Generic[T], ABC):
    """
    Nonfunctional annotation.
    Indicates that a type is the encoded form of its generic parameter.
    Prefer annotating with the encoded type itself where possible
    since we can't enforce this constraint.
    """


R = TypeVar("R", bound=enc)


atomic_types = (
    bool
    | int
    | float
    | complex
    | str
    | bytes
    | bytearray
    | NoneType
    | None
    | Generic
    | TypeVar
)
container_types = (
    set | frozenset | Mapping | Sequence | Iterator | GeneratorType | AsyncGeneratorType
)
composite_types = (
    (namedtuple | tuple)  # see utils.is_namedtuple
    | DataclassInstance
    | AttrsInstance
    | pydantic.BaseModel
    | ModuleType
    | container_types
    | type
    | object
)
function_types = Callable | FunctionType | MethodType | classmethod | staticmethod

tree_types = atomic_types | container_types | composite_types | "tree"


class tree(Generic[T]):
    def __instancecheck__(self, __instance: Any) -> bool:
        return super().__instancecheck__(__instance) or typingx.isinstancex(
            __instance, tree_types
        )

    def __subclasscheck__(self, __subclass: type) -> bool:
        return super().__subclasscheck__(__subclass) or typingx.issubclassx(
            __subclass, tree_types
        )


def is_pydantic_model_instance(object):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name
    return isinstance(object, pydantic.BaseModel)


def is_namedtuple_instance(object):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name
    return typingx.isinstancex(object, NamedTuple) or (
        isinstance(object, tuple) and hasattr(object, "_fields")
    )


def is_dataclass_instance(object):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name

    # is_dataclass returns true for dataclass types, so we need to exclude those
    return typingx.isinstancex(object, DataclassInstance) or (
        dataclasses.is_dataclass(object) and not isinstance(object, type)
    )


def is_object(obj):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name
    return isinstance(obj, object) or typingx.isinstancex(obj, Any)


def is_pydantic_model_type(pydantic_model_t, /):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name
    return issubclass(pydantic_model_t, pydantic.BaseModel)


def is_namedtuple_type(namedtuple_t, /):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name
    return typingx.issubclassx(namedtuple_t, NamedTuple) or (
        issubclass(namedtuple_t, tuple) and hasattr(namedtuple_t, "_fields")
    )


def is_dataclass_type(dataclass_t, /):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name

    # is_dataclass returns true for dataclass instances, so we need to exclude those
    return typingx.issubclassx(dataclass_t, DataclassInstance) or (
        dataclasses.is_dataclass(dataclass_t) and isinstance(dataclass_t, type)
    )


def is_type(t, /):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name
    return isinstance(t, (type, typing.TypeVar))


def is_lambda(lambda_, /):
    # NOTE: do NOT change signature w/o updating codebase.
    # must match Engine.<operator>.__params__[1].name
    return (
        isinstance(lambda_, type(lambda: None)) and lambda_.__qualname__ == "<lambda>"
    )


def is_callable(callable_t, /):
    # we want to explicitly exclude type constructors in this case
    # because they are treated differently than regular functions
    # if you want to recognize a constructor, just pass the __new__ method
    return issubclass(callable_t, (Callable, Protocol))


def get_keys(
    object,
    visibility: Literal["public", "protected", "private"] = "public",
    inherited_members=True,
) -> Generator[str, None, None]:
    keys = dir(object)

    if isinstance(object, type) and inherited_members is False:
        for base in object.__bases__:
            keys = [k for k in keys if k not in dir(base)]

    for k in keys:
        match visibility:
            case "public":
                if k.startswith("_"):
                    continue
            case "protected":
                if k.startswith("__"):
                    continue
            case "private":
                pass
            case _:
                raise ValueError(f"Invalid visibility: {visibility}")
        yield k


def get_keys_and_values(
    object,
    visibility: Literal["public", "protected", "private"] = "public",
    inherited_members=True,
) -> Generator[tuple[str, Any], None, None]:
    for k in get_keys(object, visibility, inherited_members):
        yield k, getattr(object, k)
