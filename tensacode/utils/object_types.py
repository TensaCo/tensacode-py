from abc import abstractmethod
import types
from typing import ClassVar, Dict, Iterator, List, Literal, Set, Callable, Type, Any
import typing
import pydantic

OBJECT_TYPES = Literal[
    "primitive",
    "complex",
    "list",
    "tuple",
    "dict",
    "set",
    "frozenset",
    "bytes",
    "none",
    "callable",
    "module",
    "class",
    "instance",
    "iterator",
    "coroutine",
    "slice",
    "range",
    "ellipsis",
    "type",
    "tensor",
    "graph",
    "tree",
    "stream",
]
