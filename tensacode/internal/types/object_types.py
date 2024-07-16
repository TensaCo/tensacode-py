from typing import Literal


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
