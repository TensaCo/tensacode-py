from __future__ import annotations
from typing import TypeVar, Union


K, V = TypeVar("K"), TypeVar("V")
nested_dict = dict[K, "nested_dict[K, V]"] | dict[K, V]

Predicate = TypeVar("Predicate", bound=callable[..., bool])


def make_union(types):
    if len(types) == 1:
        return types[0]
    else:
        return Union[types[0], make_union(types[1:])]
