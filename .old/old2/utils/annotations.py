from __future__ import annotations

import inspect
from functools import wraps
from typing import TypeVar, Generic, Protocol

import ivy

from tensacode._utils.registry import Registry, HighestScore


type NumericalTensor = float|int|ivy.Array # TODO: syntax not correct
type like[T <= object] = object
type enc[T <= object] = NumericalTensor # Indicates that this type is an encoded form of T


class SupportsEncode[T](Protocol):
    def encode(self, object: T, *args, **kwargs) -> enc[T]: ...

def encode_args(fn, *,
        ignore_arg_indeces: list[int],
        ignore_kwarg_names: list[str]):
    """ Encodes args to wrapped function which have params
    annotated with `enc[T]` but are passed as `T`'s.
    """
    @wraps(fn)
    def _fn(self: SupportsEncode, *_args, **_kwargs):
        args = []
        kwargs = {}
        # TODO



        fn(self, *args, **kwargs)
    return _fn
