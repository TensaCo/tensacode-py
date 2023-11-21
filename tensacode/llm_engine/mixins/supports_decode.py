from __future__ import annotations
from abc import ABC

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import _DataclassT, dataclass
import functools
from functools import singledispatchmethod
import inspect
from pathlib import Path
import pickle
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    TypeVar,
)
from box import Box
from uuid import uuid4
import attr
from jinja2 import Template
import loguru
from glom import glom
from pydantic import Field
import typingx
import pydantic, sqlalchemy, dataclasses, attr, typing


import tensacode as tc
from tensacode.utils.decorators import (
    Decorator,
    Default,
    dynamic_defaults,
    is_attrs_instance,
    is_attrs_type,
    is_dataclass_instance,
    is_dataclass_type,
    is_namedtuple_instance,
    is_namedtuple_type,
    is_object_instance,
    is_type,
    is_pydantic_model_instance,
    is_pydantic_model_type,
    is_sqlalchemy_instance,
    is_sqlalchemy_model_type,
    overloaded,
)
from tensacode.utils.oo import HasDefault, Namespace
from tensacode.utils.string0 import render_invocation, render_stacktrace
from tensacode.utils.types import (
    enc,
    T,
    R,
    atomic_types,
    container_types,
    composite_types,
    tree_types,
    tree,
    DataclassInstance,
    AttrsInstance,
)
from tensacode.utils.internal_types import nested_dict
from tensacode.base.base_engine import BaseEngine
from tensacode.llm_engine.base_llm_engine import BaseLLMEngine
import tensacode.base.mixins as mixins


class SupportsDecodeMixin(
    Generic[T, R], BaseLLMEngine[T, R], mixins.SupportsDecodeMixin[T, R], ABC
):
    @overloaded
    def _decode(
        self,
        object_enc: R,
        type: type[T],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(is_object_instance)
    def _decode_to_object(
        self,
        object_enc: R,
        type: type[T],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda object: callable(object))
    def _decode_to_function(
        self,
        object_enc: R,
        type: type[Callable],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, pydantic.BaseModel))
    def _decode_to_pydantic_model_instance(
        self,
        object_enc: R,
        type: type[pydantic.BaseModel],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: is_namedtuple_type(type))
    def _decode_to_namedtuple_instance(
        self,
        object_enc: R,
        type: type[NamedTuple],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: is_dataclass_type(type))
    def _decode_to_dataclass_instance(
        self,
        object_enc: R,
        type: type[DataclassInstance],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    #  TODO: pull these into a separate mixin `supports_create_type`
    # @_decode.overload(lambda type: is_type(type))
    # def _decode_to_type(
    #     self,
    #     object_enc: R,
    #     type: type,
    #     /,
    #     examples: list[T] = None,
    #     depth_limit: int | None = None,
    #     instructions: R | None = None,
    #     **kwargs,
    # ) -> T:
    #     raise NotImplementedError()

    # @_decode.overload(lambda type: is_pydantic_model_type(type))
    # def _decode_to_pydantic_model_type(
    #     self,
    #     object_enc: R,
    #     type: type[type[pydantic.BaseModel]],
    #     /,
    #     examples: list[T] = None,
    #     depth_limit: int | None = None,
    #     instructions: R | None = None,
    #     **kwargs,
    # ) -> T:
    #     raise NotImplementedError()

    # @_decode.overload(lambda type: is_namedtuple_type(type))
    # def _decode_to_namedtuple_type(
    #     self,
    #     object_enc: R,
    #     type: type[NamedTuple],
    #     /,
    #     examples: list[T] = None,
    #     depth_limit: int | None = None,
    #     instructions: R | None = None,
    #     **kwargs,
    # ) -> T:
    #     raise NotImplementedError()

    # @_decode.overload(lambda type: is_dataclass_type(type))
    # def _decode_to_dataclass_type(
    #     self,
    #     object_enc: R,
    #     type: type[DataclassInstance],
    #     /,
    #     examples: list[T] = None,
    #     depth_limit: int | None = None,
    #     instructions: R | None = None,
    #     **kwargs,
    # ) -> T:
    #     raise NotImplementedError()

    @_decode.overload(lambda object_enc: object_enc is None)
    def _decode_to_none(
        self,
        object_enc: R,
        type: type[None],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, bool))
    def _decode_to_bool(
        self,
        object_enc: R,
        type: type[bool],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, int))
    def _decode_to_int(
        self,
        object_enc: R,
        type: type[int],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, float))
    def _decode_to_float(
        self,
        object_enc: R,
        type: type[float],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, complex))
    def _decode_to_complex(
        self,
        object_enc: R,
        type: type[complex],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, str))
    def _decode_to_str(
        self,
        object_enc: R,
        type: type[str],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
