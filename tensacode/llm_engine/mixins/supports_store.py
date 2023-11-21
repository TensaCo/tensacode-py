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


class SupportsStoreMixin(
    Generic[T, R], BaseLLMEngine[T, R], mixins.SupportsStoreMixin[T, R], ABC
):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def store(
        self,
        object: composite_types[T],
        /,
        values: list[T] = None,
        allowed_glob: str = None,
        disallowed_glob: str = None,
        depth_limit: int = DefaultParam(qualname="hparams.store.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.store.instructions"),
        **kwargs,
    ):
        """
        Stores the `object` with the given `values`.

        This method uses a specific storage algorithm (which can be customized) to store the input object along with its values. The storage process can be controlled by the `allowed_glob`, `disallowed_glob`, and `depth_limit` parameters.

        You can customize the storage algorithm by either subclassing `Engine` or adding a `__tc_store__` classmethod to `object`'s type class. The `__tc_store__` method should take in the same arguments as `Engine.store` and perform the storage operation.

        Args:
            object (T): The object to be stored. This could be any data structure like a list, dictionary, custom class, etc.
            values (list[T]): The values to be stored along with the object.
            allowed_glob (str): A glob pattern that specifies which parts of the object are allowed to be stored. Default is None, which means all parts are allowed.
            disallowed_glob (str): A glob pattern that specifies which parts of the object are not allowed to be stored. Default is None, which means no parts are disallowed.
            depth_limit (int): The maximum depth to which the storage process should recurse. This is useful for controlling the complexity of the storage, especially for deeply nested structures. Default is set in the engine's parameters.
            instructions (enc[str]): Additional instructions to the storage algorithm. This could be used to customize the storage process, for example by specifying certain features to focus on or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific storage algorithms.

        Returns:
            None

        Example:
            >>> engine = Engine()
            >>> class Person:
            ...    name: str
            ...    bio: str
            ...    thoughts: list[str]
            ...    friends: list[Person]
            >>> john, teyoni, huimin = ... # create people
            >>> person = engine.store(john, [huimin], instructions="she is his friend")
        """
        try:
            return type(object).__tc_store__(
                self,
                object,
                values=values,
                allowed_glob=allowed_glob,
                disallowed_glob=disallowed_glob,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._store(
            object,
            values=values,
            allowed_glob=allowed_glob,
            disallowed_glob=disallowed_glob,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @abstractmethod
    def _store(
        self,
        object: composite_types[T],
        /,
        values: list[T],
        allowed_glob: str,
        disallowed_glob: str,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ):
        raise NotImplementedError()
