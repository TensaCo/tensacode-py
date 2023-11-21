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


class SupportsRetrieveMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def retrieve(
        self,
        object: composite_types[T],
        /,
        count: int = DefaultParam(qualname="hparams.retrieve.count"),
        allowed_glob: str = None,
        disallowed_glob: str = None,
        depth_limit: int = DefaultParam(qualname="hparams.retrieve.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.retrieve.instructions"),
        **kwargs,
    ) -> T:
        """
        Retrieves an object from the engine.

        This method is used to retrieve an object from the engine based on the provided parameters. The object is retrieved in the form specified by the 'object' parameter.

        Args:
            object (composite_types[T]): The type of object to be retrieved.
            count (int): The number of objects to retrieve. Default is set in the engine's hyperparameters.
            allowed_glob (str): A glob pattern that the retrieved object's qualname (relative to `object`) must match. If None, no name filtering is applied.
            disallowed_glob (str): A glob pattern that the retrieved object's qualname (relative to `object`) must not match. If None, no name filtering is applied.
            depth_limit (int): The maximum depth to which the retrieval process should recurse. This is useful for controlling the complexity of the retrieval, especially for deeply nested structures. Default is set in the engine's hyperparameters.
            instructions (enc[str]): Additional instructions to the retrieval algorithm. This could be used to customize the retrieval process, for example by specifying certain areas of the search space to prioritize or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific retrieval algorithms. Varies by `Engine`.

        Returns:
            T: The retrieved object. The exact type and structure of this depends on the `Engine` used.

        Example:
            >>> engine = Engine()
            >>> class Person:
            ...    name: str
            ...    bio: str
            ...    thoughts: list[str]
            ...    friends: list[Person]
            >>> john, teyoni, huimin = ... # create people
            >>> person = engine.retrieve(john, instructions="find john's least favorite friend")
        """

        try:
            return type(object).__tc_retrieve__(
                self,
                object,
                count=count,
                allowed_glob=allowed_glob,
                disallowed_glob=disallowed_glob,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._retrieve(
            object,
            count=count,
            allowed_glob=allowed_glob,
            disallowed_glob=disallowed_glob,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @abstractmethod
    def _retrieve(
        self,
        object: composite_types[T],
        /,
        count: int,
        allowed_glob: str,
        disallowed_glob: str,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
