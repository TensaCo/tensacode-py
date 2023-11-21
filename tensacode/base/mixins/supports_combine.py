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


class SupportsCombineMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def combine(
        self,
        objects: Sequence[T],
        /,
        depth_limit: int = DefaultParam(qualname="hparams.combine.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.combine.instructions"),
        **kwargs,
    ) -> T:
        """
        Combines multiple objects into a single object.

        This method is used to combine multiple objects into a single object based on the provided parameters. The combined object is returned in the form specified by the 'objects' parameter.

        Args:
            objects (Sequence[T]): The sequence of objects to be combined.
            depth_limit (int): The maximum depth to which the combination process should recurse. This is useful for controlling the complexity of the combination, especially for deeply nested structures. Default is set in the engine's hyperparameters.
            instructions (enc[str]): Additional instructions to the combination algorithm. This could be used to customize the combination process, for example by specifying certain areas of the search space to prioritize or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific combination algorithms. Varies by `Engine`.

        Returns:
            T: The combined object. The exact type and structure of this depends on the `Engine` used.

        Example:
            >>> engine = Engine()
            >>> class Person:
            ...    name: str
            ...    bio: str
            ...    thoughts: list[str]
            ...    friends: list[Person]
            >>> john, teyoni, huimin = ... # create people
            >>> group = engine.combine([john, teyoni, huimin], instructions="make them into a composite person")
            >>> print(group)
            ... Person(name="John, Teyoni, and Huimin", bio="...", thoughts=["...", "...", "..."], friends=[...])
        """
        try:
            return type(objects[0]).__tc_combine__(
                self,
                objects,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._combine(
            objects, depth_limit=depth_limit, instructions=instructions, **kwargs
        )

    @abstractmethod
    def _combine(
        self,
        objects: Sequence[T],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
