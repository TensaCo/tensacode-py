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


class SupportsSplitMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def split(
        self,
        object: T,
        /,
        num_splits: int = DefaultParam(qualname="hparams.split.num_splits"),
        depth_limit: int = DefaultParam(qualname="hparams.split.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.split.instructions"),
        **kwargs,
    ) -> tuple[T]:
        """
        Splits an object into a specified number of parts.

        This method is used to split an object into a specified number of parts based on the provided parameters. The object is split in the form specified by the 'object' parameter.

        Args:
            object (T): The object to be split.
            num_splits (int): The number of parts to split the object into. Default is set in the engine's hyperparameters.
            depth_limit (int): The maximum depth to which the splitting process should recurse. This is useful for controlling the complexity of the splitting, especially for deeply nested structures. Default is set in the engine's hyperparameters.
            instructions (enc[str]): Additional instructions to the splitting algorithm. This could be used to customize the splitting process, for example by specifying certain areas of the search space to prioritize or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific splitting algorithms. Varies by `Engine`.

        Returns:
            tuple[T]: The split parts of the object. The exact type and structure of this depends on the `Engine` used.

        Example:
            >>> engine = Engine()
            >>> class Person:
            ...    name: str
            ...    bio: str
            ...    thoughts: list[str]
            ...    friends: list[Person]
            >>> john, teyoni, huimin = ... # create people
            >>> group = engine.combine([john, teyoni, huimin], instructions="make them into a composite person")
            >>> john_split, teyoni_split, huimin_split = engine.split(group, instructions="split them into their original forms")
            >>> print(john_split)
            ... Person(name="John", bio="...", thoughts=["..."], friends=[...])
        """
        try:
            return type(object).__tc_split__(
                self,
                object,
                num_splits=num_splits,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._split(
            object,
            num_splits=num_splits,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @abstractmethod
    def _split(
        self,
        object: T,
        /,
        num_splits: int,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> tuple[T]:
        raise NotImplementedError()
