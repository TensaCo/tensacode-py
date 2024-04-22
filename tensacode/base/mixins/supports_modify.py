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

from tensacode.base.base_engine import BaseEngine


class SupportsModifyMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.autoconvert

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def modify(
        self,
        object: T,
        /,
        depth_limit: int = DefaultParam(qualname="hparams.modify.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.modify.instructions"),
        **kwargs,
    ) -> T:
        """
        Modifies an object *in place* based on the `instructions`.

        Args:
            object (composite_types[T]): The type of object to be modified.
            depth_limit (int): The maximum depth to which the modification process should recurse. This is useful for controlling the complexity of the modification, especially for deeply nested structures. Default is set in the engine's hyperparameters.
            instructions (enc[str]): Additional instructions to the modification algorithm. This could be used to customize the modification process, for example by specifying certain areas of the search space to prioritize or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific modification algorithms. Varies by `Engine`.

        Returns:
            T: The modified object. The exact type and structure of this depends on the `Engine` used.

        Example:
            >>> engine = Engine()
            >>> class Person:
            ...    name: str
            ...    bio: str
            ...    thoughts: list[str]
            ...    friends: list[Person]
            >>> john, teyoni, huimin = ... # create people
            >>> engine.modify(john, instructions="john's favorite color is blue")
        """
        try:
            return type(object).__tc_modify__(
                self,
                object,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._modify(object, depth_limit=depth_limit, instructions=instructions)

    @abstractmethod
    def _modify(
        self,
        object: T,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
