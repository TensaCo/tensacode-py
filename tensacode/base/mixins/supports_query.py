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


class SupportsQueryMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def query(
        self,
        object: T,
        /,
        query: enc[T],
        depth_limit: int = DefaultParam(qualname="hparams.query.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.query.instructions"),
        **kwargs,
    ) -> R:
        """
        Extracts latent information from the `object` based on the `query`.

        This method is used to extract information from the `object` based on the `query`. The `query` is a encoded string that specifies what information to extract from the `object`.

        Args:
            object (T): The object to extract information from.
            query (enc[T]): The query specifying what information to extract.
            depth_limit (int, optional): The maximum depth to which the extraction process should recurse. Defaults to engine's hyperparameters.
            instructions (enc[str], optional): Additional instructions to the extraction algorithm. Defaults to engine's hyperparameters.
            **kwargs: Additional keyword arguments that might be needed for specific extraction algorithms.

        Returns:
            R: The extracted information. The exact type and structure of this depends on the `Engine` used.

        Example:
            >>> engine = Engine()
            >>> class Person:
            ...    name: str
            ...    bio: str
            ...    thoughts: list[str]
            ...    friends: list[Person]
            >>> john, teyoni, huimin = ... # create people
            >>> info = engine.query(john, query="Does John know that Teyoni has a crush on him?")
            >>> engine.decode(info, type=bool)
            ... True
        """

        try:
            return type(object).__tc_query__(
                self,
                object,
                query=query,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._query(object, query=query, depth_limit=depth_limit, **kwargs)

    @abstractmethod
    def _query(
        self,
        object: T,
        /,
        query: R,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        raise NotImplementedError()
