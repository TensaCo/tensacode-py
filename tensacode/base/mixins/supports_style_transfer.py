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


class SupportsStyleTransferMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def style_transfer(
        self,
        object: T,
        style: enc[T] = None,
        exemplar: T = None,
        /,
        depth_limit: int = DefaultParam(
            qualname="hparams.style_transfer.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            qualname="hparams.style_transfer.instructions",
        ),
        **kwargs,
    ) -> T:
        """
        Performs style transfer on the given object.

        Args:
            object (T): The object to perform style transfer on.
            style (enc[T], optional): The style to transfer. If not provided, an exemplar must be given. Defaults to None.
            exemplar (T, optional): An exemplar object to guide the style transfer. If not provided, a style must be given. Defaults to None.
            depth_limit (int, optional): The maximum depth to explore for style transfer. Defaults to engine.correct.depth_limit.
            instructions (enc[str], optional): Encoded instructions for the engine. Defaults to engine.correct.instructions.
            **kwargs: Additional keyword arguments.

        Returns:
            T: The object after style transfer.
        """
        try:
            return type(object).__tc_style_transfer__(
                self,
                object,
                style=style,
                exemplar=exemplar,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._style_transfer(
            object,
            style=style,
            exemplar=exemplar,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @abstractmethod
    def _style_transfer(
        self,
        object: T,
        style: R,
        exemplar: T,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
