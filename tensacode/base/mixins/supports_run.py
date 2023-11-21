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


class SupportsRunMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def run(
        self,
        instructions: enc[str] = DefaultParam(qualname="hparams.run.instructions"),
        /,
        budget: Optional[float] = DefaultParam(qualname="hparams.run.budget"),
        **kwargs,
    ) -> Any:
        """
        Executes the engine with the given instructions and an optional budget.

        Args:
            instructions (enc[str]): Encoded instructions for the engine.
            budget (float, optional): The budget for the engine to run. Defaults to None.

        Returns:
            Any: The result of the engine run, if any.
        """
        return self._run(instructions, budget=budget, **kwargs)

    @abstractmethod
    def _run(
        self,
        instructions: R | None = None,
        format: Literal["block", "inline"] = "inline",
        visibility: Literal["public", "protected", "private"] = "public",
        /,
        budget: Optional[float],
        **kwargs,
    ) -> Any:
        raise NotImplementedError()
