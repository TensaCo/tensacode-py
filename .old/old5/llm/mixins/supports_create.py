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
from tensacode.llm.base_llm_engine import BaseLLMEngine
import tensacode.base.mixins as mixins


class SupportsCreateMixin(
    Generic[T, R], BaseLLMEngine[T, R], mixins.SupportsCreateMixin[T, R], ABC
):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.autoconvert

    @dynamic_defaults()
    @trace()
    def create(
        self,
        object_enc: R,
        /,
        depth_limit: int = DefaultParam(qualname="hparams.create.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.create.instructions"),
        **kwargs,
    ) -> T:
        """
        Like decode, but also determines the type you want to create.
        """
