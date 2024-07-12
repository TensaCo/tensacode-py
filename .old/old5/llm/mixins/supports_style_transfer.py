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


class SupportsStyleTransferMixin(
    Generic[T, R], BaseLLMEngine[T, R], mixins.SupportsStyleTransferMixin[T, R], ABC
):
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
