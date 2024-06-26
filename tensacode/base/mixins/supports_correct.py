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


class SupportsCorrectMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.autoconvert

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def correct(
        self,
        object: T,
        /,
        threshold: float = DefaultParam(qualname="hparams.correct.threshold"),
        depth_limit: int = DefaultParam(qualname="hparams.correct.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.correct.instructions"),
        **kwargs,
    ) -> T:
        """
        Corrects the given object based on the provided threshold, depth limit, and instructions.

        Args:
            object (T): The object to correct.
            threshold (float, optional): The threshold for correction. Defaults to hparams.correct.threshold.
            depth_limit (int, optional): The maximum depth to explore for correction. Defaults to hparams.correct.depth_limit.
            instructions (enc[str], optional): Encoded instructions for the engine. Defaults to hparams.correct.instructions.

        Returns:
            T: The corrected object.
        """
        try:
            return type(object).__tc_correct__(
                self,
                object,
                threshold=threshold,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._correct(
            object,
            threshold=threshold,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @abstractmethod
    def _correct(
        self,
        object: T,
        /,
        threshold: float,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
