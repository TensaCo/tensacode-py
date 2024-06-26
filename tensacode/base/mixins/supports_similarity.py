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


class SupportsSimilarityMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.autoconvert

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def similarity(
        self,
        objects: tuple[T],
        /,
        depth_limit: int = DefaultParam(qualname="hparams.similarity.depth_limit"),
        instructions: enc[str] = DefaultParam(
            qualname="hparams.similarity.instructions"
        ),
        **kwargs,
    ) -> float:
        """
        Calculates the similarity between the given objects.

        Args:
            objects (tuple[T]): The objects to compare.
            depth_limit (int, optional): The maximum depth to explore. Defaults to hparams.similarity.depth_limit.
            instructions (enc[str], optional): Encoded instructions for the engine. Defaults to hparams.similarity.instructions.

        Returns:
            float: The similarity score between the objects.
        """
        try:
            return type(objects[0]).__tc_similarity__(
                self,
                objects,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._similarity(
            objects, depth_limit=depth_limit, instructions=instructions, **kwargs
        )

    @abstractmethod
    def _similarity(
        self,
        objects: tuple[T],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> float:
        raise NotImplementedError()
