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


class SupportsChoiceMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    class Branch(NamedTuple):
        condition: enc[T]
        function: Callable[..., T]
        args: tuple[Any, ...]
        kwargs: dict[str, Any]

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def choice(
        self,
        data: enc[T],
        branches: tuple[SupportsChoiceMixin.Branch, ...],
        /,
        mode: Literal["first-winner", "last-winner"] = "first-winner",
        default_case_idx: int | None = None,
        threshold: float = DefaultParam("hparams.choice.threshold"),
        randomness: float = DefaultParam("hparams.choice.randomness"),
        depth_limit: int = DefaultParam(qualname="hparams.choice.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.choice.instructions"),
        **kwargs,
    ) -> T:
        """
        Executes a choice operation based on the provided conditions and functions.

        Args:
            data (T): The data to be used in the choice operation.
            branches (tuple[SupportsChoiceMixin.Branch, ...]): The branches of the choice operation. Each branch is a tuple of a condition, a function, and the arguments and keyword arguments to be passed to the function.
            mode (str): The mode of operation. Can be either "first-winner" or "last-winner".
            default_case_idx (int, optional): The index of the default case to use if no condition surpasses the threshold. Defaults to None.
            threshold (float): The threshold value for condition evaluation.
            randomness (float): The randomness factor in choice selection.
            depth_limit (int): The maximum depth for recursion.
            instructions (str): Additional instructions for the choice operation.
            **kwargs: Additional keyword arguments.

        Returns:
            T: The result of the executed function corresponding to the winning condition.
        """
        match mode:
            case "first-winner":
                # evaluate the conditions in order
                # pick first one to surpass threshold
                # default to default_case or raise ValueError if no default_case
                return self._choice_first_winner(
                    data,
                    branches,
                    default_case_idx=default_case_idx,
                    threshold=threshold,
                    randomness=randomness,
                    depth_limit=depth_limit,
                    instructions=instructions,
                    **kwargs,
                )
            case "last-winner":
                # evaluate all conditions
                # pick global max
                # default to default_case or raise ValueError if no default_case
                return self._choice_last_winner(
                    data,
                    branches,
                    default_case_idx=default_case_idx,
                    threshold=threshold,
                    randomness=randomness,
                    depth_limit=depth_limit,
                    instructions=instructions,
                    **kwargs,
                )
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    @abstractmethod
    def _choice_first_winner(
        self,
        data: R,
        branches: tuple[SupportsChoiceMixin.Branch, ...],
        /,
        default_case_idx: int | None,
        threshold: float,
        randomness: float,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @abstractmethod
    def _choice_last_winner(
        self,
        data: R,
        branches: tuple[SupportsChoiceMixin.Branch, ...],
        /,
        default_case_idx: int | None,
        threshold: float,
        randomness: float,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
