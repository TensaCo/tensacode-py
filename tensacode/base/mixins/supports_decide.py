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
from tensacode.base.mixins.supports_choice import SupportsChoiceMixin

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

S = TypeVar("S")


class SupportsDecideMixin(
    Generic[T, R],
    SupportsChoiceMixin[T, R],
    BaseEngine[T, R],
    ABC,
):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args
    Branch = SupportsChoiceMixin.Branch

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def decide(
        self,
        data: enc[T],
        condition: enc[T],
        if_true: Callable[..., S] = None,
        if_false: Callable[..., S] = None,
        *,
        threshold: float = DefaultParam("hparams.choice.threshold"),
        randomness: float = DefaultParam("hparams.choice.randomness"),
        depth_limit: int = DefaultParam(qualname="hparams.choice.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.choice.instructions"),
        **kwargs,
    ) -> bool | (S | None):
        """
        Makes a decision based on the provided condition. If the condition is met, the `if_true` function is called. Otherwise, the `if_false` function is called.

        Can be called in two ways:
        - `fn_result = engine.decide(condition, if_true=..., if_false=..., ...)`: in this case, you define the true and false branches as separate functions and pass them in as arguments. `engine.decide` then calls the appropriate function based on the condition.
        - `bool_result = engine.decide(condition, ...)`: in this case, you don't define the true and false branches, and `engine.decide` returns a boolean value based on the condition.

        Args:
            data (enc[T]): The data to make the decision on.
            condition (enc[T]): The condition to evaluate.
            if_true (Callable): The function to call if the condition is met.
            if_false (Callable): The function to call if the condition is not met.
            threshold (float): The threshold for making the decision.
            randomness (float): The randomness factor in the decision making.
            depth_limit (int): The maximum depth to which the decision process should recurse.
            instructions (enc[str]): Additional instructions to the decision algorithm.
            **kwargs: Additional keyword arguments that might be needed for specific decision algorithms.

        Returns:
            The result of the `if_true` function if the condition is met, or the result of the `if_false` function if the condition is not met.

        Example:
            >>> engine = Engine()
            >>> class Person:
            ...    name: str
            ...    bio: str
            ...    thoughts: list[str]
            ...    friends: list[Person]
            >>> john, teyoni, huimin = ... # create people
            >>> person = engine.retrieve(john, instructions="find john's least favorite friend")
            ... # based on John's friends, decide if he is a jerk or not
            >>> engine.decide(person, if_true=lambda: print("John is a jerk"), if_false=lambda: print("John is a nice guy"))
            ... John is a jerk
        """
        return self._decide(
            data,
            condition,
            if_true,
            if_false,
            threshold=threshold,
            randomness=randomness,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    def _decide(
        self,
        data: R,
        condition: R,
        if_true: Callable[..., S] | None,
        if_false: Callable[..., S] | None,
        **kwargs,
    ) -> bool | (S | None):
        match if_true, if_false:
            case None, None:
                latent = self.encode(
                    {
                        "data": data,
                        "condition": condition,
                    }
                )
                decision = self.decode_to_bool(latent)
                return decision
            case None, _:

                @functools.wraps(if_false)
                def deferred_fn(*args, **kwargs):
                    if_false_signature = inspect.signature(if_false)
                    if_false_params = if_false_signature.parameters
                    bound_values = if_false_signature.bind(*args, **kwargs)
                    param_values = {
                        name: bound_values.arguments.get(name, param.default)
                        for name, param in if_false_params.items()
                    }

                    latent = self.encode(
                        {
                            "data": data,
                            "condition": condition,
                            "if_true": None,
                            "if_false": if_false,
                            "params": param_values,
                        }
                    )
                    decision = self.decode_to_bool(latent)
                    match decision:
                        case True:
                            return None
                        case False:
                            return if_false(*args, **kwargs)

                return deferred_fn

            case _, None:

                @functools.wraps(if_true)
                def deferred_fn(*args, **kwargs):
                    if_true_signature = inspect.signature(if_false)
                    if_true_params = if_true_signature.parameters
                    bound_values = if_true_signature.bind(*args, **kwargs)
                    param_values = {
                        name: bound_values.arguments.get(name, param.default)
                        for name, param in if_true_params.items()
                    }

                    latent = self.encode(
                        {
                            "data": data,
                            "condition": condition,
                            "if_true": if_true,
                            "if_false": None,
                            "params": param_values,
                        }
                    )
                    decision = self.decode_to_bool(latent)
                    match decision:
                        case True:
                            return if_true(*args, **kwargs)
                        case False:
                            return None

                return deferred_fn
            case _, _:

                def deferred_fn(*args, **kwargs):
                    if_false_signature = inspect.signature(if_false)
                    if_false_params = if_false_signature.parameters
                    if_false_bound_values = if_false_signature.bind(*args, **kwargs)
                    if_false_param_values = {
                        name: if_false_bound_values.arguments.get(name, param.default)
                        for name, param in if_false_params.items()
                    }
                    if_true_signature = inspect.signature(if_false)
                    if_true_params = if_true_signature.parameters
                    if_true_bound_values = if_true_signature.bind(*args, **kwargs)
                    if_true_param_values = {
                        name: if_true_bound_values.arguments.get(name, param.default)
                        for name, param in if_true_params.items()
                    }
                    param_values = {
                        **if_false_param_values,
                        **if_true_param_values,
                    }

                    latent = self.encode(
                        {
                            "data": data,
                            "condition": condition,
                            "if_true": if_true,
                            "if_false": if_false,
                            "params": param_values,
                        }
                    )
                    decision = self.decode_to_bool(latent)
                    match decision:
                        case True:
                            return if_true(*args, **kwargs)
                        case False:
                            return if_false(*args, **kwargs)

                return deferred_fn
