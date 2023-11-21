from __future__ import annotations
from abc import ABC

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import _DataclassT, dataclass
from enum import Enum
import functools
from functools import singledispatchmethod
import inspect
from pathlib import Path
import pickle
from textwrap import dedent
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
from tensacode.base.mixins.supports_encode import SupportsEncodeMixin
import typingx
import pydantic, sqlalchemy, dataclasses, attr, typing

from langchain.output_parsers.boolean import BooleanOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import RetryWithErrorOutputParser
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
from tensacode.llm_engine.base_llm_engine import BaseLLMEngine
import tensacode.base.mixins as mixins


S = TypeVar("S")


class SupportsDecideMixin(
    Generic[T, R],
    # just about every mixin needs to inherit from SupportsEncodeMixin
    SupportsEncodeMixin[T, R],
    BaseLLMEngine[T, R],
    mixins.SupportsDecideMixin[T, R],
    ABC,
):
    kernel = BaseLLMEngine[T, R].kernel
    parser = BooleanOutputParser()
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=kernel)

    template_no_fns = dedent(
        """\
            DECIDE based on the following condition: {condition}
            
            Input: {data}
            
            Output either {true_val} or {false_val}.
            
            Output: 
        """
    )
    prompt_no_fns = PromptTemplate(
        template=template_no_fns,
        input_variables=["data", "condition"],
        partial_variables={
            "true_val": parser.true_val,
            "false_val": parser.false_val,
        },
    )
    pipeline_no_fns = prompt_no_fns | kernel | retry_parser

    template_if_true_fn_only = dedent(
        """\
            DECIDE based on the following condition: {condition}
            
            Input: {data}
            
            Output either {true_val} or {false_val}.
            
            If {true_val}, then {true_fn_invocation} will be invoked.
            
            Output: 
        """
    )
    prompt_if_true_fn_only = PromptTemplate(
        template=template_if_true_fn_only,
        input_variables=["data", "condition", "true_fn_invocation"],
        partial_variables={
            "true_val": parser.true_val,
            "false_val": parser.false_val,
        },
    )
    pipeline_if_true_fn_only = prompt_if_true_fn_only | kernel | retry_parser

    template_if_false_fn_only = dedent(
        """\
            DECIDE based on the following condition: {condition}
            
            Input: {data}
            
            Output either {true_val} or {false_val}.
            
            If {false_val}, then {false_fn_invocation} will be invoked.
            
            Output: 
        """
    )
    prompt_if_false_fn_only = PromptTemplate(
        template=template_if_false_fn_only,
        input_variables=["data", "condition", "false_fn_invocation"],
        partial_variables={
            "true_val": parser.true_val,
            "false_val": parser.false_val,
        },
    )
    pipeline_if_false_fn_only = prompt_if_false_fn_only | kernel | retry_parser

    template_both_fns = dedent(
        """\
            DECIDE based on the following condition: {condition}
            
            Input: {data}
            
            Output either {true_val} or {false_val}.
            
            If {true_val}, then {true_fn_invocation} will be invoked. If {false_val}, then {false_fn_invocation} will be invoked.
            
            Output: 
        """
    )
    prompt_2_fns = PromptTemplate(
        template=template_both_fns,
        input_variables=[
            "data",
            "condition",
            "true_fn_invocation",
            "false_fn_invocation",
        ],
        partial_variables={
            "true_val": parser.true_val,
            "false_val": parser.false_val,
        },
    )
    pipeline_2_fns = prompt_2_fns | kernel | retry_parser

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
                decision = self.pipeline_no_fns.invoke(
                    {
                        "data": data,
                        "condition": condition,
                    }
                )
                return decision

            case None, _:

                @functools.wraps(if_false)
                def deferred_fn(*args, **kwargs):
                    false_invocation = self._render_invocation(
                        if_false,
                        args=args,
                        kwargs=kwargs,
                    )
                    decision = self.pipeline_if_false_fn_only.invoke(
                        {
                            "data": data,
                            "condition": condition,
                            "false_fn_invocation": false_invocation,
                        }
                    )
                    match decision:
                        case True:
                            return None
                        case False:
                            return if_false(*args, **kwargs)

                return deferred_fn

            case _, None:

                @functools.wraps(if_true)
                def deferred_fn(*args, **kwargs):
                    true_invocation = self._render_invocation(
                        if_true,
                        args=args,
                        kwargs=kwargs,
                    )
                    decision = self.pipeline_if_true_fn_only.invoke(
                        {
                            "data": data,
                            "condition": condition,
                            "true_fn_invocation": true_invocation,
                        }
                    )
                    match decision:
                        case True:
                            return if_true(*args, **kwargs)
                        case False:
                            return None

                return deferred_fn
            case _, _:

                def deferred_fn(*args, **kwargs):
                    true_invocation = self._render_invocation(
                        if_true,
                        args=args,
                        kwargs=kwargs,
                    )
                    false_invocation = self._render_invocation(
                        if_false,
                        args=args,
                        kwargs=kwargs,
                    )
                    decision = self.pipeline_2_fns.invoke(
                        {
                            "data": data,
                            "condition": condition,
                            "true_fn_invocation": true_invocation,
                            "false_fn_invocation": false_invocation,
                        }
                    )
                    match decision:
                        case True:
                            return if_true(*args, **kwargs)
                        case False:
                            return if_false(*args, **kwargs)

                return deferred_fn
