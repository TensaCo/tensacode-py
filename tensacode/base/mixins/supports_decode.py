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
import sys
from types import FunctionType, ModuleType
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
    Union,
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
    is_callable_type,
    is_object,
    tree_types,
    tree,
    DataclassInstance,
    AttrsInstance,
    is_type,
)
from tensacode.utils.internal_types import make_union, nested_dict
from tensacode.base.base_engine import BaseEngine


class SupportsDecodeMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @trace()
    def decode(
        self,
        object_enc: R,
        type: type[T],
        /,
        depth_limit: int = DefaultParam(qualname="hparams.decode.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.decode.instructions"),
        **kwargs,
    ) -> T:
        """
        Decodes an encoded representation of an object back into its original form or into a different form.

        One of the powerful features of this function is its ability to decode into a different type or modality than the original object. This is controlled by the `type` argument. For example, you could encode a text document into a vector representation, and then decode it into a different language or a summary.

        Args:
            object_enc (R): The encoded representation of the object to be decoded.
            type (type[T]): The expected type of the decoded object. This is used to guide the decoding process. It doesn't have to match the original type of the object before encoding.
            depth_limit (int): The maximum depth to which the decoding process should recurse. This is useful for controlling the complexity of the decoding, especially for deeply nested structures. Default is set in the engine's parameters.
            instructions (enc[str]): Additional instructions to the decoding algorithm. This could be used to customize the decoding process, for example by specifying certain features to focus on or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific decoding algorithms.

        Returns:
            T: The decoded object. The exact type and structure of this depends on the decoding algorithm used and the `type` argument.

        Examples:
            >>> engine = Engine()
            >>> encoded_obj = <encoded representation of an object>
            >>> decoded_obj = engine.decode(encoded_obj, type=NewObjectType)
            >>> print(decoded_obj)
            # Output: <decoded representation of the object in the new type>
        """

        try:
            return type(object_enc).__tc_decode__(
                self,
                object_enc,
                type,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @abstractmethod
    def _decode(
        self,
        object_enc: R,
        type: type[T],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: is_type(type))
    @abstractmethod
    def _decode_to_object(
        self,
        object_enc: R,
        type: type[T],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: is_callable_type(type))
    @abstractmethod
    def _decode_to_function(
        self,
        object_enc: R,
        type: type[FunctionType],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, pydantic.BaseModel))
    @abstractmethod
    def _decode_to_pydantic_model_instance(
        self,
        object_enc: R,
        type: type[pydantic.BaseModel],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, NamedTuple))
    @abstractmethod
    def _decode_to_namedtuple_instance(
        self,
        object_enc: R,
        type: type[NamedTuple],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: is_dataclass_type(type))
    @abstractmethod
    def _decode_to_dataclass_instance(
        self,
        object_enc: R,
        type: type[DataclassInstance],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    # TODO: i need to figure out the decode_to_*_instance v. decode_to_*_type distinction
    # because I want users to be able to do both with only changing on parameter

    @_decode.overload(lambda type: issubclass(type, type))
    @abstractmethod
    def _decode_to_type(
        self,
        object_enc: R,
        type: type,
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, type[pydantic.BaseModel]))
    @abstractmethod
    def _decode_to_pydantic_model_type(
        self,
        object_enc: R,
        type: type[type[pydantic.BaseModel]],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, type[NamedTuple]))
    @abstractmethod
    def _decode_to_namedtuple_type(
        self,
        object_enc: R,
        type: type[type[NamedTuple]],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, type))
    @abstractmethod
    def _decode_to_dataclass_type(
        self,
        object_enc: R,
        type: type[type],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: typingx.issubclassx(type, ModuleType))
    @abstractmethod
    def _decode_to_module(
        self,
        object_enc: R,
        type: type[ModuleType],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        register_module=True,
        **kwargs,
    ) -> T:
        module_dict = self._decode_to_dict(
            object_enc,
            dict[str, Any],
            depth_limit=depth_limit,
            instructions=f"Generate the module __dict__ items. {instructions}",
            **kwargs,
        )
        module_name = self.decode_to_str(
            object_enc,
            str,
            depth_limit=depth_limit,
            instructions=f"Generate the module name. {instructions}",
            **kwargs,
        )
        module = ModuleType(module_name)
        module.__dict__.update(module_dict)
        if register_module:
            sys.modules[module_name] = module
        return module

    @_decode.overload(lambda type: issubclass(type, Iterable[T]))
    @abstractmethod
    def _decode_to_iterable(
        self,
        object_enc: R,
        type: type[Iterable[T]],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        num_samples: int | None = None,
        decide_samples: Literal["per_iteration", "at_start"] = "at_start",
        **kwargs,
    ) -> T:
        elem_type = typingx.get_args(type).get(0) or object
        latent = dict(
            input=object_enc,
            output=[],
            type=type,
            elem_type=elem_type,
            examples=examples,
            depth_limit=depth_limit,
            instructions=instructions,
        )
        match decide_samples:
            case "per_iteration":
                while self.decode_to_bool(
                    latent,
                    instructions="DECIDE whether to continue generating sequence items or not",
                ):
                    next = self._decode(
                        object_enc,
                        elem_type,
                        depth_limit=depth_limit,
                        instructions=instructions,
                        **kwargs,
                    )
                    latent["output"].append(next)
                    yield next
            case "at_start":
                if num_samples is None:
                    num_samples = self.decode_to_int(
                        latent, ge=0, instructions="DECIDE num sequence elements"
                    )
                for _ in range(num_samples):
                    next = self._decode(
                        object_enc,
                        elem_type,
                        depth_limit=depth_limit,
                        instructions=instructions,
                        **kwargs,
                    )
                    latent["output"].append(next)
                    yield next
            case _:
                raise ValueError(f"Invalid value for decide_samples: {decide_samples}")

    @_decode.overload(lambda type: typingx.issubclassx(type, Sequence[T]))
    def _decode_to_seq(
        self,
        object_enc: R,
        type: type[Sequence[T]] = list[T],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        num_samples: int | None = None,
        decide_samples: Literal["per_iteration", "at_start"] = "at_start",
        **kwargs,
    ) -> Sequence[T]:
        it = self._decode_to_iterable(
            object_enc,
            type,
            examples=examples,
            depth_limit=depth_limit,
            instructions=instructions,
            num_samples=num_samples,
            decide_samples=decide_samples,
            **kwargs,
        )
        return type(it)

    @_decode.overload(lambda type: typingx.issubclassx(type, tuple[T]))
    def _decode_to_tuple(
        self,
        object_enc: R,
        type: type[tuple] = tuple,
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        num_samples: int | None = None,
        decide_samples: Literal["per_iteration", "at_start"] = "at_start",
        **kwargs,
    ) -> T:
        elem_types = typingx.get_args(type)

        # if tuple has no args, size is unknown
        if elem_types == ():
            it = self._decode_to_iterable(
                object_enc,
                type[Any, ...],
                examples=examples,
                depth_limit=depth_limit - 1,
                instructions=instructions,
                num_samples=num_samples,
                decide_samples=decide_samples,
                **kwargs,
            )
            return type(it)

        # if tuple has definite args, size is known
        elif len(elem_types) > 0 and all(
            elem_type is not ... for elem_type in elem_types
        ):
            elems = [
                self._decode(
                    object_enc,
                    elem_type,
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    **kwargs,
                )
                for elem_type in elem_types
            ]
            return type(elems)

        # if tuple has ... args, size is unknown
        elif len(elem_types) > 0 and any(elem_type is ... for elem_type in elem_types):
            elem_types_before_ellipsis = elem_types[: elem_types.index(...)]
            elem_types_after_ellipsis = elem_types[elem_types.index(...) + 1 :]

            elems_before_ellipsis = [
                self._decode(
                    object_enc,
                    elem_type,
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    **kwargs,
                )
                for elem_type in elem_types_before_ellipsis
            ]
            elems_after_ellipsis = [
                self._decode(
                    object_enc,
                    elem_type,
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    **kwargs,
                )
                for elem_type in elem_types_after_ellipsis
            ]
            elems_around_ellipsis_it = self._decode_to_iterable(
                object_enc,
                Iterable[make_union[elem_types_before_ellipsis]],
                examples=examples,
                depth_limit=depth_limit - 1,
                instructions=instructions,
                num_samples=num_samples,
                decide_samples=decide_samples,
                **kwargs,
            )
            return type(
                elems_before_ellipsis
                + tuple(elems_around_ellipsis_it)
                + elems_after_ellipsis
            )

        else:
            raise ValueError(
                f"Invalid generic specification for tuple: {type}. This should never happen."
            )

    @_decode.overload(lambda type: typingx.issubclassx(type, set[T]))
    def _decode_to_set(
        self,
        object_enc: R,
        type: type[set[T] | frozenset[T]] = set[T],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        num_samples: int | None = None,
        decide_samples: Literal["per_iteration", "at_start"] = "at_start",
        **kwargs,
    ) -> T:
        seq = self._decode_to_seq(
            object_enc,
            type,
            examples=examples,
            depth_limit=depth_limit,
            instructions=instructions,
            num_samples=num_samples,
            decide_samples=decide_samples,
            **kwargs,
        )
        return type(seq)

    @_decode.overload(lambda type: typingx.issubclassx(type, Mapping[Any, T]))
    @abstractmethod
    def _decode_to_map(
        self,
        object_enc: R,
        type: type[Mapping[Any, T]],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        num_samples: int | None = None,
        decide_samples: Literal["per_iteration", "at_start"] = "per_iteration",
        keys: Iterable[Any] = None,
        **kwargs,
    ) -> T:
        key_type, value_type = typingx.get_args(type)
        it = self._decode_to_iterable(
            object_enc,
            list[tuple[key_type, value_type]],
            examples=examples,
            depth_limit=depth_limit,
            instructions=instructions,
            num_samples=num_samples,
            decide_samples=decide_samples,
            **kwargs,
        )
        return type(dict(it))

    @_decode.overload(lambda type: issubclass(type, None))
    @abstractmethod
    def _decode_to_none(
        self,
        object_enc: R,
        type: type[None],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, bool))
    @abstractmethod
    def _decode_to_bool(
        self,
        object_enc: R,
        type: type[bool],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, int))
    def _decode_to_int(
        self,
        object_enc: R,
        type: type[int],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, float))
    @abstractmethod
    def _decode_to_float(
        self,
        object_enc: R,
        type: type[float],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, complex))
    @abstractmethod
    def _decode_to_complex(
        self,
        object_enc: R,
        type: type[complex],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, str))
    @abstractmethod
    def _decode_to_str(
        self,
        object_enc: R,
        type: type[str],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @_decode.overload(lambda type: issubclass(type, bytes))
    @abstractmethod
    def _decode_to_bytes(
        self,
        object_enc: R,
        type: type[bytes],
        /,
        examples: list[T] = None,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> T:
        return self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )
