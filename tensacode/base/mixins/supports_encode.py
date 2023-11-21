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
from tensacode.llm_engine.base_llm_engine import BaseLLMEngine
import tensacode.base.mixins as mixins


from tensacode.base.base_engine import BaseEngine
from tensacode.utils.decorators import overloaded


class SupportsEncodeMixin(BaseEngine):
    encoded_args = BaseEngine.encoded_args
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def encode(
        self,
        object: T,
        /,
        depth_limit: int = DefaultParam(qualname="hparams.encode.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.encode.instructions"),
        **kwargs,
    ) -> R:
        """
        Produces an encoded representation of the `object`.

        Encodings are useful for creating a common representation of objects that can be compared for similarity, fed into a neural network, or stored in a database. This method uses a specific encoding algorithm (which can be customized) to convert the input object into a format that is easier to process and analyze.

        You can customize the encoding algorithm by either subclassing `Engine` or adding a `__tc_encode__` classmethod to `object`'s type class. The `__tc_encode__` method should take in the same arguments as `Engine.encode` and return the encoded representation of the object. See `Engine.Proto.__tc_encode__` for an example.

        Args:
            object (T): The object to be encoded. This could be any data structure like a list, dictionary, custom class, etc.
            depth_limit (int): The maximum depth to which the encoding process should recurse. This is useful for controlling the complexity of the encoding, especially for deeply nested structures. Default is set in the engine's parameters.
            instructions (enc[str]): Additional instructions to the encoding algorithm. This could be used to customize the encoding process, for example by specifying certain features to focus on or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific encoding algorithms. Varies by `Engine`.

        Returns:
            R: The encoded representation of the object. The exact type and structure of this depends on the `Engine` used.

        Examples:
            >>> engine = Engine()
            >>> obj = {"name": "John", "age": 30, "city": "New York"}
            >>> encoded_obj = engine.encode(obj)
            >>> print(encoded_obj)
            # Output: <encoded representation of obj>
        """
        try:
            return type(object).__tc_encode__(
                self,
                object,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        self._encode(
            object, depth_limit=depth_limit, instructions=instructions, **kwargs
        )

    @overloaded
    @abstractmethod
    def _encode(
        self,
        object: T,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        raise NotImplementedError()

    @_encode.overload(is_object_instance)
    @abstractmethod
    def _encode_object(
        self,
        object: object,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: callable(object))
    @abstractmethod
    def _encode_function(
        self,
        object: Callable,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(is_pydantic_model_instance)
    @abstractmethod
    def _encode_pydantic_model_instance(
        self,
        object: pydantic.BaseModel,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(is_namedtuple_instance)
    @abstractmethod
    def _encode_namedtuple_instance(
        self,
        object: NamedTuple,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(is_type)
    @abstractmethod
    def _encode_type(
        self,
        object: type,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(is_pydantic_model_type)
    @abstractmethod
    def _encode_pydantic_model_type(
        self,
        object: type[pydantic.BaseModel],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(is_namedtuple_type)
    @abstractmethod
    def _encode_namedtuple_type(
        self,
        object: type[NamedTuple],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: isinstance(object, ModuleType))
    @abstractmethod
    def _encode_module_type(
        self,
        object: ModuleType,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: object is None)
    @abstractmethod
    def _encode_none(
        self,
        object: None,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            **kwargs,
        )

    @_encode.overload(lambda object: isinstance(object, bool))
    @abstractmethod
    def _encode_bool(
        self,
        object: bool,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            **kwargs,
        )

    @_encode.overload(lambda object: isinstance(object, int))
    @abstractmethod
    def _encode_int(
        self,
        object: int,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            **kwargs,
        )

    @_encode.overload(lambda object: isinstance(object, float))
    @abstractmethod
    def _encode_float(
        self,
        object: float,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            **kwargs,
        )

    @_encode.overload(lambda object: isinstance(object, complex))
    @abstractmethod
    def _encode_complex(
        self,
        object: complex,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            **kwargs,
        )

    @_encode.overload(lambda object: isinstance(object, str))
    @abstractmethod
    def _encode_str(
        self,
        object: str,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            **kwargs,
        )

    @_encode.overload(lambda object: isinstance(object, bytes))
    @abstractmethod
    def _encode_bytes(
        self,
        object: bytes,
        /,
        depth_limit: int | None = None,
        bytes_per_group=4,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            **kwargs,
        )

    @_encode.overload(lambda object: isinstance(object, Iterable))
    @abstractmethod
    def _encode_iterable(
        self,
        object: Iterable,
        /,
        depth_limit: int | None = None,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: typingx.isinstance(object, Sequence[T]))
    @abstractmethod
    def _encode_seq(
        self,
        object: Sequence,
        /,
        depth_limit: int | None = None,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: typingx.isinstance(object, Set[T]))
    @abstractmethod
    def _encode_set(
        self,
        object: set,
        /,
        depth_limit: int | None = None,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: typingx.isinstance(object, Mapping[Any, T]))
    @abstractmethod
    def _encode_map(
        self,
        object: Mapping,
        /,
        depth_limit: int | None = None,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        return self._encode(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            force_inline=force_inline,
            **kwargs,
        )
