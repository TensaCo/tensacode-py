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
from pydantic import BaseModel, Field
import typingx
import pydantic, sqlalchemy, dataclasses, attr, typing


import tensacode as tc
from tensacode._internal.decorators import overloaded
from tensacode._internal.depends import InjectDefaults, depends, inject
from tensacode._internal.typing import Action, Example
from tensacode.base.base_engine import BaseEngine
from tensacode.llm.base_llm_engine import BaseLLMEngine
import tensacode.base.mixins as mixins


from tensacode.base.base_engine import BaseEngine, T, R


@InjectDefaults
class SupportsEncodeMixin(Generic[T, R], BaseEngine[T, R]):
    trace = BaseEngine.trace
    autoconvert = BaseEngine.autoconvert
    param = BaseEngine.param

    class EncodeArgs(BaseEngine.Args):
        pass

    default_encode_args: EncodeArgs = EncodeArgs()

    @inject()
    @autoconvert()
    @trace()
    def encode(self, object: T, /, args: EncodeArgs = None) -> R:
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
        if not args:
            args = self.EncodeArgs()
        self._get_defaults(args, names=("default_args", "default_encode_args"))
        with self.with_instructions(args.instructions), self.with_examples(
            args.examples
        ):
            if hasattr(object, "__tc_encode__"):
                return object.__tc_encode__(self, object, args=args)
            else:
                return self._encode(object, args=args)

    @overloaded
    @abstractmethod
    def _encode(self, object: T, /, args: EncodeArgs) -> R:
        raise NotImplementedError()

    @_encode.overload(types.is_type(types.object))
    @abstractmethod
    def _encode(self, object: types.object, /, args: EncodeArgs) -> R:
        raise NotImplementedError()

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
