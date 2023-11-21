from __future__ import annotations
from abc import ABC

from abc import ABC, abstractmethod
import builtins
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import _DataclassT, dataclass
import functools
from functools import singledispatchmethod
import inspect
from math import remainder
from pathlib import Path
import pickle
from textwrap import dedent, indent
from types import FunctionType, LambdaType, ModuleType
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
from pydantic.fields import FieldInfo
from box import Box
from uuid import uuid4
import attr
from jinja2 import Template
import loguru
from glom import glom
from pydantic import Field
from tensacode.utils.misc import get_keys, try_
from tensacode.utils.repr import render_block
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
    is_lambda,
    is_object,
    tree_types,
    tree,
    DataclassInstance,
    AttrsInstance,
    py_object,
)
from tensacode.utils.internal_types import nested_dict
from tensacode.base.base_engine import BaseEngine
from tensacode.llm_engine.base_llm_engine import BaseLLMEngine
import tensacode.base.mixins as mixins
import inspect_mate_pp


class SupportsEncodeMixin(
    Generic[T, R], BaseLLMEngine[T, R], mixins.HasEncodeMixin[T, R], ABC
):
    @overloaded
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
        return super()._encode_object(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: is_object(object))
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
        if depth_limit is not None and depth_limit <= 0:
            return

        if repr(object) != py_object.__repr__(object):
            return repr(object)

        keys = set(get_keys(object, visibility=visibility))
        attribute_keys = {
            k
            for k in keys
            if not inspect.isfunction(getattr(object, k))
            and not inspect.ismodule(getattr(object, k))
            and not inspect.isclass(getattr(object, k))
        }

        attribute_tuples = [
            (
                self._encode(
                    k,
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
                self._encode(
                    object.__annotations__.get(k, None),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
                self._encode(
                    getattr(object, k),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
            )
            for k in attribute_keys
        ]
        other_tuples = [
            (
                None,
                None,
                self._encode(
                    getattr(object, k),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=force_inline,
                    **kwargs,
                ),
            )
            for k in keys - attribute_keys
        ]

        return self._render_composite(
            self._encode_type(object.__class__, force_inline=True),
            tuples=attribute_tuples + other_tuples,
            docstring=inspect.getdoc(object),
            force_inline=force_inline,
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
        fn_start_keyword: str = "def",
        **kwargs,
    ) -> R:
        '''
        Generates a string representation of the given function including its name, parameters,
        annotations, default values, return type, and docstring, handling various parameter types.

        Args:
            func (Callable): The function to represent.

        Returns:
            str: The string representation of the function.

        Example usage:
            >>> def new_test_function(x: int, y: list[str], z: dict[str, float]) -> bool:
                    """This is a new test function with complex annotations."""
                    return True
                # Get the representation
            >>> self._encode_function(new_test_function)
            ... def new_test_function(x: int, y: list[str], z: dict[str, float]) -> bool:
                    """This is a new test function with complex annotations."""
                    return True

        '''
        func = object  # we can't change the args tho

        # Extract decorators and assign `func` to the innermost function
        decorators = []
        while hasattr(func, "__wrapped__"):
            decorators.append(func)
            func = func.__wrapped__

        # Get the function's name
        func_name = func.__name__

        # Get the signature of the function
        signature = inspect.signature(func)

        # Construct parameter string with annotations, default values, and special types
        params = []
        positional_only_separator_added = False
        for name, param in signature.parameters.items():
            annotation = (
                self._render(
                    param.annotation,
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    force_inline=True,
                )
                if param.annotation is not param.empty
                else None
            )

            # Handle variable positional (*args) and keyword (**kwargs) parameters
            if param.kind == param.VAR_POSITIONAL:
                param_str = f"*{name}: {annotation}" if annotation else name
                params.append(param_str)
            elif param.kind == param.VAR_KEYWORD:
                param_str = f"**{name}: {annotation}" if annotation else name
                params.append(param_str)
            else:
                param_str = f"{name}: {annotation}" if annotation else name
                # Add default value if present
                if param.default is not inspect.Parameter.empty:
                    param_str += f"={param.default}"
                params.append(param_str)

            # Add '/' after positional-only parameters
            if (
                param.kind == param.POSITIONAL_ONLY
                and not positional_only_separator_added
            ):
                params.append("/")
                positional_only_separator_added = True

        # Remove the last '/' if it's at the end of the list
        if params and params[-1] == "/":
            params.pop()

        params_str = ", ".join(params)

        header = f"{fn_start_keyword} {func_name}({params_str})"

        # Get the return type
        if (
            return_annotation := signature.return_annotation
            is not inspect.Signature.empty
        ):
            return_annotation_enc = self._encode(
                return_annotation,
                depth_limit=depth_limit - 1,
                instructions=instructions,
                visibility=visibility,
                force_inline=True,
            )
            header += f" -> {return_annotation_enc}"

        # Add decorators (if any)
        for decorator in reversed(decorators):
            rendered_dec = self._encode(
                decorator,
                depth_limit=depth_limit - 1,
                instructions=instructions,
                visibility="public",
                force_inline=True,
            )
            header = f"@{rendered_dec}\n{header}"

        # Get the body
        body: str
        docstring = inspect.getdoc(func)
        match force_inline, docstring:
            case True, None:
                body = "..."
            case False, None:
                body = try_(lambda: inspect.getsource(func)) or ""
            case True, docstring:
                body = f'"""{docstring}"""\n...'
            case False, docstring:
                body = (
                    f'"""{docstring}"""\n{try_(lambda: inspect.getsource(func)) or ""}'
                )

        # Combine everything into the final string
        return self._render_block(header, body).strip()

    @_encode.overload(lambda object: is_pydantic_model_instance(object))
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
        if depth_limit is not None and depth_limit <= 0:
            return

        keys = set(get_keys(object, visibility=visibility))
        field_keys = {k for k in keys if k in object.__fields__}
        attribute_keys = {
            k
            for k in keys
            if not inspect.isfunction(getattr(object, k))
            and not inspect.ismodule(getattr(object, k))
            and not inspect.isclass(getattr(object, k))
        }
        field_tuples = [
            (
                # althought we could technically just write `k`,
                # im doing this so we can later abstract to non-text engines
                self._encode(
                    k,
                    depth_limit=depth_limit,
                    instructions=instructions,
                    visibility="public",
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
                self._encode(
                    object.__fields__[k].type_,
                    depth_limit=depth_limit,
                    instructions=instructions,
                    visibility="public",
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
                self._render_field(getattr(object, k)),
            )
            for k in field_keys
        ]
        other_attribute_tuples = [
            (
                self._encode(
                    k,
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
                self._encode(
                    object.__annotations__.get(k, None),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
                self._encode(
                    getattr(object, k),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
            )
            for k in (keys & attribute_keys) - field_keys
        ]
        other_nonattribute_tuples = [
            (
                None,
                None,
                self._encode(
                    getattr(object, k),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=force_inline,
                    **kwargs,
                ),
            )
            for k in (keys - attribute_keys) - field_keys
        ]

        return self._render_composite(
            self._encode_type(object.__class__, force_inline=True),
            tuples=(field_tuples + other_attribute_tuples + other_nonattribute_tuples),
            docstring=inspect.getdoc(object),
            force_inline=force_inline,
        )

    @_encode.overload(lambda object: is_namedtuple_instance(object))
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
        if depth_limit is not None and depth_limit <= 0:
            return

        return self._encode_object(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: is_type(object))
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
        if depth_limit is not None and depth_limit <= 0:
            return

        if force_inline:
            # check for generic types
            if hasattr(object, "__origin__") and object.__origin__ is not None:
                origin_str = self._encode_type(object.__origin__, format="inline")
                args_str = [
                    self._encode_type(arg, format="inline") for arg in object.__args__
                ]
                return f"{origin_str}[{', '.join(args_str)}]"
            # if not generic, just return the name
            else:
                if object in builtins.__dict__.values():
                    # builtins don't need to be qualified
                    return object.__name__
                return (
                    getattr(object, "__qualname__")
                    or getattr(object, "__name__")
                    or repr(object)
                )

        # otherwise, render the class like a normal blocked structure
        keys = set(get_keys(object, visibility=visibility))
        attribute_keys = {
            k
            for k in keys
            if not inspect.isfunction(getattr(object, k))
            and not inspect.ismodule(getattr(object, k))
            and not inspect.isclass(getattr(object, k))
        }

        attribute_tuples = [
            (
                self._encode(
                    k,
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
                self._encode(
                    object.__annotations__.get(k, None),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
                self._encode(
                    getattr(object, k),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=True,
                    **kwargs,
                ),
            )
            for k in attribute_keys
        ]
        other_tuples = [
            (
                None,
                None,
                self._encode(
                    getattr(object, k),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    inherited_members=inherited_members,
                    force_inline=force_inline,
                    **kwargs,
                ),
            )
            for k in keys - attribute_keys
        ]

        # My apolagies, this approach does not distinguish between lambdas v. block functions
        # and variables with a type value v. block-defined classes

        def get_name(object):
            if object in builtins.__dict__.values():
                # builtins don't need to be qualified
                return object.__name__
            return (
                getattr(object, "__qualname__")
                or getattr(object, "__name__")
                or repr(object)
            )

        header = f"class {get_name(object)}"
        bases = object.__bases__
        bases.remove(object)
        if bases:
            header += f"({', '.join(get_name(base) for base in bases)})"

        return self._render_composite(
            header,
            tuples=attribute_tuples + other_tuples,
            docstring=inspect.getdoc(object),
            force_inline=False,
        )

    @_encode.overload(lambda object: is_pydantic_model_type(object))
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
        if depth_limit is not None and depth_limit <= 0:
            return

        return self._encode_type(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: is_namedtuple_type(object))
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
        if depth_limit is not None and depth_limit <= 0:
            return

        if force_inline:
            return (
                getattr(object, "__qualname__")
                or getattr(object, "__name__")
                or repr(object)
            )

        fields = object._fields

        # FIXME: we don't currently support inherited_members for namedtuple types
        # as there is no way to determine whether a fied declared on a base was not
        # overriden on the child

        field_tuples = []
        for field in fields:
            name = field
            annotation = object.__annotations__.get(field, None)
            default = object._field_defaults.get(field, None)
            encoded_annotation = self._encode(
                annotation,
                depth_limit=depth_limit - 1,
                instructions=instructions,
                force_inline=True,
                visibility=visibility,
                **kwargs,
            )
            encoded_default = self._encode(
                default,
                depth_limit=depth_limit - 1,
                instructions=instructions,
                force_inline=True,
                visibility=visibility,
                **kwargs,
            )
            field_tuples.append(
                (
                    name,
                    encoded_annotation,
                    encoded_default,
                )
            )

        namedtuple_cls_name = (
            getattr(object, "__qualname__")
            or getattr(object, "__name__")
            or repr(object)
        )
        return self._render_composite(
            namedtuple_cls_name,
            tuples=field_tuples,
            docstring=inspect.getdoc(object),
            force_inline=True,
        )

    @_encode.overload(lambda object: isinstance(object, ModuleType))
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
        if depth_limit is not None and depth_limit <= 0:
            return

        if force_inline:
            return (
                getattr(object, "__qualname__")
                or getattr(object, "__name__")
                or repr(object)
            )

        keys = list(get_keys(object, visibility=visibility))
        variable_keys = {
            k
            for k in keys
            if not inspect.isfunction(getattr(object, k))
            and not inspect.ismodule(getattr(object, k))
            and not inspect.isclass(getattr(object, k))
        }
        variable_tuples: list[tuple[str, str, str]] = [
            (
                k,
                self._encode(
                    object.__annotations__.get(k, None),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    force_inline=False,
                    **kwargs,
                ),
                self._encode(
                    getattr(object, k),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    force_inline=False,
                    **kwargs,
                ),
            )
            for k in variable_keys
        ]
        other_tuples: list[tuple[str, str, str]] = [
            (
                None,
                None,
                self._encode(
                    getattr(object, k),
                    depth_limit=depth_limit - 1,
                    instructions=instructions,
                    visibility=visibility,
                    force_inline=True,
                    **kwargs,
                ),
            )
            for k in keys - variable_keys
        ]

        # My apolagies, this approach does not distinguish between lambdas v. block functions
        # and variables with a type value v. block-defined classes

        module_name = (
            getattr(object, "__qualname__")
            or getattr(object, "__name__")
            or repr(object)
        )
        return self._render_composite(
            f"module {module_name}",
            variable_tuples + other_tuples,
            docstring=inspect.getdoc(object),
            force_inline=False,
        )

    @_encode.overload(lambda object: object is None)
    def _encode_none(
        self,
        object: None,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return None

    @_encode.overload(lambda object: isinstance(object, bool))
    def _encode_bool(
        self,
        object: bool,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return str(object)

    @_encode.overload(lambda object: isinstance(object, int))
    def _encode_int(
        self,
        object: int,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return self.p.number_to_words(object)

    @_encode.overload(lambda object: isinstance(object, float))
    def _encode_float(
        self,
        object: float,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        precision_threshold = 1

        decimal_part = object - int(object)
        if decimal_part * 10**precision_threshold != int(
            decimal_part * 10**precision_threshold
        ):
            return self.p.number_to_words(object)
        else:
            return str(object)

    @_encode.overload(lambda object: isinstance(object, complex))
    def _encode_complex(
        self,
        object: complex,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return str(object).replace("j", "i")

    @_encode.overload(lambda object: isinstance(object, str))
    def _encode_str(
        self,
        object: str,
        /,
        depth_limit: int | None = None,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return object

    @_encode.overload(lambda object: isinstance(object, bytes))
    def _encode_bytes(
        self,
        object: bytes,
        /,
        depth_limit: int | None = None,
        bytes_per_group=4,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        result = ""
        for i in range(0, len(object), bytes_per_group):
            group = object[i : i + bytes_per_group]
            result += "".join(f"{byte:02x}" for byte in group) + " "
        if len(object) % bytes_per_group != 0:  # handle remainder bytes
            remainder = object[(len(object) // bytes_per_group) * bytes_per_group :]
            result += "".join(f"{byte:02x}" for byte in remainder)
        return result.strip()

    @_encode.overload(lambda object: isinstance(object, Iterable))
    def _encode_iterable(
        self,
        object: Iterable,
        /,
        depth_limit: int | None = None,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return self._encode_seq(
            list(object),
            depth_limit=depth_limit - 1,
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: typingx.isinstance(object, Sequence[T]))
    def _encode_seq(
        self,
        object: Sequence,
        /,
        depth_limit: int | None = None,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        items_enc = [
            self._encode(item, depth_limit=depth_limit - 1, force_inline=True, **kwargs)
            for item in object
        ]

        if force_inline:
            return f"[{', '.join(items_enc)}]"
        else:
            return Template(r"{%for item in items%}- {{item}}\n{%endfor%}]").render(
                items=items_enc
            )

    @_encode.overload(lambda object: typingx.isinstance(object, Set[T]))
    def _encode_set(
        self,
        object: set,
        /,
        depth_limit: int | None = None,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        return self._encode_seq(
            object,
            depth_limit=depth_limit,  # don't decrement since this is only a horizontal call
            force_inline=force_inline,
            **kwargs,
        )

    @_encode.overload(lambda object: typingx.isinstance(object, Mapping[Any, T]))
    def _encode_map(
        self,
        object: Mapping,
        /,
        depth_limit: int | None = None,
        force_inline: bool = False,
        **kwargs,
    ) -> R:
        if depth_limit is not None and depth_limit <= 0:
            return

        encoded_items = [
            (
                self._encode(
                    k, depth_limit=depth_limit - 1, force_inline=True, **kwargs
                ),
                self._encode(
                    v, depth_limit=depth_limit - 1, force_inline=True, **kwargs
                ),
            )
            for k, v in object.items()
        ]

        if force_inline:
            return f"{{{', '.join(f'{k}: {v}' for k, v in encoded_items)}}}"
        else:
            return Template(r"{%for k, v in items%}{{k}}: {{v}}{%endfor%}").render(
                items=encoded_items
            )

    @_encode.overload(lambda object: isinstance(object, Ellipsis))
    def _encode_ellipsis(
        self,
        object: object,
        /,
        **kwargs,
    ) -> R:
        return "..."

    @_encode.overload(lambda object: isinstance(object, slice))
    def _encode_slice(
        self,
        object: object,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        **kwargs,
    ) -> R:
        start_enc = self.encode(
            object.start,
            depth_limit=depth_limit - 1,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=True,
            **kwargs,
        )
        stop_enc = self.encode(
            object.stop,
            depth_limit=depth_limit - 1,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=True,
            **kwargs,
        )
        step_enc = self.encode(
            object.step,
            depth_limit=depth_limit - 1,
            instructions=instructions,
            visibility=visibility,
            inherited_members=inherited_members,
            force_inline=True,
            **kwargs,
        )
        match start_enc, stop_enc, step_enc:
            case None, None, None:
                return ":"
            case None, None, _:
                return f"::{step_enc}"
            case None, _, None:
                return f":{stop_enc}"
            case None, _, _:
                return f":{stop_enc}:{step_enc}"
            case _, None, None:
                return f"{start_enc}:"
            case _, None, _:
                return f"{start_enc}::{step_enc}"
            case _, _, None:
                return f"{start_enc}:{stop_enc}"
            case _, _, _:
                return f"{start_enc}:{stop_enc}:{step_enc}"

    @_encode.overload(lambda object: is_lambda(object))
    def _encode_lambda(
        self,
        object: LambdaType,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        **kwargs,
    ) -> R:
        return self._encode_function(
            object,
            depth_limit=depth_limit,
            instructions=instructions,
            fn_start_keyword="lambda",
            force_inline=True,
            **kwargs,
        )

    INDENTATION = 4 * " "

    def _render_block(self, header, body) -> str:
        return f"{header}:\n{indent(body, self.INDENTATION)}"

    def _render_invocation(
        self, fn: str or Callable, /, args=None, kwargs=None, result=None
    ) -> str:
        fn_str = fn.__name__ if callable(fn) else fn

        args = args or []
        kwargs = kwargs or {}

        args_str = [self.encode(arg, force_inline=True) for arg in args]
        kwargs_str = [
            f"{k}={self.encode(v, force_inline=True)}" for k, v in kwargs.items()
        ]

        result_str = (
            f" -> {self.encode(result, force_inline=True)}"
            if result is not None
            else ""
        )

        if args_str and kwargs_str:
            return f"{fn_str}({args_str}, {kwargs_str}){result_str}"
        elif args_str:
            return f"{fn_str}({args_str}){result_str}"
        elif kwargs_str:
            return f"{fn_str}({kwargs_str}){result_str}"
        else:
            return f"{fn_str}(){result_str}"

    def _render_field(field: FieldInfo) -> str:
        items = {
            k: getattr(field, k, None)
            for k in dir(field)
            if not k.startswith("_") and getattr(field, k) != getattr(Field, k)
        }
        return f"Field({', '.join(f'{k}={v}' for k, v in items.items())})"

    def _render_composite(
        self,
        header,
        tuples: list[tuple[str, str, str]],
        docstring: str = None,
        force_inline: bool = False,
    ) -> str:
        items_str = []
        for name, annotation, value in tuples:
            if not name:
                # this happens for methods
                items_str.append(value)
            match annotation, value:
                case None, None:
                    items_str.append(name)
                case None, value:
                    items_str.append(f"{name}={value}")
                case annotation, None:
                    items_str.append(f"{name}: {annotation}")
                case annotation, value:
                    items_str.append(f"{name}: {annotation}={value}")

        if force_inline:
            return f"{header}({', '.join(items_str)})"
        else:
            body = "\n".join(items_str)
            if docstring:
                body = f'"""{docstring}"""\n' + body
            return self._render_block(header, body)
