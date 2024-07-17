"""This module is for making conventional data structures for the Tensacode Intermediate Representation (TCIR)."""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from pathlib import Path
import types
from typing import (
    ClassVar,
    Any,
    Iterator,
    Literal,
    Callable,
    Self,
    TypeVar,
    Union,
    Optional,
    Dict,
    Tuple,
    cast,
)

from pydantic import BaseModel, Field, validator
from typingx import issubclassx, isinstancex
import inspect_mate_pp
import inspect
import stringcase

from tensacode.internal.tcir.main import TCIRBase


class TCIRValue(TCIRBase, ABC):
    # immutable: bool = False

    @classmethod
    def register_custom(
        cls, model_key: str, value: Any, *, name: str = None, **class_dict_extras
    ) -> type[Self]:
        return type(
            name or stringcase.camelcase(model_key),
            (cls._CustomRegisteredTCIRValue, cls),
            {
                "model_key": model_key,
                "_python_value": value,
                **class_dict_extras,
            },
        )

    class _CustomRegisteredTCIRValue:
        # leaving this in the class body so that subclasses can override it
        model_key: ClassVar[str]
        _python_value: ClassVar[Any]

        @classmethod
        def can_parse_python(cls, value: Any, *, depth: int = 4) -> bool:
            return value.get("model_key", None) == cls.model_key

        @classmethod
        def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> Self:
            assert value == self._python_value, f"{value} != {self._python_value}"
            return cls()

        def to_python(self) -> Any:
            return self._python_value


class TCIRAtomicValue(TCIRValue, ABC):
    _python_type: ClassVar[type]
    _python_value: Any

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return isinstance(value, cls._python_type)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        return cls(_atomic_python_value=value)

    def to_python(self) -> Any:
        return self._python_value


class TCIRType(TCIRValue, ABC):
    name: str
    _python_type_type: ClassVar[type[Any]] = type
    _python_type_value: type[Any]

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return issubclassx(value, cls._python_type_type)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        if depth <= 0:
            return None
        return cls(_python_type_value=value, **extra_kwargs)

    def to_python(self) -> type[Any]:
        # override this if you can, otherwise the class won't be able to deserialize
        return self._python_type_value


class TCIRUnionType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "union_type"
    _python_type_type: ClassVar[type[Any]] = Union
    union_of_types: tuple[TCIRType, ...]

    @validator("_python_type_value", always=True)
    def init_python_type_value(cls, v, values):
        if v is None:
            return Union[tuple(t.to_python() for t in values.get("union_of_types", ()))]
        return v

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        if hasattr(value, "__args__") and isinstance(value.__args__, tuple):
            union_of_types = tuple(TCIRType.from_python(t) for t in value.__args__)
            return super().from_python(
                value=value,
                depth=depth,
                union_of_types=union_of_types,
                **extra_kwargs,
            )
        else:
            return super().from_python(
                value=value,
                depth=depth,
                union_of_types=(),
                **extra_kwargs,
            )


class TCIRProductType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "product_type"
    _python_type_type: ClassVar[type[Any]] = type
    product_of_types: tuple[TCIRType, ...]

    @validator("_python_type_value", always=True)
    def init_python_type_value(cls, v, values):
        if v is None:
            return type(
                values.get("name", "ProductType"),
                tuple(t.to_python() for t in values.get("product_of_types", ())),
            )
        return v

    @classmethod
    def can_parse_python(cls, value: TCIRBase) -> bool:
        bases_without_object = [b for b in value.__bases__ if b != object]
        return issubclassx(value, type) and (len(bases_without_object) > 1)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        bases = value.__bases__
        if len(bases) == 1 and bases[0] == object:
            bases = ()
        return super().from_python(
            value=value,
            depth=depth,
            product_of_types=tuple(TCIRType.from_python(t) for t in bases),
            **extra_kwargs,
        )


class TCIROptionalType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "optional_type"
    _python_type_type: ClassVar[type[Any]] = Optional
    optional_for_type: TCIRType

    @validator("_python_type_value", always=True)
    def init_python_type_value(cls, v, values):
        if v is None:
            return Optional[values.get("optional_for_type", Any).to_python()]
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return issubclassx(value, cls._python_type_type)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        return super().from_python(
            value=value,
            depth=depth,
            optional_for_type=TCIRType.from_python(value.__args__[0]),
            **extra_kwargs,
        )


class TCIREnumType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "enum"
    _python_type_type: ClassVar[type[Any]] = Enum
    options: dict[str, TCIRValue]

    @validator("_python_type_value", always=True)
    def init_python_type_value(cls, v, values):
        if v is None:
            return Enum(
                values.get("name", "EnumType"),
                {
                    name: value.to_python()
                    for name, value in values.get("options", {}).items()
                },
            )
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return issubclassx(value, cls._python_type_type)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        return super().from_python(
            value=value,
            depth=depth,
            options={entry.name: TCIRValue.from_python(entry.value) for entry in value},
            **extra_kwargs,
        )


class TCIRLiteralType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "literal"
    _python_type_type: ClassVar[type[Any]] = Literal
    value: TCIRValue

    @validator("_python_type_value", always=True)
    def init_python_type_value(cls, v, values):
        if v is None:
            return Literal[values.get("value", Any).to_python()]
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return issubclassx(value, cls._python_type_type)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        return super().from_python(
            value=value,
            depth=depth,
            value=TCIRValue.from_python(value.__args__[0]),
            **extra_kwargs,
        )


class TCIRLiteralSetType(TCIRLiteralType):
    model_key: ClassVar[OBJECT_TYPES] = "literal_set"
    _python_type_type: ClassVar[type[Any]] = Literal
    options: tuple[TCIRAtomicValue, ...]
    _PYTHON_PARSING_PRIORITY: ClassVar[int] = (
        TCIRLiteralType._PYTHON_PARSING_PRIORITY + 1
    )

    @validator("_python_type_value", always=True)
    def init_python_type_value(cls, v, values):
        if v is None:
            return Literal[tuple(t.to_python() for t in values.get("options", (Any,)))]
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return issubclassx(value, cls._python_type_type) and len(value.__args__) > 1

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        return super().from_python(
            value=value,
            depth=depth,
            members=tuple(TCIRAtomicValue.from_python(arg) for arg in value.__args__),
            **extra_kwargs,
        )


import inspect
import annotated_types


# class TCIRScope(TCIRBase, ABC):

#     name: str
#     members: dict[str, TCIRBase]
#     annotations: dict[str, TCIRType]

#     @classmethod
#     def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
#         return super().from_python(
#             value=value,
#             depth=depth,
#             name=value.__name__,
#             members={k: TCIRBase.from_python(getattr(value, k)) for k in dir(value)},
#             annotations={
#                 k: TCIRType.from_python(v)
#                 for k, v in inspect.get_annotations(value).items()
#             },
#             **extra_kwargs,
#         )


class TCIRClass(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "class"
    # because all types will match our filter, so we have to make sure this is the last one to be parsed
    _PYTHON_PARSING_PRIORITY: ClassVar[int] = -100
    _python_type_type: ClassVar[type[Any]] = type

    name: str
    members: dict[str, TCIRBase]
    bases: tuple[TCIRType, ...]

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return issubclassx(value, cls._python_type_type)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        return super().from_python(
            value=value,
            depth=depth,
            members={k: TCIRBase.from_python(getattr(value, k)) for k in dir(value)},
            annotations={
                k: TCIRType.from_python(v)
                for k, v in inspect.get_annotations(value).items()
            },
            **extra_kwargs,
        )


class TCIRIsNumeric(TCIRAtomicValue, ABC):
    pass


class TCIRCompositeValue(TCIRValue, ABC):
    pass


class TCIRInteger(TCIRIsNumeric, TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "integer"
    _atomic_python_type: ClassVar[type] = int
    _atomic_python_value: int


class TCIRFloat(TCIRIsNumeric, TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "float"
    _atomic_python_type: ClassVar[type] = float
    _atomic_python_value: float


from typing import ClassVar
from tensacode.internal.utils.misc import Complex


class TCIRComplexNumber(TCIRIsNumeric, TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "complex_number"
    _atomic_python_type: ClassVar[type] = complex
    _atomic_python_value: Complex


class TCIRBoolean(TCIRAtomicValue):
    model_key: ClassVar[OBJECT_TYPES] = "boolean"
    _atomic_python_type: ClassVar[type] = bool
    _atomic_python_value: bool


class TCIRString(TCIRAtomicValue, TCIRIsIterable):
    model_key: ClassVar[OBJECT_TYPES] = "string"
    _atomic_python_type: ClassVar[type] = str
    _atomic_python_value: str


class TCIRList(TCIRCompositeValue):
    model_key: ClassVar[OBJECT_TYPES] = "list"
    element_types: list[TCIRType]
    values: list[TCIRValue]
    _python_value: list[Any]

    @validator("_python_value", always=True)
    def init_python_value(cls, v, values):
        if v is None:
            return [v.to_python() for v in values.get("values", [])]
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return isinstance(value, list)

    @classmethod
    def from_python(cls, value: list, *, depth: int = 4, **extra_kwargs) -> TCIRList:
        items = value
        return cls(
            element_type=(
                items.__annotations__[0]
                if items.__annotations__
                else TCIRType.from_python(items[0]) if len(items) > 0 else None
            ),
            values=[TCIRValue.from_python(v) for v in items],
            **extra_kwargs,
        )

    def to_python(self) -> Any:
        return self._python_value


class TCIRTuple(TCIRCompositeValue):
    model_key: ClassVar[OBJECT_TYPES] = "list"
    element_types: tuple[TCIRType, ...]
    values: tuple[TCIRValue, ...]
    _python_value: tuple[Any, ...]

    @validator("_python_value", always=True)
    def init_python_value(cls, v, values):
        if v is None:
            return tuple(v.to_python() for v in values.get("values", []))
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return isinstance(value, tuple)

    @classmethod
    def from_python(cls, value: tuple, *, depth: int = 4, **extra_kwargs) -> TCIRTuple:
        items = value
        return cls(
            element_types=tuple(
                items.__annotations__
                if items.__annotations__
                else (
                    [TCIRType.from_python(v) for v in items] if len(items) > 0 else None
                )
            ),
            values=tuple(TCIRValue.from_python(v) for v in items),
            **extra_kwargs,
        )

    def to_python(self) -> Any:
        return self._python_value


class TCIRDict(TCIRCompositeValue):
    model_key: ClassVar[OBJECT_TYPES] = "dict"
    key_type: TCIRType
    value_type: TCIRType
    items: dict[TCIRValue, TCIRValue]
    _python_value: dict[Any, Any]

    @validator("_python_value", always=True)
    def init_python_value(cls, v, values):
        if v is None:
            return {
                k.to_python(): v.to_python() for k, v in values.get("items", {}).items()
            }
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return isinstance(value, (dict, Mapping, MappingProxyType))

    @classmethod
    def from_python(cls, value: dict, *, depth: int = 4, **extra_kwargs) -> TCIRDict:
        items = value
        return cls(
            key_type=(
                TCIRType.from_python(next(iter(items.keys()))) if items else None
            ),
            value_type=(
                TCIRType.from_python(next(iter(items.values()))) if items else None
            ),
            items={
                TCIRValue.from_python(k): TCIRValue.from_python(v)
                for k, v in items.items()
            },
            **extra_kwargs,
        )

    def to_python(self) -> Any:
        return self._python_value


class TCIRSet(TCIRCompositeValue):
    model_key: ClassVar[OBJECT_TYPES] = "set"
    element_type: TCIRType
    elements: set[TCIRValue]
    _python_value: set[Any]

    @validator("_python_value", always=True)
    def init_python_value(cls, v, values):
        if v is None:
            return {e.to_python() for e in values.get("elements", set())}
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return isinstance(value, (set, frozenset))

    @classmethod
    def from_python(cls, value: set, *, depth: int = 4, **extra_kwargs) -> TCIRSet:
        elements = value
        return cls(
            element_type=(
                TCIRType.from_python(next(iter(elements))) if elements else None
            ),
            elements={TCIRValue.from_python(e) for e in elements},
            **extra_kwargs,
        )

    def to_python(self) -> Any:
        return self._python_value


class TCIRModule(TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "module"
    name: str
    members: dict[str, TCIRBase]
    annotations: dict[str, TCIRType]

    @validator("_python_value", always=True)
    def init_python_value(cls, v, values):
        if v is None:
            try:
                return __import__(values.get("name", ""))
            except ImportError:
                # Generate a module on the fly
                module_name = values.get("name", "")
                module = types.ModuleType(module_name)
                for member_name, member_value in values.get("members", {}).items():
                    setattr(module, member_name, member_value.to_python())
                for annotation_name, annotation_type in values.get(
                    "annotations", {}
                ).items():
                    module.__annotations__[annotation_name] = (
                        annotation_type.to_python()
                    )
                return module
        return v

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return inspect.ismodule(value)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        return cls(
            name=value.__name__,
            members={k: TCIRBase.from_python(getattr(value, k)) for k in dir(value)},
            annotations={
                k: TCIRType.from_python(v)
                for k, v in inspect.get_annotations(value).items()
            },
            **extra_kwargs,
        )

    def to_python(self) -> Any:
        return self._python_value


class TCIRInstance(TCIRValue, TCIRScope):
    model_key: ClassVar[OBJECT_TYPES] = "instance"
    type: TCIRType  # use the actual `Class` object to refer to the instance
    _python_value: Any
    # all objects are instances of `object`, so we make sure this is the last one to be parsed
    _PYTHON_PARSING_PRIORITY: ClassVar[int] = -100

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return (
            isinstance(value, object)
            and not inspect.ismodule(value)
            and not issubclassx(value, type)
        )

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        return cls(
            type=TCIRType.from_python(type(value)),
            members={k: TCIRBase.from_python(getattr(value, k)) for k in dir(value)},
            annotations={
                k: TCIRType.from_python(v)
                for k, v in getattr(value, "__annotations__", {}).items()
            },
            **extra_kwargs,
        )

    def to_python(self) -> Any:
        return self._python_value

    @validator("_python_value", always=True)
    def init_python_value(cls, v, values):
        if v is None:
            instance_type = values.get("type").to_python()
            instance = object.__new__(instance_type)
            for member_name, member_value in values.get("members", {}).items():
                setattr(instance, member_name, member_value.to_python())
            instance.__annotations__ = {
                k: v.to_python() for k, v in values.get("annotations", {}).items()
            }
            return instance
        return v


# class TCIRIterator(TCIRIsIterable, TCIRValue, NotSerializable):
#     model_key: ClassVar[OBJECT_TYPES] = "iterator"
#     yield_type: TCIRType


# class TCIRStream(TCIRIsIterable, TCIRValue, NotSerializable):
#     model_key: ClassVar[OBJECT_TYPES] = "stream"
#     yield_type: TCIRType


class TCIRFile(TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "file"
    path: Path
    file_mode: Literal["r", "w", "a", "rb", "wb", "ab"]
    encoding: str | None
    errors: str | None


from tensacode.internal.types.object_types import OBJECT_TYPES


class TCIRFunction(TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "function"

    posonly_arg_types: list[TCIRType]
    arg_types: list[TCIRType]
    vararg_type: TCIRType | None
    kwonly_arg_types: dict[str, TCIRType]
    kwarg_type: TCIRType | None
    return_type: TCIRType

    code: TCIRCode


class TCIRCoroutine(TCIRFunction, TCIRIsIterable):
    model_key: ClassVar[OBJECT_TYPES] = "coroutine"
    send_type: TCIRType
    receive_type: TCIRType


class TCIRCode(TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "code"
    statements: list[TCIRStatement]


class TCIRMethod(TCIRFunction, TCIRScopedMember):
    model_key: ClassVar[OBJECT_TYPES] = "method"


class TCIRRange(TCIRCoroutine, TCIRCompositeValue):
    model_key: ClassVar[OBJECT_TYPES] = "range"
    start: TCIRValue
    stop: TCIRValue
    step: TCIRValue


class TCIRSlice(TCIRCompositeValue):
    model_key: ClassVar[OBJECT_TYPES] = "slice"
    start: int | None
    stop: int | None
    step: int | None


TCIREllipsis = TCIRValue.register_custom("TCIREllipsis", exact_match=...)

TCIRSingleIndex = TCIRIsHashable | TCIRSlice | TCIREllipsis
TCIRIndex = TCIRSingleIndex | list[TCIRSingleIndex]


class TCIRTensor(TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "tensor"
    shape: list[int] | dict[str, int]
    dtype: str | TCIRType
    data: typing.Any


class TCIRBytes(TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "bytes"
    length: int | None
    type = BytesType
    bytes_value: bytes


class TCIRNone(TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "none"
    type = NoneType
