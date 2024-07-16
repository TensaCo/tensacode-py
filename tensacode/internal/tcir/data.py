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

from pydantic import BaseModel, Field
from typingx import issubclassx, isinstancex
import inspect_mate_pp
import inspect
import stringcase

from tensacode.internal.tcir.main import TCIRAny


class TCIRValue(TCIRAny, ABC):
    # immutable: bool = False

    @classmethod
    def register(
        cls, model_key: str, value: Any, *, name: str = None, **class_dict_extras
    ) -> type[Self]:
        CustomTCIRClass = type(
            name or stringcase.camelcase(model_key),
            (cls._CustomRegisteredTCIRValue, cls),
            {
                "model_key": model_key,
                "_python_value": value,
                **class_dict_extras,
            },
        )

        TCIRAny.from_python.register(
            CustomTCIRClass.can_parse_python,
            priority=cls._PYTHON_PARSING_PRIORITY + 1,
        )(CustomTCIRClass.from_python)

        return CustomTCIRClass

    class _CustomRegisteredTCIRValue:
        model_key: ClassVar[str]
        _python_value: ClassVar[Any]

        @classmethod
        def can_parse_python(cls, value: Any, *, depth: int = 4) -> bool:
            return value.get("model_key", None) == cls.model_key

        @classmethod
        def from_python(cls, value: Any, *, depth: int = 4) -> Self:
            assert value == self._python_value, f"{value} != {self._python_value}"
            return cls()

        def to_python(self) -> TCIRAny:
            return self._python_value

    # @abstractmethod
    # @classmethod
    # def can_parse_python(cls, value: Any) -> bool: ...

    # @polymorphic
    # @classmethod
    # def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
    #     if depth <= 0:
    #         return None
    #     return cls.model_validate(value)

    # @abstractmethod
    # def to_python(self) -> TCIRAny:
    #     raise NotImplementedError()


class TCIRAtomicValue(TCIRValue, ABC):
    _atomic_python_type: ClassVar[type]
    _atomic_python_value: Any

    @classmethod
    def can_parse_python(cls, value: TCIRAny) -> bool:
        return isinstance(value, cls._atomic_python_type)

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(_atomic_python_value=value)

    def to_python(self) -> TCIRAny:
        return self._atomic_python_value


class TCIRType(TCIRValue, ABC):
    name: str
    _atomic_python_supertype: ClassVar[type[Any]]
    _atomic_python_type: type[Any]

    # TODO: left off herer

    @classmethod
    @abstractmethod
    def can_parse_python(cls, value: TCIRAny) -> bool:
        return issubclassx(value, type)

    @classmethod
    @abstractmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny: ...

    def to_python(self) -> type[Any]:
        return type


class TCIRUnionType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "union_type"
    union_of_types: tuple[TCIRType, ...]

    @classmethod
    def can_parse_python(cls, value: TCIRAny) -> bool:
        return issubclassx(value, Union)

    @TCIRAny.from_python.register(
        lambda value, depth: can_parse_python(value, depth=depth)
    )
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        if hasattr(value, "__args__") and isinstance(value.__args__, tuple):
            union_of_types = tuple(TCIRType.from_python(t) for t in value.__args__)
            return TCIRUnionType(union_of_types=union_of_types)
        else:
            return TCIRUnionType(union_of_types=())

    def to_python(self) -> type[Any]:
        return Union[*[t.to_python() for t in self.union_of_types]]


class TCIRProductType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "product_type"
    product_of_types: tuple[TCIRType, ...]

    @classmethod
    def can_parse_python(cls, value: TCIRAny) -> bool:
        bases_without_object = [b for b in value.__bases__ if b != object]
        return issubclassx(value, type) and (len(bases_without_object) > 1)

    @TCIRAny.from_python.register(
        lambda value, depth: can_parse_python(value, depth=depth),
        # lower priority bec we don't want to decompose classes usually
        priority=TCIRClass._PYTHON_PARSING_PRIORITY - 1,
    )
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        bases = value.__bases__
        if len(bases) == 1 and bases[0] == object:
            bases = ()
        return TCIRProductType(
            product_of_types=tuple(TCIRType.from_python(t) for t in bases)
        )

    def to_python(self) -> type[Any]:
        return type(self.name, tuple(t.to_python() for t in self.product_of_types))


class TCIROptionalType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "optional_type"
    optional_for_type: TCIRType

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return issubclassx(value, Optional)

    @TCIRAny.from_python.register(
        lambda value, depth: can_parse_python(value, depth=depth)
    )
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return TCIROptionalType(
            optional_for_type=TCIRType.from_python(value.__args__[0])
        )

    def to_python(self) -> Optional[type[Any]]:
        return Optional[self.optional_for_type.to_python()]


class TCIREnumType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "enum"
    members: dict[str, TCIRValue]

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return issubclass(value, Enum)

    @TCIRAny.from_python.register(
        lambda value, depth: can_parse_python(value, depth=depth)
    )
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return TCIREnumType(
            members={m: TCIRValue.from_python(v) for m, v in value.__members__.items()}
        )

    def to_python(self) -> type[Any]:
        return type(self.name, (Enum,), self.members)


class TCIRLiteralType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "literal"
    value: TCIRValue

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return isinstance(value, Literal)

    @TCIRAny.from_python.register(
        lambda value, depth: can_parse_python(value, depth=depth)
    )
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(value=TCIRValue.from_python(value.__args__[0]))

    def to_python(self) -> type[Any]:
        return Literal[self.value.to_python()]


class TCIRLiteralSetType(TCIRLiteralType):
    model_key: ClassVar[OBJECT_TYPES] = "literal_set"
    members: tuple[TCIRAtomicValue, ...]

    @classmethod
    def can_parse_python(cls, value: Any) -> bool:
        return isinstance(value, Literal) and len(value.__args__) > 1

    @TCIRAny.from_python.register(
        lambda value, depth: TCIRLiteralSetType.can_parse_python(value),
        priority=TCIRLiteralType._PYTHON_PARSING_PRIORITY + 1,
    )
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(
            members=tuple(TCIRAtomicValue.from_python(arg) for arg in value.__args__)
        )

    def to_python(self) -> type[Any]:
        return Literal[tuple(member.to_python() for member in self.members)]


class TCIRScope(TCIRAny, ABC):

    name: str
    members: dict[str, TCIRAny]

    # TODO: stopped off herer

    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(
            name=value.__name__,
            members={k: getattr(value, k) for k in dir(value)},
        )


# class TCIRScopedMember(TCIRAny):
#     scope: TCIRScope


# class TCIRVariable(TCIRScopedMember, TCIRAny):
#     name: str
#     val: TCIRValue
#     annotations: tuple[TCIRValue, ...]


class TCIRClass(TCIRType, TCIRScope, ABC):
    model_key: ClassVar[OBJECT_TYPES] = "class"

    name: str
    members: dict[str, TCIRAny]
    bases: tuple[TCIRType, ...]

    def to_python(self) -> type[Any]:
        return type(self.name, tuple(b.to_python() for b in self.bases), self.members)

    @TCIRAny.from_python.register(lambda cls, value: cls.can_parse_python(value))
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(
            members=tuple(TCIRAtomicValue.from_python(v) for v in value.__args__)
        )


class TCIRIsIterable(TCIRValue, ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[TCIRValue]: ...


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
    element_type: TCIRType | list[TCIRType] = TCIRValue
    value: list[TCIRValue]

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRDict(TCIRCompositeValue):
    model_key: ClassVar[OBJECT_TYPES] = "dict"
    key_type: TCIRType
    value_type: TCIRType
    value: dict[TCIRValue, TCIRValue]

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRSet(TCIRCompositeValue):
    model_key: ClassVar[OBJECT_TYPES] = "set"
    element_type: TCIRType
    value: set[TCIRValue]

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRModule(TCIRValue, TCIRScope):
    model_key: ClassVar[OBJECT_TYPES] = "module"
    qualname: str

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRInstance(TCIRValue, TCIRScope):
    model_key: ClassVar[OBJECT_TYPES] = "instance"
    type: TCIRType  # use the actual `Class` object to refer to the instance

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRIterator(TCIRIsIterable, TCIRValue, NotSerializable):
    model_key: ClassVar[OBJECT_TYPES] = "iterator"
    yield_type: TCIRType


class TCIRStream(TCIRIsIterable, TCIRValue, NotSerializable):
    model_key: ClassVar[OBJECT_TYPES] = "stream"
    yield_type: TCIRType


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


TCIREllipsis = TCIRValue.register("TCIREllipsis", exact_match=...)

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