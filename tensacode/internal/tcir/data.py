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

from pydantic import BaseModel
from typingx import issubclassx, isinstancex
import inspect_mate_pp
import inspect
import stringcase

from tensacode.internal.tcir.main import TCIRAny


class TCIRValue(TCIRAny, ABC):
    immutable: bool = False

    @classmethod
    def can_parse_python(cls, value: TCIRAny) -> bool:
        return isinstance(value, cls._native_type)

    @classmethod
    def register(
        cls, model_key: str, value: Any, *, name: str = None, **class_dict_extras
    ) -> type[Self]:
        custom_registered_tcir_value_class = type(
            name or stringcase.camelcase(model_key),
            (cls._CustomRegisteredTCIRValue, cls),
            {
                "model_key": model_key,
                "_python_value": value,
                **class_dict_extras,
            },
        )

        @TCIRAny.from_python.register(lambda v: v == value)
        @classmethod
        def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
            return custom_registered_tcir_value_class()

        setattr(custom_registered_tcir_value_class, "from_python", from_python)
        return custom_registered_tcir_value_class

    class _CustomRegisteredTCIRValue:
        model_key: ClassVar[str]
        _python_value: ClassVar[Any]

        @classmethod
        def can_parse_python(cls, value: TCIRAny, *, depth: int = 4) -> bool:
            return value == cls._python_value

        def to_python(self) -> TCIRAny:
            return self._python_value


class TCIRAtomicValue(TCIRValue, ABC):
    _atomic_python_type: ClassVar[type]
    _atomic_python_value: Any

    @classmethod
    def can_parse_python(cls, value: TCIRAny) -> bool:
        return isinstance(value, cls._atomic_python_type)

    @TCIRAny.from_python.register(lambda cls, value: cls.can_parse_python(value))
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(_atomic_python_value=value)

    def to_python(self) -> TCIRAny:
        return self._atomic_python_value


class TCIRType(TCIRValue, ABC):
    name: str

    @cached_property
    @abstractmethod
    def _native_type(self) -> type: ...

    def issubclass(self, other) -> bool:
        return issubclassx(other, self._native_type)

    @classmethod
    def can_parse_python(cls, value: TCIRAny) -> bool:
        return issubclassx(value, cls._native_type)

    @classmethod
    @abstractmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny: ...

    def to_python(self) -> TCIRAny:
        return self._native_type


class TCIRUnionType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "union_type"
    types: tuple[TCIRType, ...]

    @cached_property
    def native_type(self) -> type:
        return Union[*[t._native_type for t in self.types]]

    @TCIRAny.from_python.register(lambda cls, value: cls.can_parse_python(value))
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        if hasattr(value, "__args__") and isinstance(value.__args__, tuple):
            return cls(types=tuple(TCIRType.from_python(t) for t in value.__args__))
        else:
            return cls(types=())


class TCIRProductType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "product_type"
    types: tuple[TCIRType, ...]

    @cached_property
    def native_type(self) -> type:
        return type(self.name, tuple(t._native_type for t in self.types))

    @TCIRAny.from_python.register(lambda cls, value: cls.can_parse_python(value))
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        bases = value.__bases__
        if len(bases) == 1 and bases[0] == object:
            bases = ()
        return cls(types=tuple(TCIRType.from_python(t) for t in bases))


class TCIROptionalType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "optional_type"
    type: TCIRType

    @cached_property
    def native_type(self) -> type:
        return Optional[self.type._native_type]

    @TCIRAny.from_python.register(lambda cls, value: cls.can_parse_python(value))
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(type=TCIRType.from_python(value.__args__[0]))


class TCIREnumType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "enum"
    members: dict[str, TCIRValue]

    @cached_property
    def native_type(self) -> type:
        return type(self.name, (Enum,), self.members)

    @TCIRAny.from_python.register(lambda cls, value: cls.can_parse_python(value))
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(
            members={m: TCIRValue.from_python(v) for m, v in value.__members__.items()}
        )


class TCIRLiteralType(TCIRType):
    model_key: ClassVar[OBJECT_TYPES] = "literal"
    value: TCIRValue
    
    @cached_property
    def native_type(self) -> type:
        return Literal[self.value._native_type]

    @TCIRAny.from_python.register(lambda cls, value: cls.can_parse_python(value))
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(value=TCIRValue.from_python(value.__args__[0]))

class TCIRLiteralSetType(TCIRLiteralType):
    model_key: ClassVar[OBJECT_TYPES] = "literal_set"
    members: tuple[TCIRAtomicValue, ...]

    @cached_property
    def native_type(self) -> type:
        return Literal[*[member.native_type for member in self.members]]

    @TCIRAny.from_python.register(lambda cls, value: cls.can_parse_python(value))
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4) -> TCIRAny:
        return cls(members=tuple(TCIRAtomicValue.from_python(v) for v in value.__args__))

class TCIRScope(ABC):

    members: dict[str, TCIRScopedMember]

    classes: dict[str, TCIRType]
    functions: dict[str, TCIRFunction]
    variables: dict[str, TCIRVariable]

    @cached_property
    def variable_values(self) -> dict[str, TCIRAny]:
        return {k: v.val for k, v in self.variables.items()}

    @cached_property
    def members(self) -> dict[str, TCIRAny]:
        return {**self.classes, **self.functions, **self.variables}

    @cached_property
    def values(self) -> dict[str, TCIRAny]:
        return {**self.classes, **self.functions, **self.variable_values}

    @classmethod
    def create_from_python(cls, val: TCIRAny, depth: int) -> TCIRValue:
        classes = {
            cls_i.name: cls_i.create_from_python(cls_i, depth - 1)
            for cls_i in inspect_mate_pp.get_classes(val)
        }
        functions = {
            fn_i.name: fn_i.create_from_python(fn_i, depth - 1)
            for fn_i in inspect_mate_pp.get_functions(val)
        }
        variables = {
            var_i.name: var_i.create_from_python(var_i, depth - 1)
            for var_i in inspect_mate_pp.get_variables(val)
        }
        return cls(functions=functions, classes=classes, variables=variables)


class TCIRScopedMember(TCIRAny):
    scope: TCIRScope


class TCIRVariable(TCIRScopedMember, TCIRAny):
    name: str
    val: TCIRValue
    annotations: tuple[TCIRValue, ...]


class TCIRClass(TCIRType, TCIRScope, ABC):
    model_key: ClassVar[OBJECT_TYPES] = "class"

    name: str
    bases: tuple[TCIRType, ...]
    class_variables: dict[str, TCIRClassVariable]
    instance_variables: dict[str, TCIRInstanceVariable]

    @cached_property
    def variables(self) -> dict[str, TCIRClassVariable | TCIRInstanceVariable]:
        return {**self.class_variables, **self.instance_variables}

    @cached_property
    def _native_type(self) -> TCIRType:
        # if self.name in self._REGISTERED_CLASSES:
        #     return self._REGISTERED_CLASSES[self.name]
        return type(self.name, self.bases, self.values)

    @classmethod
    def can_parse_python(cls, val: Any, *, depth: int = 4) -> bool:
        return isinstancex(val, type)

    @classmethod
    def create_from_python(cls, val: type, *, depth: int = 4) -> TCIRValue:
        cls.register(val.__name__, val)

        # name = val.__name__
        # bases: tuple[TCIRType, ...]
        # if cls == type:
        #     bases = []
        # elif cls == object:
        #     bases = []
        # else:
        #     bases = tuple(cls.create_from_python(b) for b in val.__bases__)
        # values = {}

        # classes = {
        #     k: TCIRType.create_from_python(v)
        #     for k, v in val.__dict__.items()
        #     if isinstance(v, type)
        # }
        # functions = inspect_mate_pp.get_all_methods(val)
        # variables = inspect_mate_pp.get_all_attributes(val)
        # return cls(
        #     name=name,
        #     bases=bases,
        #     classes=classes,
        #     functions=functions,
        #     variables=variables,
        # )


class TCIRIsIterable(TCIRValue, ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[TCIRValue]: ...


class TCIRIsNumeric(TCIRAtomicValue, ABC):
    @property
    def number_value(self) -> int | float | complex: ...
    @number_value.setter
    def number_value(self, value: int | float | complex): ...

    @classmethod
    def can_parse_python(cls, val: Any, *, depth: int = 4) -> bool:
        return isinstance(val, cls._native_type)

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRCompositeValue(TCIRValue, ABC):
    pass


class TCIRInteger(TCIRIsNumeric, TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "integer"
    value: int


class TCIRFloat(TCIRIsNumeric, TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "float"
    value: float

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRComplexNumber(TCIRIsNumeric, TCIRValue):
    model_key: ClassVar[OBJECT_TYPES] = "complex_number"
    real_value: float
    imaginary_value: float

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRBoolean(TCIRAtomicValue):
    model_key: ClassVar[OBJECT_TYPES] = "boolean"
    value: bool

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


class TCIRString(TCIRAtomicValue, TCIRIsIterable):
    model_key: ClassVar[OBJECT_TYPES] = "string"
    value: str

    @classmethod
    def create_from_python(cls, val: TCIRAny) -> TCIRValue:
        return super().create_from_python(val)


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
