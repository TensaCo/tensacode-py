from __future__ import annotations
from abc import ABC
from functools import cached_property
from pathlib import Path
import types
from typing import ClassVar, Literal
import typing
from tensacode.internal.tcir.tcir_main import (
    Class,
    DeclarationScope,
    IsImmutable,
    IsMutable,
    NotSerializable,
    Type,
    Value,
    Primitive,
    Composite,
    IsHashable,
    IsIterable,
    IsNumeric,
    IsCallable,
)


class IntegerType(Type):
    model_key: ClassVar[str] = "integer"


class Integer(IsNumeric, Value):
    type = IntegerType
    value: int


class FloatType:
    model_key: ClassVar[str] = "float"


class Float(IsNumeric, Value):
    type = FloatType
    value: float


class ComplexNumberType(Type):
    model_key: ClassVar[str] = "complex_number"


class ComplexNumber(IsNumeric, Value):
    type = ComplexNumberType
    real_value: float
    imaginary_value: float


class BooleanType(Type):
    model_key: ClassVar[str] = "boolean"


class Boolean(Primitive):
    type = BooleanType
    value: bool


class CharType(Type):
    model_key: ClassVar[str] = "char"


class Char(Primitive):
    type = CharType
    value: str


class StringType(Type):
    model_key: ClassVar[str] = "string"


class String(Primitive, IsIterable):
    type = StringType
    value: str


class MutableListType(Type):
    model_key: ClassVar[str] = "mutable_list"


class MutableList(Composite, IsMutable):
    type = MutableListType
    element_type: Type | list[Type] = Value
    value: list[Value]


class ImmutableListType(Type):
    model_key: ClassVar[str] = "immutable_list"


class ImmutableList(Composite, IsImmutable):
    type = ImmutableListType
    element_type: Type | list[Type] = Value
    value: list[Value]


class MutableDictType(Type):
    model_key: ClassVar[str] = "mutable_dict"


class MutableDict(Composite, IsMutable):
    type = MutableDictType
    key_type: Type
    value_type: Type
    value: dict[Value, Value]


class ImmutableDictType(Type):
    model_key: ClassVar[str] = "immutable_dict"


class ImmutableDict(Composite, IsImmutable):
    type = ImmutableDictType
    key_type: Type
    value_type: Type
    value: dict[Value, Value]


class MutableSetType(Type):
    model_key: ClassVar[str] = "mutable_set"


class MutableSet(Composite, IsMutable):
    type = MutableSetType
    element_type: Type
    value: set[Value]


class ImmutableSetType(Type):
    model_key: ClassVar[str] = "immutable_set"


class ImmutableSet(Composite, IsImmutable):
    type = ImmutableSetType
    element_type: Type
    value: set[Value]


class ModuleType(Type):
    model_key: ClassVar[str] = "module"


class Module(Value, DeclarationScope):
    type = ModuleType
    module_name: str


class Instance(Value, DeclarationScope):
    type: Type


class IteratorType(Type):
    model_key: ClassVar[str] = "iterator"


class Iterator(IsIterable, Value, NotSerializable):
    type = IteratorType
    yield_type: Type


class StreamType(Type):
    model_key: ClassVar[str] = "stream"


class Stream(IsIterable, Value, NotSerializable):
    type = StreamType
    yield_type: Type


class FileType(Type):
    model_key: ClassVar[str] = "file"


class File(Value):
    type = FileType
    path: Path
    file_mode: Literal["r", "w", "a", "rb", "wb", "ab"]
    encoding: str | None
    errors: str | None


class FunctionType(Type):
    model_key: ClassVar[str] = "function"
    posonly_arg_types: list[Type]
    arg_types: list[Type]
    vararg_type: Type | None
    kwonly_arg_types: dict[str, Type]
    kwarg_type: Type | None
    return_type: Type


class Function(Value):
    type = FunctionType
    code: Code


class CoroutineType(FunctionType):
    model_key: ClassVar[str] = "coroutine"
    send_type: Type
    receive_type: Type


class Coroutine(Function, IsIterable):
    type = CoroutineType


class LambdaType(FunctionType):
    model_key: ClassVar[str] = "lambda"


class Lambda(Function):
    type = LambdaType


class CodeType(Type):
    model_key: ClassVar[str] = "code"


class Code(Value):
    type = CodeType


class MethodType(FunctionType):
    model_key: ClassVar[str] = "method"


class Method(Function):
    type = MethodType
    owner_class: Class


class InstanceMethodType(MethodType):
    model_key: ClassVar[str] = "instance_method"


class InstanceMethod(Method):
    type = InstanceMethodType


class StaticMethodType(MethodType):
    model_key: ClassVar[str] = "static_method"


class StaticMethod(Method):
    type = StaticMethodType


class ClassMethodType(MethodType):
    model_key: ClassVar[str] = "class_method"


class ClassMethod(Method):
    type = ClassMethodType


class RangeType(Type):
    model_key: ClassVar[str] = "range"


class Range(Coroutine):
    type = RangeType
    start: Value
    stop: Value
    step: Value


class SliceType(Type):
    model_key: ClassVar[str] = "slice"


class Slice(Value):
    type = SliceType
    start: int | None
    stop: int | None
    step: int | None


class EllipsisType(Type):
    model_key: ClassVar[str] = "ellipsis"


class Ellipsis(Value):
    type = EllipsisType


SingleIndex = IsHashable | Slice | Ellipsis
Index = SingleIndex | list[SingleIndex]


class TensorType(Type):
    model_key: ClassVar[str] = "tensor"
    shape: list[int] | dict[str, int]
    dtype: str | Type


class Tensor(Value):
    type = TensorType
    data: typing.Any


class BytesType(Type):
    model_key: ClassVar[str] = "bytes"
    length: int | None


class Bytes(Value):
    type = BytesType
    bytes_value: bytes


class NoneType(Type):
    model_key: ClassVar[str] = "none"


class NoneValue(Value):
    type = NoneType
