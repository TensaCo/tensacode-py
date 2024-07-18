from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Callable, Iterable
import inspect
import io
import types
import datetime
import enum
import re
import dataclasses
from types import NoneType, EllipsisType, UnionType
from pydantic import BaseModel
from dataclasses import fields, is_dataclass, MISSING
import types
import weakref
import collections
import array
import ctypes
import decimal
import fractions
import pathlib
import ipaddress
import uuid
from tensacode.internal.utils.functional import polymorphic
from tensacode.internal.tcir.structs import *


@polymorphic
def parse_node(value: Any) -> Node:
    return CompositeValueNode(**value)  # Default case


@parse_node.register(lambda v: v is None)
def parse_none(value: None) -> NoneNode:
    return NoneNode()


@parse_node.register(lambda v: v is Ellipsis)
def parse_ellipsis(value: EllipsisType) -> EllipsisNode:
    return EllipsisNode()


@parse_node.register(lambda v: isinstance(v, bool))
def parse_bool(value: bool) -> BoolNode:
    return BoolNode(value=value)


@parse_node.register(lambda v: isinstance(v, int))
def parse_int(value: int) -> IntNode:
    return IntNode(value=value)


@parse_node.register(lambda v: isinstance(v, float))
def parse_float(value: float) -> FloatNode:
    return FloatNode(value=value)


@parse_node.register(lambda v: isinstance(v, complex))
def parse_complex(value: complex) -> ComplexNumberNode:
    return ComplexNumberNode(value=value)


@parse_node.register(lambda v: isinstance(v, str))
def parse_string(value: str) -> StringNode:
    return StringNode(value=value)


@parse_node.register(lambda v: isinstance(v, bytes))
def parse_bytes(value: bytes) -> BytesNode:
    return BytesNode(value=value)


@parse_node.register(lambda v: isinstance(v, (list, tuple, set, frozenset)))
def parse_sequence(value: Iterable) -> SequenceNode:
    return SequenceNode(items=[parse_node(v) for v in value])


@parse_node.register(lambda v: isinstance(v, dict))
def parse_mapping(value: dict) -> MappingNode:
    return MappingNode(items={parse_node(k): parse_node(v) for k, v in value.items()})


@parse_node.register(lambda v: isinstance(v, tuple) and hasattr(v, "_fields"))
def parse_named_tuple(value: tuple) -> MappingNode:
    return MappingNode(
        items={
            StringNode(value=field): parse_node(getattr(value, field))
            for field in value._fields
        }
    )


@parse_node.register(lambda v: is_dataclass(v))
def parse_dataclass(value: Any) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="dataclass"),
        name=StringNode(value=type(value).__name__),
        **{
            field.name: parse_node(getattr(value, field.name))
            for field in fields(value)
        },
    )


@parse_node.register(lambda v: isinstance(v, BaseModel))
def parse_pydantic_model(value: BaseModel) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="pydantic_model"),
        name=StringNode(value=type(value).__name__),
        **{
            field.name: parse_node(getattr(value, field.name))
            for field in fields(value)
        },
    )


@parse_node.register(lambda v: isinstance(v, types.TracebackType))
def parse_traceback(value: types.TracebackType) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="traceback"),
        tb_frame=parse_node(value.tb_frame),
        tb_lasti=IntNode(value=value.tb_lasti),
        tb_lineno=IntNode(value=value.tb_lineno),
        tb_next=parse_node(value.tb_next) if value.tb_next else NoneNode(),
    )


@parse_node.register(lambda v: isinstance(v, (weakref.ref, weakref.ProxyType)))
def parse_weakref(value: weakref.ref | weakref.ProxyType) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="weakref"),
        referent_type=StringNode(value=type(value()).__name__ if value() else "None"),
    )


@parse_node.register(lambda v: isinstance(v, collections.deque))
def parse_deque(value: collections.deque) -> SequenceNode:
    return SequenceNode(items=[parse_node(item) for item in value])


@parse_node.register(lambda v: isinstance(v, collections.Counter))
def parse_counter(value: collections.Counter) -> MappingNode:
    return MappingNode(
        items={parse_node(k): IntNode(value=v) for k, v in value.items()}
    )


@parse_node.register(lambda v: isinstance(v, collections.OrderedDict))
def parse_ordered_dict(value: collections.OrderedDict) -> MappingNode:
    return MappingNode(items={parse_node(k): parse_node(v) for k, v in value.items()})


@parse_node.register(lambda v: isinstance(v, collections.defaultdict))
def parse_defaultdict(value: collections.defaultdict) -> MappingNode:
    return MappingNode(items={parse_node(k): parse_node(v) for k, v in value.items()})


@parse_node.register(lambda v: isinstance(v, types.MappingProxyType))
def parse_mapping_proxy(value: types.MappingProxyType) -> MappingNode:
    return MappingNode(items={parse_node(k): parse_node(v) for k, v in value.items()})


@parse_node.register(lambda v: isinstance(v, array.array))
def parse_array(value: array.array) -> SequenceNode:
    return SequenceNode(items=[parse_node(item) for item in value])


@parse_node.register(lambda v: isinstance(v, memoryview))
def parse_memoryview(value: memoryview) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="memoryview"),
        format=StringNode(value=value.format),
        itemsize=IntNode(value=value.itemsize),
        ndim=IntNode(value=value.ndim),
        shape=(
            SequenceNode(items=[IntNode(value=dim) for dim in value.shape])
            if value.shape
            else NoneNode()
        ),
    )


@parse_node.register(lambda v: isinstance(v, ctypes._SimpleCData))
def parse_ctypes(value: ctypes._SimpleCData) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="ctypes"),
        ctype=StringNode(value=type(value).__name__),
        value=parse_node(value.value),
    )


@parse_node.register(lambda v: isinstance(v, decimal.Decimal))
def parse_decimal(value: decimal.Decimal) -> FloatNode:
    return FloatNode(value=float(value))


@parse_node.register(lambda v: isinstance(v, fractions.Fraction))
def parse_fraction(value: fractions.Fraction) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="Fraction"),
        numerator=IntNode(value=value.numerator),
        denominator=IntNode(value=value.denominator),
        float=FloatNode(value=float(value)),
    )


@parse_node.register(lambda v: isinstance(v, pathlib.Path))
def parse_path(value: pathlib.Path) -> StringNode:
    return StringNode(value=str(value))


@parse_node.register(
    lambda v: isinstance(v, (ipaddress.IPv4Address, ipaddress.IPv6Address))
)
def parse_ip_address(
    value: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> StringNode:
    return StringNode(value=str(value))


@parse_node.register(lambda v: isinstance(v, uuid.UUID))
def parse_uuid(value: uuid.UUID) -> StringNode:
    return StringNode(value=str(value))


@parse_node.register(lambda v: isinstance(v, types.GeneratorType))
def parse_generator(value: types.GeneratorType) -> SequenceNode:
    return SequenceNode(items=[parse_node(item) for item in value])


@parse_node.register(lambda v: isinstance(v, io.IOBase))
def parse_file(value: io.IOBase) -> FileNode:
    return FileNode(
        name=parse_node(getattr(value, "name", None)),
        mode=parse_node(getattr(value, "mode", None)),
        closed=BoolNode(value=value.closed),
        encoding=parse_node(getattr(value, "encoding", None)),
        path=parse_node(getattr(value, "path", None)),
        lines=SequenceNode(
            value=[StringNode(value=line) for line in value.readlines()]
        ),
    )


@parse_node.register(
    lambda v: isinstance(v, (datetime.datetime, datetime.date, datetime.time))
)
def parse_datetime(
    value: datetime.datetime | datetime.date | datetime.time,
) -> CompositeValueNode:
    if isinstance(value, datetime.datetime):
        return CompositeValueNode(
            type=StringNode(value="datetime"),
            year=IntNode(value=value.year),
            month=IntNode(value=value.month),
            day=IntNode(value=value.day),
            hour=IntNode(value=value.hour),
            minute=IntNode(value=value.minute),
            second=IntNode(value=value.second),
            microsecond=IntNode(value=value.microsecond),
            tzinfo=parse_node(str(value.tzinfo) if value.tzinfo else None),
        )
    elif isinstance(value, datetime.date):
        return CompositeValueNode(
            type=StringNode(value="date"),
            year=IntNode(value=value.year),
            month=IntNode(value=value.month),
            day=IntNode(value=value.day),
        )
    else:  # time
        return CompositeValueNode(
            type=StringNode(value="time"),
            hour=IntNode(value=value.hour),
            minute=IntNode(value=value.minute),
            second=IntNode(value=value.second),
            microsecond=IntNode(value=value.microsecond),
            tzinfo=parse_node(str(value.tzinfo) if value.tzinfo else None),
        )


@parse_node.register(lambda v: isinstance(v, enum.Enum))
def parse_enum(value: enum.Enum) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="enum"),
        enum_class=StringNode(value=value.__class__.__name__),
        name=StringNode(value=value.name),
        value=parse_node(value.value),
    )


@parse_node.register(lambda v: isinstance(v, re.Pattern))
def parse_regex(value: re.Pattern) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="regex"),
        pattern=StringNode(value=value.pattern),
        flags=IntNode(value=value.flags),
    )


@parse_node.register(lambda v: isinstance(v, tuple) and hasattr(v, "_fields"))
def parse_namedtuple(value: tuple) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="namedtuple"),
        name=StringNode(value=type(value).__name__),
        fields=MappingNode(
            value={
                StringNode(value=field): parse_node(getattr(value, field))
                for field in value._fields
            }
        ),
    )


try:
    import numpy as np

    @parse_node.register(lambda v: isinstance(v, np.ndarray))
    def parse_numpy_array(value: np.ndarray) -> CompositeValueNode:
        return CompositeValueNode(
            type=StringNode(value="numpy.ndarray"),
            shape=SequenceNode(items=[IntNode(value=dim) for dim in value.shape]),
            dtype=StringNode(value=str(value.dtype)),
            data=TensorNode(value=value),
        )

except ImportError:
    pass

# Add support for pandas Series and DataFrame if pandas is available
try:
    import pandas as pd

    @parse_node.register(lambda v: isinstance(v, pd.Series))
    def parse_pandas_series(value: pd.Series) -> PandasDataFrameNode:
        return PandasDataFrameNode(
            data=SequenceNode(items=[parse_node(item) for item in value]),
            index=SequenceNode(items=[parse_node(idx) for idx in value.index]),
            columns=SequenceNode(items=[StringNode(value=str(value.name))]),
        )

    @parse_node.register(lambda v: isinstance(v, pd.DataFrame))
    def parse_pandas_dataframe(value: pd.DataFrame) -> PandasDataFrameNode:
        return PandasDataFrameNode(
            data=SequenceNode(
                value=[
                    SequenceNode(items=[parse_node(item) for item in row])
                    for _, row in value.iterrows()
                ]
            ),
            index=SequenceNode(items=[parse_node(idx) for idx in value.index]),
            columns=SequenceNode(
                value=[StringNode(value=col) for col in value.columns]
            ),
        )

except ImportError:
    pass

# Add support for custom types or any other specific types you need to handle


@parse_node.register(lambda v: isinstance(v, type))
def parse_class(value: type) -> ClassNode:
    return ClassNode(
        name=value.__name__,
        parents=[parse_node(base) for base in value.__bases__ if base != object],
        members=[
            parse_node(getattr(value, name))
            for name in dir(value)
            if not name.startswith("__")
        ],
    )


@parse_node.register(lambda v: callable(v))
def parse_function(value: Callable) -> FunctionNode:
    sig = inspect.signature(value)
    return FunctionNode(
        name=value.__name__,
        parameters=[
            parse_parameter(param)
            for param in inspect.signature(value).parameters.values()
        ],
        return_type=(
            parse_node(sig.return_annotation)
            if sig.return_annotation != inspect.Signature.empty
            else NoneNode()
        ),
        body=inspect.getsource(value),
    )


@parse_node.register(lambda v: isinstance(v, inspect.Parameter))
def parse_parameter(value: inspect.Parameter) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="parameter"),
        name=StringNode(value=value.name),
        annotation=parse_node(value.annotation),
    )


@parse_node.register(lambda v: isinstance(v, types.ModuleType))
def parse_module(value: types.ModuleType) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="module"),
        name=StringNode(value=value.__name__),
        file=parse_node(value.__file__),
        members=MappingNode(
            items={
                StringNode(value=name): parse_node(getattr(value, name))
                for name in dir(value)
                if not name.startswith("__")
            }
        ),
    )


@parse_node.register(lambda v: isinstance(v, types.CodeType))
def parse_code_object(value: types.CodeType) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="code_object"),
        co_name=StringNode(value=value.co_name),
        co_filename=StringNode(value=value.co_filename),
        co_firstlineno=IntNode(value=value.co_firstlineno),
    )


@parse_node.register(lambda v: isinstance(v, types.FrameType))
def parse_frame_object(value: types.FrameType) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="frame"),
        f_code=parse_node(value.f_code),
        f_lineno=IntNode(value=value.f_lineno),
        f_locals=parse_node(value.f_locals),
    )


@parse_node.register(lambda v: isinstance(v, (types.MethodType, types.FunctionType)))
def parse_method(value: types.MethodType | types.FunctionType) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(
            value="method" if isinstance(value, types.MethodType) else "function"
        ),
        name=StringNode(value=value.__name__),
        qualname=StringNode(value=value.__qualname__),
        module=StringNode(value=value.__module__),
    )


@parse_node.register(lambda v: isinstance(v, property))
def parse_property(value: property) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="property"),
        fget=parse_node(value.fget),
        fset=parse_node(value.fset),
        fdel=parse_node(value.fdel),
    )


@parse_node.register(lambda v: isinstance(v, slice))
def parse_slice(value: slice) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="slice"),
        start=parse_node(value.start),
        stop=parse_node(value.stop),
        step=parse_node(value.step),
    )


@parse_node.register(lambda v: isinstance(v, range))
def parse_range(value: range) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="range"),
        start=IntNode(value=value.start),
        stop=IntNode(value=value.stop),
        step=IntNode(value=value.step),
    )


@parse_node.register(lambda v: isinstance(v, types.CoroutineType))
def parse_coroutine(value: types.CoroutineType) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="coroutine"),
        name=StringNode(value=value.__name__),
        qualname=StringNode(value=value.__qualname__),
    )


@parse_node.register(lambda v: hasattr(v, "__iter__") and hasattr(v, "__next__"))
def parse_iterator(value: Any) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="iterator"),
        class_=StringNode(value=type(value).__name__),
    )


@parse_node.register(lambda v: inspect.isasyncgen(v))
def parse_async_generator(value: Any) -> CompositeValueNode:
    return CompositeValueNode(
        type=StringNode(value="async_generator"),
        name=StringNode(value=value.__name__),
        qualname=StringNode(value=value.__qualname__),
    )
