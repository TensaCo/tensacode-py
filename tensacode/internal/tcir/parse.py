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
from tensacode.internal.tcir.nodes import *


@polymorphic
def parse_node(value: Any) -> Node:
    """
    Default parser for any value. Returns a CompositeValueNode.

    Args:
        value (Any): The value to parse.

    Returns:
        Node: A CompositeValueNode representing the input value.

    >>> isinstance(parse_node({"a": 1, "b": 2}), CompositeValueNode)
    True
    >>> isinstance(parse_node([1, 2, 3]), CompositeValueNode)
    True
    >>> isinstance(parse_node("test"), CompositeValueNode)
    True
    """
    return CompositeValueNode(**value)  # Default case


@parse_node.register(lambda v: v is None)
def parse_none(value: None) -> NoneNode:
    """
    Parse a None value into a NoneNode.

    Args:
        value (None): The None value to parse.

    Returns:
        NoneNode: A NoneNode representing None.

    >>> isinstance(parse_none(None), NoneNode)
    True
    >>> parse_none(None).value is None
    True
    """
    return NoneNode()


@parse_node.register(lambda v: v is Ellipsis)
def parse_ellipsis(value: EllipsisType) -> EllipsisNode:
    """
    Parse an Ellipsis value into an EllipsisNode.

    Args:
        value (EllipsisType): The Ellipsis value to parse.

    Returns:
        EllipsisNode: An EllipsisNode representing Ellipsis.

    >>> isinstance(parse_ellipsis(...), EllipsisNode)
    True
    >>> parse_ellipsis(...).value is Ellipsis
    True
    """
    return EllipsisNode()


@parse_node.register(lambda v: isinstance(v, bool))
def parse_bool(value: bool) -> BoolNode:
    """
    Parse a boolean value into a BoolNode.

    Args:
        value (bool): The boolean value to parse.

    Returns:
        BoolNode: A BoolNode representing the input boolean.

    >>> isinstance(parse_bool(True), BoolNode)
    True
    >>> parse_bool(False).value is False
    True
    """
    return BoolNode(value=value)


@parse_node.register(lambda v: isinstance(v, int))
def parse_int(value: int) -> IntNode:
    """
    Parse an integer value into an IntNode.

    Args:
        value (int): The integer value to parse.

    Returns:
        IntNode: An IntNode representing the input integer.

    >>> isinstance(parse_int(42), IntNode)
    True
    >>> parse_int(-10).value == -10
    True
    """
    return IntNode(value=value)


@parse_node.register(lambda v: isinstance(v, float))
def parse_float(value: float) -> FloatNode:
    """
    Parse a float value into a FloatNode.

    Args:
        value (float): The float value to parse.

    Returns:
        FloatNode: A FloatNode representing the input float.

    >>> isinstance(parse_float(3.14), FloatNode)
    True
    >>> parse_float(-0.5).value == -0.5
    True
    """
    return FloatNode(value=value)


@parse_node.register(lambda v: isinstance(v, complex))
def parse_complex(value: complex) -> ComplexNumberNode:
    """
    Parse a complex number into a ComplexNumberNode.

    Args:
        value (complex): The complex number to parse.

    Returns:
        ComplexNumberNode: A ComplexNumberNode representing the input complex number.

    >>> isinstance(parse_complex(1+2j), ComplexNumberNode)
    True
    >>> parse_complex(3-4j).value == (3-4j)
    True
    """
    return ComplexNumberNode(value=value)


@parse_node.register(lambda v: isinstance(v, str))
def parse_string(value: str) -> StringNode:
    """
    Parse a string value into a StringNode.

    Args:
        value (str): The string value to parse.

    Returns:
        StringNode: A StringNode representing the input string.

    >>> isinstance(parse_string("hello"), StringNode)
    True
    >>> parse_string("world").value == "world"
    True
    """
    return StringNode(value=value)


@parse_node.register(lambda v: isinstance(v, bytes))
def parse_bytes(value: bytes) -> BytesNode:
    """
    Parse a bytes object into a BytesNode.

    Args:
        value (bytes): The bytes object to parse.

    Returns:
        BytesNode: A BytesNode representing the input bytes object.

    >>> isinstance(parse_bytes(b"hello"), BytesNode)
    True
    >>> parse_bytes(b"world").value == b"world"
    True
    """
    return BytesNode(value=value)


@parse_node.register(lambda v: isinstance(v, (list, tuple, set, frozenset)))
def parse_sequence(value: Iterable) -> SequenceNode:
    """
    Parse an iterable (list, tuple, set, or frozenset) into a SequenceNode.

    Args:
        value (Iterable): The iterable to parse.

    Returns:
        SequenceNode: A SequenceNode representing the input iterable.

    >>> isinstance(parse_sequence([1, 2, 3]), SequenceNode)
    True
    >>> isinstance(parse_sequence((4, 5, 6)), SequenceNode)
    True
    >>> isinstance(parse_sequence({7, 8, 9}), SequenceNode)
    True
    >>> isinstance(parse_sequence(frozenset([10, 11, 12])), SequenceNode)
    True
    """
    return SequenceNode(items=[parse_node(v) for v in value])


@parse_node.register(lambda v: isinstance(v, dict))
def parse_mapping(value: dict) -> MappingNode:
    """
    Parse a dictionary into a MappingNode.

    Args:
        value (dict): The dictionary to parse.

    Returns:
        MappingNode: A MappingNode representing the input dictionary.

    >>> isinstance(parse_mapping({"a": 1, "b": 2}), MappingNode)
    True
    >>> len(parse_mapping({"x": 10, "y": 20}).items) == 2
    True
    """
    return MappingNode(items={parse_node(k): parse_node(v) for k, v in value.items()})


@parse_node.register(lambda v: isinstance(v, tuple) and hasattr(v, "_fields"))
def parse_named_tuple(value: tuple) -> MappingNode:
    """
    Parse a named tuple into a MappingNode.

    Args:
        value (tuple): The named tuple to parse.

    Returns:
        MappingNode: A MappingNode representing the input named tuple.

    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> p = Point(1, 2)
    >>> isinstance(parse_named_tuple(p), MappingNode)
    True
    >>> len(parse_named_tuple(p).items) == 2
    True
    """
    return MappingNode(
        items={
            StringNode(value=field): parse_node(getattr(value, field))
            for field in value._fields
        }
    )


@parse_node.register(lambda v: is_dataclass(v))
def parse_dataclass(value: Any) -> CompositeValueNode:
    """
    Parse a dataclass instance into a CompositeValueNode.

    Args:
        value (Any): The dataclass instance to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the input dataclass instance.

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Person:
    ...     name: str
    ...     age: int
    >>> p = Person("Alice", 30)
    >>> isinstance(parse_dataclass(p), CompositeValueNode)
    True
    >>> parse_dataclass(p).type.value == "dataclass"
    True
    """
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
    """
    Parse a Pydantic model instance into a CompositeValueNode.

    Args:
        value (BaseModel): The Pydantic model instance to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the input Pydantic model instance.

    >>> from pydantic import BaseModel
    >>> class User(BaseModel):
    ...     name: str
    ...     email: str
    >>> u = User(name="Bob", email="bob@example.com")
    >>> isinstance(parse_pydantic_model(u), CompositeValueNode)
    True
    >>> parse_pydantic_model(u).type.value == "pydantic_model"
    True
    """
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
    """
    Parse a traceback object into a CompositeValueNode.

    Args:
        value (types.TracebackType): The traceback object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the input traceback object.

    >>> import sys
    >>> try:
    ...     1/0
    ... except ZeroDivisionError:
    ...     tb = sys.exc_info()[2]
    >>> isinstance(parse_traceback(tb), CompositeValueNode)
    True
    >>> parse_traceback(tb).type.value == "traceback"
    True
    """
    return CompositeValueNode(
        type=StringNode(value="traceback"),
        tb_frame=parse_node(value.tb_frame),
        tb_lasti=IntNode(value=value.tb_lasti),
        tb_lineno=IntNode(value=value.tb_lineno),
        tb_next=parse_node(value.tb_next) if value.tb_next else NoneNode(),
    )


@parse_node.register(lambda v: isinstance(v, (weakref.ref, weakref.ProxyType)))
def parse_weakref(value: weakref.ref | weakref.ProxyType) -> CompositeValueNode:
    """
    Parse a weak reference object into a CompositeValueNode.

    Args:
        value (weakref.ref | weakref.ProxyType): The weak reference object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the input weak reference object.

    >>> import weakref
    >>> class Dummy:
    ...     pass
    >>> obj = Dummy()
    >>> ref = weakref.ref(obj)
    >>> isinstance(parse_weakref(ref), CompositeValueNode)
    True
    >>> parse_weakref(ref).type.value == "weakref"
    True
    """
    return CompositeValueNode(
        type=StringNode(value="weakref"),
        referent_type=StringNode(value=type(value()).__name__ if value() else "None"),
    )


@parse_node.register(lambda v: isinstance(v, collections.deque))
def parse_deque(value: collections.deque) -> SequenceNode:
    """
    Parse a deque object into a SequenceNode.

    Args:
        value (collections.deque): The deque object to parse.

    Returns:
        SequenceNode: A SequenceNode representing the input deque object.

    >>> from collections import deque
    >>> d = deque([1, 2, 3])
    >>> isinstance(parse_deque(d), SequenceNode)
    True
    >>> len(parse_deque(d).items) == 3
    True
    """
    return SequenceNode(items=[parse_node(item) for item in value])


@parse_node.register(lambda v: isinstance(v, collections.Counter))
def parse_counter(value: collections.Counter) -> MappingNode:
    """
    Parse a Counter object into a MappingNode.

    Args:
        value (collections.Counter): The Counter object to parse.

    Returns:
        MappingNode: A MappingNode representing the input Counter object.

    >>> from collections import Counter
    >>> c = Counter(['a', 'b', 'c', 'a', 'b', 'b'])
    >>> isinstance(parse_counter(c), MappingNode)
    True
    >>> len(parse_counter(c).items) == 3
    True
    """
    return MappingNode(
        items={parse_node(k): IntNode(value=v) for k, v in value.items()}
    )


@parse_node.register(lambda v: isinstance(v, collections.OrderedDict))
def parse_ordered_dict(value: collections.OrderedDict) -> MappingNode:
    """
    Parse an OrderedDict object into a MappingNode.

    Args:
        value (collections.OrderedDict): The OrderedDict object to parse.

    Returns:
        MappingNode: A MappingNode representing the input OrderedDict object.

    >>> from collections import OrderedDict
    >>> od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    >>> isinstance(parse_ordered_dict(od), MappingNode)
    True
    >>> len(parse_ordered_dict(od).items) == 3
    True
    """
    return MappingNode(items={parse_node(k): parse_node(v) for k, v in value.items()})


@parse_node.register(lambda v: isinstance(v, collections.defaultdict))
def parse_defaultdict(value: collections.defaultdict) -> MappingNode:
    """
    Parse a defaultdict object into a MappingNode.

    Args:
        value (collections.defaultdict): The defaultdict object to parse.

    Returns:
        MappingNode: A MappingNode representing the input defaultdict object.

    >>> from collections import defaultdict
    >>> dd = defaultdict(int, {'a': 1, 'b': 2})
    >>> isinstance(parse_defaultdict(dd), MappingNode)
    True
    >>> len(parse_defaultdict(dd).items) == 2
    True
    """
    return MappingNode(items={parse_node(k): parse_node(v) for k, v in value.items()})


@parse_node.register(lambda v: isinstance(v, types.MappingProxyType))
def parse_mapping_proxy(value: types.MappingProxyType) -> MappingNode:
    """
    Parse a MappingProxyType object into a MappingNode.

    Args:
        value (types.MappingProxyType): The MappingProxyType object to parse.

    Returns:
        MappingNode: A MappingNode representing the input MappingProxyType object.

    >>> from types import MappingProxyType
    >>> mp = MappingProxyType({'a': 1, 'b': 2})
    >>> isinstance(parse_mapping_proxy(mp), MappingNode)
    True
    >>> len(parse_mapping_proxy(mp).items) == 2
    True
    """
    return MappingNode(items={parse_node(k): parse_node(v) for k, v in value.items()})


@parse_node.register(lambda v: isinstance(v, array.array))
def parse_array(value: array.array) -> SequenceNode:
    """
    Parse an array object into a SequenceNode.

    Args:
        value (array.array): The array object to parse.

    Returns:
        SequenceNode: A SequenceNode representing the input array object.

    >>> import array
    >>> arr = array.array('i', [1, 2, 3])
    >>> isinstance(parse_array(arr), SequenceNode)
    True
    >>> len(parse_array(arr).items) == 3
    True
    """
    return SequenceNode(items=[parse_node(item) for item in value])


@parse_node.register(lambda v: isinstance(v, memoryview))
def parse_memoryview(value: memoryview) -> CompositeValueNode:
    """
    Parse a memoryview object into a CompositeValueNode.

    Args:
        value (memoryview): The memoryview object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the input memoryview object.

    >>> mv = memoryview(b'abcdefg')
    >>> isinstance(parse_memoryview(mv), CompositeValueNode)
    True
    >>> parse_memoryview(mv).type.value == 'memoryview'
    True
    >>> parse_memoryview(mv).format.value == 'B'
    True
    >>> parse_memoryview(mv).itemsize.value == 1
    True
    >>> parse_memoryview(mv).ndim.value == 1
    True
    """
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
    """
    Parse a ctypes._SimpleCData object into a CompositeValueNode.

    Args:
        value (ctypes._SimpleCData): The ctypes._SimpleCData object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the input ctypes._SimpleCData object.

    >>> import ctypes
    >>> c_int = ctypes.c_int(42)
    >>> result = parse_ctypes(c_int)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "ctypes"
    True
    >>> result.ctype.value == "c_int"
    True
    >>> isinstance(result.value, IntNode)
    True
    >>> result.value.value == 42
    True
    """
    return CompositeValueNode(
        type=StringNode(value="ctypes"),
        ctype=StringNode(value=type(value).__name__),
        value=parse_node(value.value),
    )


@parse_node.register(lambda v: isinstance(v, decimal.Decimal))
def parse_decimal(value: decimal.Decimal) -> FloatNode:
    """
    Parse a Decimal object into a FloatNode.

    Args:
        value (decimal.Decimal): The Decimal object to parse.

    Returns:
        FloatNode: A FloatNode representing the input Decimal object.

    >>> import decimal
    >>> d = decimal.Decimal('3.14')
    >>> result = parse_decimal(d)
    >>> isinstance(result, FloatNode)
    True
    >>> result.value == 3.14
    True
    >>> abs(result.value - 3.14) < 1e-6  # Check for float equality with small tolerance
    True
    """
    return FloatNode(value=float(value))


@parse_node.register(lambda v: isinstance(v, fractions.Fraction))
def parse_fraction(value: fractions.Fraction) -> CompositeValueNode:
    """
    Parse a Fraction object into a CompositeValueNode.

    Args:
        value (fractions.Fraction): The Fraction object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the input Fraction object.

    >>> import fractions
    >>> f = fractions.Fraction(3, 4)
    >>> result = parse_fraction(f)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "Fraction"
    True
    >>> result.numerator.value == 3
    True
    >>> result.denominator.value == 4
    True
    >>> isinstance(result.float, FloatNode)
    True
    >>> abs(result.float.value - 0.75) < 1e-6
    True
    """
    return CompositeValueNode(
        type=StringNode(value="Fraction"),
        numerator=IntNode(value=value.numerator),
        denominator=IntNode(value=value.denominator),
        float=FloatNode(value=float(value)),
    )


@parse_node.register(lambda v: isinstance(v, pathlib.Path))
def parse_path(value: pathlib.Path) -> StringNode:
    """
    Parse a Path object into a StringNode.

    Args:
        value (pathlib.Path): The Path object to parse.

    Returns:
        StringNode: A StringNode representing the input Path object as a string.

    >>> import pathlib
    >>> p = pathlib.Path("/home/user/documents")
    >>> result = parse_path(p)
    >>> isinstance(result, StringNode)
    True
    >>> result.value == "/home/user/documents"
    True
    """
    return StringNode(value=str(value))


@parse_node.register(
    lambda v: isinstance(v, (ipaddress.IPv4Address, ipaddress.IPv6Address))
)
def parse_ip_address(
    value: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> StringNode:
    """
    Parse an IPv4Address or IPv6Address object into a StringNode.

    Args:
        value (ipaddress.IPv4Address | ipaddress.IPv6Address): The IP address object to parse.

    Returns:
        StringNode: A StringNode representing the input IP address object as a string.

    >>> import ipaddress
    >>> ipv4 = ipaddress.IPv4Address('192.168.0.1')
    >>> result = parse_ip_address(ipv4)
    >>> isinstance(result, StringNode)
    True
    >>> result.value == '192.168.0.1'
    True
    >>> ipv6 = ipaddress.IPv6Address('2001:db8::1')
    >>> result = parse_ip_address(ipv6)
    >>> isinstance(result, StringNode)
    True
    >>> result.value == '2001:db8::1'
    True
    """
    return StringNode(value=str(value))


@parse_node.register(lambda v: isinstance(v, uuid.UUID))
def parse_uuid(value: uuid.UUID) -> StringNode:
    """
    Parse a UUID object into a StringNode.

    Args:
        value (uuid.UUID): The UUID object to parse.

    Returns:
        StringNode: A StringNode representing the input UUID object as a string.

    >>> import uuid
    >>> test_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
    >>> result = parse_uuid(test_uuid)
    >>> isinstance(result, StringNode)
    True
    >>> result.value == '12345678-1234-5678-1234-567812345678'
    True
    """
    return StringNode(value=str(value))


@parse_node.register(lambda v: isinstance(v, types.GeneratorType))
def parse_generator(value: types.GeneratorType) -> SequenceNode:
    """
    Parse a generator object into a SequenceNode.

    Args:
        value (types.GeneratorType): The generator object to parse.

    Returns:
        SequenceNode: A SequenceNode containing parsed items from the generator.

    >>> def gen():
    ...     yield 1
    ...     yield 2
    ...     yield 3
    >>> g = gen()
    >>> result = parse_generator(g)
    >>> isinstance(result, SequenceNode)
    True
    >>> len(result.items) == 3
    True
    >>> all(isinstance(item, IntNode) for item in result.items)
    True
    >>> [item.value for item in result.items] == [1, 2, 3]
    True
    """
    return SequenceNode(items=[parse_node(item) for item in value])


@parse_node.register(lambda v: isinstance(v, io.IOBase))
def parse_file(value: io.IOBase) -> FileNode:
    """
    Parse a file-like object into a FileNode.

    Args:
        value (io.IOBase): The file-like object to parse.

    Returns:
        FileNode: A FileNode representing the input file-like object.

    >>> import io
    >>> f = io.StringIO("line1\\nline2\\nline3")
    >>> f.name = "test.txt"
    >>> f.mode = "r"
    >>> result = parse_file(f)
    >>> isinstance(result, FileNode)
    True
    >>> isinstance(result.name, StringNode) and result.name.value == "test.txt"
    True
    >>> isinstance(result.mode, StringNode) and result.mode.value == "r"
    True
    >>> isinstance(result.closed, BoolNode) and result.closed.value == False
    True
    >>> isinstance(result.encoding, NoneNode)
    True
    >>> isinstance(result.path, NoneNode)
    True
    >>> isinstance(result.lines, SequenceNode) and len(result.lines.items) == 3
    True
    >>> all(isinstance(line, StringNode) for line in result.lines.items)
    True
    >>> [line.value.strip() for line in result.lines.items] == ["line1", "line2", "line3"]
    True
    """
    return FileNode(
        name=parse_node(getattr(value, "name", None)),
        mode=parse_node(getattr(value, "mode", None)),
        closed=BoolNode(value=value.closed),
        encoding=parse_node(getattr(value, "encoding", None)),
        path=parse_node(getattr(value, "path", None)),
        lines=SequenceNode(
            items=[StringNode(value=line) for line in value.readlines()]
        ),
    )


@parse_node.register(
    lambda v: isinstance(v, (datetime.datetime, datetime.date, datetime.time))
)
def parse_datetime(
    value: datetime.datetime | datetime.date | datetime.time,
) -> CompositeValueNode:
    """
    Parse a datetime, date, or time object into a CompositeValueNode.

    Args:
        value (datetime.datetime | datetime.date | datetime.time): The datetime-related object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the input datetime-related object.

    >>> import datetime
    >>> dt = datetime.datetime(2023, 5, 1, 12, 30, 45)
    >>> result = parse_datetime(dt)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "datetime"
    True
    >>> result.year.value == 2023
    True
    >>> result.month.value == 5
    True
    >>> result.day.value == 1
    True
    >>> result.hour.value == 12
    True
    >>> result.minute.value == 30
    True
    >>> result.second.value == 45
    True

    >>> d = datetime.date(2023, 5, 1)
    >>> result = parse_datetime(d)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "date"
    True
    >>> result.year.value == 2023
    True
    >>> result.month.value == 5
    True
    >>> result.day.value == 1
    True

    >>> t = datetime.time(12, 30, 45)
    >>> result = parse_datetime(t)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "time"
    True
    >>> result.hour.value == 12
    True
    >>> result.minute.value == 30
    True
    >>> result.second.value == 45
    True
    """
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
    """
    Parse an Enum object into a CompositeValueNode.

    Args:
        value (enum.Enum): The Enum object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the Enum, containing:
            - type: StringNode with value "enum"
            - enum_class: StringNode with the name of the Enum class
            - name: StringNode with the name of the Enum member
            - value: Parsed value of the Enum member

    >>> import enum
    >>> class Color(enum.Enum):
    ...     RED = 1
    ...     GREEN = 2
    ...     BLUE = 3
    >>> result = parse_enum(Color.RED)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "enum"
    True
    >>> result.enum_class.value == "Color"
    True
    >>> result.name.value == "RED"
    True
    >>> isinstance(result.value, IntNode)
    True
    >>> result.value.value == 1
    True
    """
    return CompositeValueNode(
        type=StringNode(value="enum"),
        enum_class=StringNode(value=value.__class__.__name__),
        name=StringNode(value=value.name),
        value=parse_node(value.value),
    )


@parse_node.register(lambda v: isinstance(v, re.Pattern))
def parse_regex(value: re.Pattern) -> CompositeValueNode:
    """
    Parse a regular expression Pattern object into a CompositeValueNode.

    Args:
        value (re.Pattern): The regular expression Pattern object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the regex pattern, containing:
            - type: StringNode with value "regex"
            - pattern: StringNode with the regex pattern string
            - flags: IntNode with the regex flags

    >>> import re
    >>> pattern = re.compile(r'\d+', re.IGNORECASE)
    >>> result = parse_regex(pattern)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "regex"
    True
    >>> result.pattern.value == r'\d+'
    True
    >>> isinstance(result.flags, IntNode)
    True
    >>> result.flags.value == re.IGNORECASE
    True
    """
    return CompositeValueNode(
        type=StringNode(value="regex"),
        pattern=StringNode(value=value.pattern),
        flags=IntNode(value=value.flags),
    )


@parse_node.register(lambda v: isinstance(v, tuple) and hasattr(v, "_fields"))
def parse_namedtuple(value: tuple) -> CompositeValueNode:
    """
    Parse a namedtuple object into a CompositeValueNode.

    Args:
        value (tuple): The namedtuple object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the namedtuple, containing:
            - type: StringNode with value "namedtuple"
            - name: StringNode with the name of the namedtuple class
            - fields: MappingNode with field names as keys and parsed field values as values

    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> p = Point(1, 2)
    >>> result = parse_namedtuple(p)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "namedtuple"
    True
    >>> result.name.value == "Point"
    True
    >>> isinstance(result.fields, MappingNode)
    True
    >>> all(isinstance(k, StringNode) for k in result.fields.items.keys())
    True
    >>> all(isinstance(v, Node) for v in result.fields.items.values())
    True
    """
    return CompositeValueNode(
        type=StringNode(value="namedtuple"),
        name=StringNode(value=type(value).__name__),
        fields=MappingNode(
            items={
                StringNode(value=field): parse_node(getattr(value, field))
                for field in value._fields
            }
        ),
    )


try:
    import numpy as np

    @parse_node.register(lambda v: isinstance(v, np.ndarray))
    def parse_numpy_array(value: np.ndarray) -> CompositeValueNode:
        """
        Parse a NumPy ndarray object into a CompositeValueNode.

        Args:
            value (np.ndarray): The NumPy ndarray to parse.

        Returns:
            CompositeValueNode: A CompositeValueNode representing the NumPy array, containing:
                - type: StringNode with value "numpy.ndarray"
                - shape: SequenceNode with IntNodes representing the array dimensions
                - dtype: StringNode with the data type of the array
                - data: TensorNode containing the array data

        >>> import numpy as np
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> result = parse_numpy_array(arr)
        >>> isinstance(result, CompositeValueNode)
        True
        >>> result.type.value == "numpy.ndarray"
        True
        >>> isinstance(result.shape, SequenceNode)
        True
        >>> all(isinstance(item, IntNode) for item in result.shape.items)
        True
        >>> isinstance(result.dtype, StringNode)
        True
        >>> isinstance(result.data, TensorNode)
        True
        """
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
        """
        Parse a pandas Series object into a PandasDataFrameNode.

        Args:
            value (pd.Series): The pandas Series to parse.

        Returns:
            PandasDataFrameNode: A PandasDataFrameNode representing the Series, containing:
                - data: SequenceNode with parsed Series values
                - index: SequenceNode with parsed index values
                - columns: SequenceNode with a single StringNode for the Series name

        >>> import pandas as pd
        >>> s = pd.Series([1, 2, 3], name='test')
        >>> result = parse_pandas_series(s)
        >>> isinstance(result, PandasDataFrameNode)
        True
        >>> isinstance(result.data, SequenceNode)
        True
        >>> isinstance(result.index, SequenceNode)
        True
        >>> isinstance(result.columns, SequenceNode)
        True
        >>> len(result.columns.items) == 1
        True
        >>> isinstance(result.columns.items[0], StringNode)
        True
        >>> result.columns.items[0].value == 'test'
        True
        """
        return PandasDataFrameNode(
            data=SequenceNode(items=[parse_node(item) for item in value]),
            index=SequenceNode(items=[parse_node(idx) for idx in value.index]),
            columns=SequenceNode(items=[StringNode(value=str(value.name))]),
        )

    @parse_node.register(lambda v: isinstance(v, pd.DataFrame))
    def parse_pandas_dataframe(value: pd.DataFrame) -> PandasDataFrameNode:
        """
        Parse a pandas DataFrame object into a PandasDataFrameNode.

        Args:
            value (pd.DataFrame): The pandas DataFrame to parse.

        Returns:
            PandasDataFrameNode: A PandasDataFrameNode representing the DataFrame, containing:
                - data: SequenceNode of SequenceNodes, each representing a row of parsed values
                - index: SequenceNode with parsed index values
                - columns: SequenceNode with StringNodes for column names

        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> result = parse_pandas_dataframe(df)
        >>> isinstance(result, PandasDataFrameNode)
        True
        >>> isinstance(result.data, SequenceNode)
        True
        >>> all(isinstance(row, SequenceNode) for row in result.data.items)
        True
        >>> isinstance(result.index, SequenceNode)
        True
        >>> isinstance(result.columns, SequenceNode)
        True
        >>> all(isinstance(col, StringNode) for col in result.columns.items)
        True
        """
        return PandasDataFrameNode(
            data=SequenceNode(
                items=[
                    SequenceNode(items=[parse_node(item) for item in row])
                    for _, row in value.iterrows()
                ]
            ),
            index=SequenceNode(items=[parse_node(idx) for idx in value.index]),
            columns=SequenceNode(
                items=[StringNode(value=col) for col in value.columns]
            ),
        )

except ImportError:
    pass

# Add support for custom types or any other specific types you need to handle


@parse_node.register(lambda v: isinstance(v, type))
def parse_class(value: type) -> ClassNode:
    """
    Parse a class object into a ClassNode.

    Args:
        value (type): The class object to parse.

    Returns:
        ClassNode: A ClassNode representing the class, containing:
            - name: The name of the class
            - parents: List of parsed parent classes (excluding 'object')
            - members: List of parsed class attributes and methods

    >>> class TestClass:
    ...     def method(self):
    ...         pass
    >>> result = parse_class(TestClass)
    >>> isinstance(result, ClassNode)
    True
    >>> result.name == "TestClass"
    True
    >>> isinstance(result.parents, list)
    True
    >>> isinstance(result.members, list)
    True
    >>> any(member.name == "method" for member in result.members)
    True
    """
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
    """
    Parse a callable object (function or method) into a FunctionNode.

    Args:
        value (Callable): The callable object to parse.

    Returns:
        FunctionNode: A FunctionNode representing the function, containing:
            - name: The name of the function
            - parameters: List of parsed function parameters
            - return_type: Parsed return type annotation (or NoneNode if not specified)
            - body: Source code of the function

    >>> def test_func(a: int, b: str = "default") -> bool:
    ...     return True
    >>> result = parse_function(test_func)
    >>> isinstance(result, FunctionNode)
    True
    >>> result.name == "test_func"
    True
    >>> isinstance(result.parameters, list)
    True
    >>> len(result.parameters) == 2
    True
    >>> isinstance(result.return_type, Node)
    True
    >>> isinstance(result.body, str)
    True
    """
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
    """
    Parse an inspect.Parameter object into a CompositeValueNode.

    Args:
        value (inspect.Parameter): The Parameter object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the parameter, containing:
            - type: StringNode with value "parameter"
            - name: StringNode with the parameter name
            - annotation: Parsed parameter type annotation

    >>> from inspect import Parameter
    >>> param = Parameter('test_param', Parameter.POSITIONAL_OR_KEYWORD, annotation=int)
    >>> result = parse_parameter(param)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "parameter"
    True
    >>> result.name.value == "test_param"
    True
    >>> isinstance(result.annotation, Node)
    True
    """
    return CompositeValueNode(
        type=StringNode(value="parameter"),
        name=StringNode(value=value.name),
        annotation=parse_node(value.annotation),
    )


@parse_node.register(lambda v: isinstance(v, types.ModuleType))
def parse_module(value: types.ModuleType) -> CompositeValueNode:
    """
    Parse a module object into a CompositeValueNode.

    Args:
        value (types.ModuleType): The module object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the module, containing:
            - type: StringNode with value "module"
            - name: StringNode with the module name
            - file: Parsed module file path
            - members: MappingNode with module attributes and their parsed values

    >>> import math
    >>> result = parse_module(math)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "module"
    True
    >>> result.name.value == "math"
    True
    >>> isinstance(result.file, Node)
    True
    >>> isinstance(result.members, MappingNode)
    True
    """
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
    """
    Parse a code object into a CompositeValueNode.

    Args:
        value (types.CodeType): The code object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the code object, containing:
            - type: StringNode with value "code_object"
            - co_name: StringNode with the code object name
            - co_filename: StringNode with the filename
            - co_firstlineno: IntNode with the first line number

    >>> def test_func():
    ...     pass
    >>> code_obj = test_func.__code__
    >>> result = parse_code_object(code_obj)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "code_object"
    True
    >>> isinstance(result.co_name, StringNode)
    True
    >>> isinstance(result.co_filename, StringNode)
    True
    >>> isinstance(result.co_firstlineno, IntNode)
    True
    """
    return CompositeValueNode(
        type=StringNode(value="code_object"),
        co_name=StringNode(value=value.co_name),
        co_filename=StringNode(value=value.co_filename),
        co_firstlineno=IntNode(value=value.co_firstlineno),
    )


@parse_node.register(lambda v: isinstance(v, types.FrameType))
def parse_frame_object(value: types.FrameType) -> CompositeValueNode:
    """
    Parse a frame object into a CompositeValueNode.

    Args:
        value (types.FrameType): The frame object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the frame object, containing:
            - type: StringNode with value "frame"
            - f_code: Parsed code object
            - f_lineno: IntNode with the current line number
            - f_locals: Parsed local variables dictionary

    >>> import sys
    >>> def test_func():
    ...     frame = sys._getframe()
    ...     return parse_frame_object(frame)
    >>> result = test_func()
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "frame"
    True
    >>> isinstance(result.f_code, Node)
    True
    >>> isinstance(result.f_lineno, IntNode)
    True
    >>> isinstance(result.f_locals, Node)
    True
    """
    return CompositeValueNode(
        type=StringNode(value="frame"),
        f_code=parse_node(value.f_code),
        f_lineno=IntNode(value=value.f_lineno),
        f_locals=parse_node(value.f_locals),
    )


@parse_node.register(lambda v: isinstance(v, (types.MethodType, types.FunctionType)))
def parse_method(value: types.MethodType | types.FunctionType) -> CompositeValueNode:
    """
    Parse a method or function object into a CompositeValueNode.

    Args:
        value (types.MethodType | types.FunctionType): The method or function object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the method or function, containing:
            - type: StringNode with value "method" or "function"
            - name: StringNode with the method or function name
            - qualname: StringNode with the qualified name
            - module: StringNode with the module name

    >>> def test_func():
    ...     pass
    >>> result = parse_method(test_func)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "function"
    True
    >>> isinstance(result.name, StringNode)
    True
    >>> isinstance(result.qualname, StringNode)
    True
    """
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
    """
    Parse a property object into a CompositeValueNode.

    Args:
        value (property): The property object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the property, containing:
            - type: StringNode with value "property"
            - fget: Parsed getter function
            - fset: Parsed setter function
            - fdel: Parsed deleter function

    >>> class TestClass:
    ...     @property
    ...     def test_prop(self):
    ...         return 42
    >>> result = parse_property(TestClass.test_prop)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "property"
    True
    >>> isinstance(result.fget, Node)
    True
    >>> isinstance(result.fset, Node)
    True
    >>> isinstance(result.fdel, Node)
    True
    """
    return CompositeValueNode(
        type=StringNode(value="property"),
        fget=parse_node(value.fget),
        fset=parse_node(value.fset),
        fdel=parse_node(value.fdel),
    )


@parse_node.register(lambda v: isinstance(v, slice))
def parse_slice(value: slice) -> CompositeValueNode:
    """
    Parse a slice object into a CompositeValueNode.

    Args:
        value (slice): The slice object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the slice, containing:
            - type: StringNode with value "slice"
            - start: Parsed start value
            - stop: Parsed stop value
            - step: Parsed step value

    >>> s = slice(1, 10, 2)
    >>> result = parse_slice(s)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "slice"
    True
    >>> isinstance(result.start, Node)
    True
    >>> isinstance(result.stop, Node)
    True
    >>> isinstance(result.step, Node)
    True
    """
    return CompositeValueNode(
        type=StringNode(value="slice"),
        start=parse_node(value.start),
        stop=parse_node(value.stop),
        step=parse_node(value.step),
    )


@parse_node.register(lambda v: isinstance(v, range))
def parse_range(value: range) -> CompositeValueNode:
    """
    Parse a range object into a CompositeValueNode.

    Args:
        value (range): The range object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the range, containing:
            - type: StringNode with value "range"
            - start: IntNode with the start value
            - stop: IntNode with the stop value
            - step: IntNode with the step value

    >>> r = range(0, 10, 2)
    >>> result = parse_range(r)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "range"
    True
    >>> isinstance(result.start, IntNode)
    True
    >>> isinstance(result.stop, IntNode)
    True
    >>> isinstance(result.step, IntNode)
    True
    """
    return CompositeValueNode(
        type=StringNode(value="range"),
        start=IntNode(value=value.start),
        stop=IntNode(value=value.stop),
        step=IntNode(value=value.step),
    )


@parse_node.register(lambda v: isinstance(v, types.CoroutineType))
def parse_coroutine(value: types.CoroutineType) -> CompositeValueNode:
    """
    Parse a coroutine object into a CompositeValueNode.

    Args:
        value (types.CoroutineType): The coroutine object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the coroutine, containing:
            - type: StringNode with value "coroutine"
            - name: StringNode with the coroutine name
            - qualname: StringNode with the qualified name

    >>> import asyncio
    >>> async def test_coroutine(): pass
    >>> coro = test_coroutine()
    >>> result = parse_coroutine(coro)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "coroutine"
    True
    >>> isinstance(result.name, StringNode)
    True
    >>> isinstance(result.qualname, StringNode)
    True
    >>> asyncio.get_event_loop().run_until_complete(coro)  # Clean up coroutine
    """
    return CompositeValueNode(
        type=StringNode(value="coroutine"),
        name=StringNode(value=value.__name__),
        qualname=StringNode(value=value.__qualname__),
    )


@parse_node.register(lambda v: hasattr(v, "__iter__") and hasattr(v, "__next__"))
def parse_iterator(value: Any) -> CompositeValueNode:
    """
    Parse an iterator object into a CompositeValueNode.

    This function handles any object that has both __iter__ and __next__ methods,
    which are the defining characteristics of an iterator in Python.

    Args:
        value (Any): The iterator object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the iterator, containing:
            - type: StringNode with value "iterator"
            - class_: StringNode with the name of the iterator's class

    >>> class TestIterator:
    ...     def __iter__(self): return self
    ...     def __next__(self): return 1
    >>> it = TestIterator()
    >>> result = parse_iterator(it)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "iterator"
    True
    >>> isinstance(result.class_, StringNode)
    True
    """
    return CompositeValueNode(
        type=StringNode(value="iterator"),
        class_=StringNode(value=type(value).__name__),
    )


@parse_node.register(lambda v: inspect.isasyncgen(v))
def parse_async_generator(value: Any) -> CompositeValueNode:
    """
    Parse an asynchronous generator object into a CompositeValueNode.

    This function handles asynchronous generator objects, which are created by
    async def functions that contain yield statements.

    Args:
        value (Any): The asynchronous generator object to parse.

    Returns:
        CompositeValueNode: A CompositeValueNode representing the async generator, containing:
            - type: StringNode with value "async_generator"
            - name: StringNode with the name of the async generator function
            - qualname: StringNode with the qualified name of the async generator function

    >>> import asyncio
    >>> async def test_async_gen():
    ...     yield 1
    >>> async_gen = test_async_gen()
    >>> result = parse_async_generator(async_gen)
    >>> isinstance(result, CompositeValueNode)
    True
    >>> result.type.value == "async_generator"
    True
    >>> isinstance(result.name, StringNode)
    True
    >>> isinstance(result.qualname, StringNode)
    True
    >>> asyncio.get_event_loop().run_until_complete(async_gen.aclose())  # Clean up async generator
    """
    return CompositeValueNode(
        type=StringNode(value="async_generator"),
        name=StringNode(value=value.__name__),
        qualname=StringNode(value=value.__qualname__),
    )

@parse_node.register(lambda v: isinstance(v, Node))
def parse_already_parsed_node(value: Node) -> Node:
    """
    Return an already parsed Node without modification.

    This function is used when the input value is already a Node instance,
    so no further parsing is needed.

    Args:
        value (Node): The already parsed Node instance.

    Returns:
        Node: The same Node instance that was passed in.

    >>> from tensacode.internal.tcir.nodes import StringNode
    >>> node = StringNode(value="test")
    >>> result = parse_already_parsed_node(node)
    >>> result is node
    True
    >>> isinstance(result, StringNode)
    True
    >>> result.value == "test"
    True
    """
    return value