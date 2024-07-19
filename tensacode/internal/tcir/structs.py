from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Callable, Iterable, Optional
import inspect
import io
import types
import datetime
import enum
import re
import dataclasses
from functools import cached_property
from types import NoneType, EllipsisType, UnionType
from typing import Mapping, Iterable
from pydantic import BaseModel, BaseConfig
from dataclasses import fields, is_dataclass, MISSING
import types
import weakref
import collections
import array
import ctypes
from io import IOBase
import decimal
import fractions
import pathlib
import ipaddress
import uuid
from tensacode.internal.utils.functional import polymorphic
from tensacode.internal.utils.pydantic import Tensor


class Node(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @cached_property
    @abstractmethod
    def dependants(self) -> list[Node]: ...

    @abstractmethod
    @property
    def python_value(self): ...


class AtomicValueNode(Node):

    value: Any

    def __hash__(self) -> int:
        return hash(self.value)

    @property
    def dependants(self) -> list[Node]:
        return []

    def python_value(self):
        return self.value


class StringNode(AtomicValueNode):
    value: str


class IntNode(AtomicValueNode):
    value: int


class FloatNode(AtomicValueNode):
    value: float


class ComplexNumberNode(AtomicValueNode):
    value: complex


class BoolNode(AtomicValueNode):
    value: bool


class NoneNode(AtomicValueNode):
    value: NoneType = None


class EllipsisNode(AtomicValueNode):
    value: EllipsisType = Ellipsis


class BytesNode(AtomicValueNode):
    value: bytes


class TensorNode(AtomicValueNode):
    value: Tensor


class SequenceNode(Node):
    items: list[Node]

    @property
    def dependants(self) -> list[Node]:
        return self.items

    @property
    def python_value(self):
        return [item.python_value for item in self.items]


class MappingNode(Node):
    items: dict[Node, Node]

    @property
    def dependants(self) -> list[Node]:
        return list(self.items.values())

    @property
    def python_value(self):
        return {k.python_value: v.python_value for k, v in self.items.items()}


class CompositeValueNode(Node):
    class Config(BaseConfig):
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def dependants(self) -> list[Node]:
        items = {k: getattr(self, k) for k in dir(self)}
        flattened = []
        while items:
            item = items.popitem()
            if isinstance(item, Node):
                flattened.append(item)
            elif isinstance(item, Mapping):
                items.append(item.values())
            elif isinstance(item, Iterable):
                items.append(list(item))
            else:
                pass

        return flattened

    @property
    def python_value(self):
        return self.model_dump()


class FunctionNode(CompositeValueNode):
    name: StringNode
    signature: dict[str, Node]
    body: Node

    @property
    def python_value(self):
        parameters = {k: v.python_value for k, v in self.signature.items()}
        return types.FunctionType(**parameters, body=self.body.python_value)


class ParameterNode(CompositeValueNode):
    name: StringNode
    annotation: Node
    value: Node

    @property
    def python_value(self):
        return inspect.Parameter(
            name=self.name.python_value,
            annotation=self.annotation.python_value,
            default=self.value.python_value,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )


class TypeNode(Node):
    name: StringNode
    type_args: Optional[list[Node]] = None


class UnionTypeNode(TypeNode):
    types: list[TypeNode]

    @cached_property
    def python_value(self):
        return types.UnionType(*self.types)


class ProductTypeNode(TypeNode):
    types: list[TypeNode]

    @cached_property
    def python_value(self):
        return type(self.name, *self.types, {})


class OptionalTypeNode(UnionTypeNode):

    @cached_property
    def type(self) -> TypeNode:
        return self.types[0]

    @type.setter
    def type(self, value: TypeNode):
        self.types[0] = value

    @property
    def python_value(self):
        return Optional[self.type.python_value]


class ClassNode(TypeNode):
    parents: list[TypeNode]
    members: list[Node]

    @cached_property
    def python_value(self):
        return type(
            self.name.python_value,
            tuple(self.parents.python_value),
            {k: v.python_value for k, v in self.members.items()},
        )


class FileNode(CompositeValueNode):
    name: StringNode
    mode: StringNode
    closed: BoolNode
    encoding: StringNode

    @cached_property
    def python_value(self):
        f = open(
            self.name.python_value,
            self.mode.python_value,
            encoding=self.encoding.python_value,
        )
        if self.closed.python_value:
            f.close()
        return f


try:
    import pandas as pd

    class PandasDataFrameNode(CompositeValueNode):
        data: SequenceNode
        index: SequenceNode
        columns: SequenceNode

        @cached_property
        def python_value(self):
            return pd.DataFrame(
                data=self.data.python_value,
                index=self.index.python_value,
                columns=self.columns.python_value,
            )

except ImportError:
    pass
