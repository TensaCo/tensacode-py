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
from abc import abstractmethod


class Node(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @cached_property
    @abstractmethod
    def dependants(self) -> list[Node]: ...

    def merge_with_identical(self, other: Node):
        if self == other:
            return other
        return self

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

    def merge_with_identical(self, other: Node):
        # no children to merge
        return super().merge_with_identical(other)


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

    def merge_with_identical(self, other: Node):
        for i, item in enumerate(self.items):
            self.items[i] = item.merge_with_identical(other)
        return self


class MappingNode(Node):
    items: dict[Node, Node]

    @property
    def dependants(self) -> list[Node]:
        return list(self.items.values())

    @property
    def python_value(self):
        return {k.python_value: v.python_value for k, v in self.items.items()}

    def merge_with_identical(self, other: Node):
        for key, value in self.items.items():
            self.items[key] = value.merge_with_identical(other)
        return self


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
            key, item = items.popitem()
            if isinstance(item, Node):
                flattened.append(item)
            elif isinstance(item, Mapping):
                items.update({f"{key}.{k}": v for k, v in item.items()})
            elif isinstance(item, Iterable) and not isinstance(item, str):
                items.update({f"{key}[{i}]": v for i, v in enumerate(item)})
        return flattened

    @property
    def python_value(self):
        return self.model_dump()

    def merge_with_identical(self, other: Node):
        for attr_name in dir(self):
            value = getattr(self, attr_name)
            if isinstance(value, Node):
                setattr(self, attr_name, value.merge_with_identical(other))
            elif isinstance(value, Mapping):
                merged_mapping = {
                    k: (
                        CompositeValueNode.merge_with_identical(v, other)
                        if isinstance(v, CompositeValueNode)
                        else v
                    )
                    for k, v in value.items()
                }
                setattr(self, attr_name, merged_mapping)
            elif isinstance(value, Iterable) and not isinstance(value, str):
                merged_iterable = [
                    (
                        CompositeValueNode.merge_with_identical(v, other)
                        if isinstance(v, CompositeValueNode)
                        else v
                    )
                    for v in value
                ]
                setattr(self, attr_name, type(value)(merged_iterable))
        return self


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

    @property
    def dependants(self) -> list[Node]:
        return [self.type_args] if self.type_args else []


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
        data: list[list[Node]]
        index: Node | list[Node]
        columns: Node | list[Node]

        @cached_property
        def python_value(self):
            return pd.DataFrame(
                data=self.data.python_value,
                index=self.index.python_value,
                columns=self.columns.python_value,
            )

except ImportError:
    pass
