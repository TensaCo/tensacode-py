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


class Node(BaseModel):
    pass


class DataNode(Node):
    pass


class AtomicValueNode(DataNode):
    value: Any


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


class BytesNode(DataNode):
    value: bytes


class TensorNode(DataNode):
    data: Tensor


class SequenceNode(DataNode):
    items: list[Node]


class MappingNode(DataNode):
    items: dict[Node, Node]


class CompositeValueNode(DataNode):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)


class FunctionNode(CompositeValueNode):
    name: StringNode
    signature: dict[str, Node]
    body: Node


class ParameterNode(CompositeValueNode):
    name: StringNode
    annotation: Node
    value: Node


class TypeNode(DataNode):
    name: StringNode


class ClassNode(TypeNode):
    parents: list[TypeNode]
    members: list[Node]


class FileNode(CompositeValueNode):
    name: StringNode
    mode: StringNode
    closed: BoolNode
    encoding: StringNode

    _file: File


class PandasDataFrameNode(CompositeValueNode):
    data: SequenceNode
    index: SequenceNode
    columns: SequenceNode
