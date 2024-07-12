from __future__ import annotations
from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar
import typing

from pydantic import BaseModel, Field, TupleGenerator, validator


class Serializable(ABC):
    pass


class NotSerializable(ABC):
    pass


class Any(BaseModel, Serializable, ABC):
    name: ClassVar[str]


class Type(Any, ABC):

    model_key: ClassVar[str]
    name: str
    _native_type: type

    def issubclass(self, other) -> bool:
        return issubclass(other, self._native_type)


class UnionType(Type):
    model_key: ClassVar[str] = "union_type"
    types: tuple[Type, ...]


class ProductType(Type):
    model_key: ClassVar[str] = "product_type"
    types: tuple[Type, ...]


class OptionalType(Type):
    model_key: ClassVar[str] = "optional_type"
    type: Type


class DeclarationScope(ABC):
    items: dict[str, Value]

    @property
    def functions(self) -> dict[str, Function]:
        return {k: v for k, v in self.items.items() if isinstance(v, Function)}

    @property
    def classes(self) -> dict[str, Type]:
        return {k: v for k, v in self.items.items() if isinstance(v, Type)}

    @property
    def variables(self) -> dict[str, Value]:
        return {
            k: v for k, v in self.items.items() if not isinstance(v, (Function, Type))
        }


class InterfaceType(Type, DeclarationScope):
    model_key: ClassVar[str] = "interface"


class EnumType(Type):
    model_key: ClassVar[str] = "enum"
    members: dict[str, Value]


class AliasType(Type):
    model_key: ClassVar[str] = "alias"
    original_type: Type


class LiteralType(Type):
    model_key: ClassVar[str] = "literal"
    value: Any


class Class(Type, ABC):

    model_key: ClassVar[str] = "class"
    name: str
    bases: tuple[Type, ...]
    items: dict[str, Value]
    annotations: dict[str, Type]

    @cached_property
    def _native_type(self) -> Type:
        if self.name in self._REGISTERED_CLASSES:
            return self._REGISTERED_CLASSES[self.name]
        return type(
            self.name,
            self.bases,
            {
                "__annotations__": self.annotations,
                **self.items,
            },
        )

    _REGISTERED_CLASSES: dict[str, Type] = {}

    @classmethod
    def register_class(cls, class_name: str, class_type: Type) -> None:
        cls._REGISTERED_CLASSES[class_name] = class_type


class Value(Any, ABC):
    type: Type

    def isinstance(self, other) -> bool:
        return isinstance(other, self.type)


class IsMutable(Value, ABC):
    pass


class IsImmutable(Value, ABC):
    pass


class IsHashable(Value, ABC):
    pass


class IsIterable(Value, ABC):
    @abstractmethod
    def __iter__(self) -> TupleGenerator: ...


class IsCallable(Value, ABC):
    pass


class Primitive(IsImmutable, Value, ABC):
    pass


class IsNumeric(Primitive, ABC):
    pass


class Composite(Value, ABC):
    pass
