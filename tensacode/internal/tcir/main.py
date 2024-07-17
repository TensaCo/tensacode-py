from __future__ import annotations
from abc import ABC, abstractmethod
from functools import cached_property
from inspect import isabstract
from typing import Callable, ClassVar, MappingProxyType, Any, Dict, Self
from typing_extensions import Unpack

from pydantic import BaseModel, Field, TupleGenerator, validator
from pydantic.config import ConfigDict
from pydantic import model_serializer, Discriminator, model_validator, ValidationError

from tensacode.internal.utils.functional import polymorphic


class BaseEntityWithDiscriminator(BaseModel):
    model_key: ClassVar[str] = "any"

    @model_validator(mode="before")
    @classmethod
    def route_to_subclass(cls, data: Any) -> TCIRAny:
        if not isinstance(data, dict):
            raise ValueError(f"{cls.__name__} must be read from a dict, got {data}")
        for subclass in cls.__subclasses__():
            if subclass.model_key == data["model_key"]:
                return subclass.model_validate(data)
        return BaseModel.model_validate(data)
        raise ValueError(f"Can't route value to subclass: {data}")


class TCIRBase(BaseEntityWithDiscriminator):

    _PYTHON_PARSING_PRIORITY = 0

    @abstractmethod
    @classmethod
    def can_parse_python(cls, value: Any) -> bool: ...

    @polymorphic
    @classmethod
    def from_python(cls, value: Any, *, depth: int = 4, **extra_kwargs) -> TCIRBase:
        if depth <= 0:
            return None
        return cls.model_validate(value)

    @abstractmethod
    def to_python(self) -> TCIRBase:
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        TCIRBase.from_python.register(
            cls.can_parse_python,
            priority=cls._PYTHON_PARSING_PRIORITY,
        )

    @property
    def deps(self) -> dict[str, TCIRBase]:
        items = self.model_dump()
        flattened_items = set()
        while items:
            k, v = items.popitem()
            if isinstance(v, (list, tuple, set, frozenset)):
                items.update(v)
            elif isinstance(v, (dict, MappingProxyType)):
                items.update(v.values())
            else:
                flattened_items.add(v)
        return {v for v in flattened_items if isinstance(v, TCIRBase)}
