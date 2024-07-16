from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel


class Entity(BaseModel, ABC):
    @property
    @abstractmethod
    def value(self) -> Any: ...

    @value.setter
    @abstractmethod
    def value(self, value: Any) -> None: ...


class Atomic(Entity):
    value: Any


class Edge(BaseModel):
    label: Optional[Entity] = None
    directed: bool = False
    source: Entity
    target: Entity


class Graph(Entity):
    nodes: list[Entity]
    edges: list[Edge]
