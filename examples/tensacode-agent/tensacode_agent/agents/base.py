from abc import ABC, abstractmethod
from typing import Any


class BaseEntity(ABC):
    pass


class Node(BaseEntity):
    pass


class Environment(BaseEntity):
    pass


class Agent(BaseEntity, ABC):

    def step(self, environment: Environment) -> None:
        pass


class CodeNode(Node):
    @abstractmethod
    def get_source(self) -> str:
        pass

    @abstractmethod
    def execute(self, locals: dict, globals: dict) -> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def value(self) -> list[str]: ...


class ClassCodeNode(CodeNode):
    name: str
