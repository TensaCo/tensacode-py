"""
Programming primitives

Primarily intended for differentiable programming, though they may find general use also.
"""

from __future__ import annotations
from abc import ABC
from pydantic import BaseModel, Field as PydanticField


class Statement(BaseModel):
    purpose: str


class Module(BaseModel):
    body: list[Statement]


class Field(BaseModel):
    name: str
    value: str | None
    annotation: str | None

    @property
    def is_valid(self) -> bool:
        return self.value is not None or self.annotation is not None


class ProtoFunction(Statement, ABC):
    signature: str


class Function(ProtoFunction):
    name: str
    body: list[Statement]


class Lambda(ProtoFunction):
    body: Statement


class Class(Module):
    fields: list[Field]
    methods: list[Function]
    staticmethods: list[Function]


class ControlFlowStatement(Statement, ABC):
    pass


class While(ControlFlowStatement):
    condition: str
    body: List[Union[Assignment, If, While, For, FunctionCall]]
    line: int = None

    def __str__(self):
        return f"while {self.condition}:\n" + "\n".join(str(x) for x in self.body)
