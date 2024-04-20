from types import NoneType
from typing import Any, Self
from pydantic import BaseModel


class TCIRNode(BaseModel):
    pass


class TCIRDataNode(TCIRNode):
    object: Any

    dict: dict[str, Any]
    name: str  # __name__
    qualname: str  # __qualname__
    module: str

    @classmethod
    def parse(cls, object: Any) -> Self:
        return cls(
            object=object,
            dict=object.__dict__,
            name=getattr(object, "__name__", None),
            qualname=getattr(object, "__qualname__", None),
            module=getattr(object, "__module__", None),
        )


class TCIRControlNode(TCIRNode):
    pass


class TCIRConditionalNode(TCIRControlNode):
    conditions_and_cases: tuple[Condition, Case]


class ModuleTypeMeta(TCIRNode):
    pass


class FunctionTypeMeta(TCIRNode):
    pass
