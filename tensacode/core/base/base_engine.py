from abc import abstractmethod
from typing import Any
from pydantic import BaseModel

from tensacode.internal.utils.language import Language
from tensacode.internal.protocols.tagged_object import HasID
from tensacode.core.base.ops.ops import BaseOp


class BaseEngine(HasID, HasRegistry, BaseModel):

    tensacode_version: str
    render_language: Language = "python"
    config: dict[str, Any]
    _registry: ClassVar[Registry["BaseEngine"]] = Registry()

    def execute(self, op: BaseOp, input: Any, **kwargs: Any) -> Any:
        return op.execute(self, input, **kwargs)
