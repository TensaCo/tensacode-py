from abc import ABC

from abc import ABC
from typing import Any

from tensacode.core.base.engine import BaseEngine
from tensacode.utils.object_types import ObjectType, LatentType


class BaseOp(ABC):
    op_name: str
    object_type: ObjectType
    latent_type: LatentType

    @abstractmethod
    def execute(self, engine: "BaseEngine", input: Any) -> Any:
        pass
