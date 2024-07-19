from abc import ABC

from abc import ABC
from typing import Any

from tensacode.core.base.engine import BaseEngine
from tensacode.internal.protocols.latent_types import LatentType


class BaseOp(ABC):
    op_name: str
    latent_type: LatentType

    @abstractmethod
    def execute(self, engine: "BaseEngine", input: Any) -> Any:
        pass
