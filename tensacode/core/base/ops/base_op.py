from abc import ABC

from abc import ABC, abstractmethod
from typing import Any

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType


class BaseOp(ABC):
    op_name: str
    latent_type: LatentType

    @abstractmethod
    def execute(self, engine: BaseEngine, input: Any, **kwargs: Any) -> Any:
        pass
