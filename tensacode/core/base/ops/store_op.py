from abc import ABC

from abc import ABC, abstractmethod
from typing import Any

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


class StoreOp(BaseOp):
    def execute(self, engine: BaseEngine, input: Any, **kwargs: Any) -> Any:
        """Store an object or information"""
        # Implementation goes here
        pass
