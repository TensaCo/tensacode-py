from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


class StoreOp(BaseOp):
    op_name: ClassVar[str] = "store"
    object_type: ClassVar[type[object]] = Any
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    def _execute(self, *args, engine: BaseEngine, **kwargs):
        """Store an object or information"""
        # Implementation goes here
        pass

    @classmethod
    def from_engine(cls, engine: BaseEngine) -> Self:
        return cls(prompt="")
