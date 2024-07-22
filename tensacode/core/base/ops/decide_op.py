from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp
from tensacode.core.base.ops.decode_op import DecodeOp


@BaseEngine.register_op_class_for_all_class_instances
class DecideOp(BaseOp):
    op_name: ClassVar[str] = "decide"
    object_type: ClassVar[type[object]] = bool
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    def _execute(self, latent: LatentType, /, engine: BaseEngine, **kwargs) -> bool:
        """Make a boolean decision"""
        return engine.decode(latent=latent, type=bool, **kwargs)

    @classmethod
    def from_engine(cls, engine: BaseEngine) -> Self:
        return cls(prompt="")
