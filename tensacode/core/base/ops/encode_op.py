from abc import abstractmethod
from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


@BaseEngine.register_op_class_for_all_class_instances
class EncodeOp(BaseOp):
    op_name: ClassVar[str] = "encode"
    object_type: ClassVar[type[object]] = Any
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    @abstractmethod
    def _execute(self, input: Any, engine: BaseEngine, **kwargs):
        """Encode an object into a representation"""
        pass

    @classmethod
    def from_engine(cls, engine: BaseEngine) -> Self:
        return cls(prompt="")
