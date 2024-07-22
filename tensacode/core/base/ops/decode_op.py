from abc import abstractmethod
from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


@BaseEngine.register_op_class_for_all_class_instances
class DecodeOp(BaseOp):
    op_name: ClassVar[str] = "decode"
    object_type: ClassVar[type[object]] = Any
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    @abstractmethod
    def _execute(
        self,
        input: LatentType,
        type: type[object],
        engine: BaseEngine,
        **kwargs,
    ):
        """Decode a representation back into an object"""
        pass
