from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


@BaseEngine.register_op_class_for_all_class_instances
class ConvertOp(BaseOp):
    op_name: ClassVar[str] = "convert"
    object_type: ClassVar[type[object]] = Any
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    def _execute(self, *args, engine: BaseEngine, **kwargs):
        """Convert between different types of objects"""
        # Implementation goes here
        pass
