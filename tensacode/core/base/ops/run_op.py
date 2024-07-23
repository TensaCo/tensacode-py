from typing import Any, ClassVar, Optional

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


@BaseEngine.register_op_class_for_all_class_instances
class RunOp(BaseOp):
    op_name: ClassVar[str] = "run"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    def _execute(self, *args, engine: BaseEngine, **kwargs):
        """Execute a specific operation or task"""
        # Implementation goes here
        pass

    @classmethod
    def from_engine(cls, engine: BaseEngine) -> "RunOp":
        return cls()
