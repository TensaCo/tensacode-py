from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


@BaseEngine.register_op_class_for_all_class_instances
class PredictOp(BaseOp):
    op_name: ClassVar[str] = "predict"
    object_type: ClassVar[type[object]] = Any
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    def _execute(self, *args, engine: BaseEngine, **kwargs):
        """Make a prediction based on input data"""
        # Implementation goes here
        # Add logic to make predictions based on the input
        pass

    @classmethod
    def from_engine(cls, engine: BaseEngine) -> Self:
        return cls(prompt="")
