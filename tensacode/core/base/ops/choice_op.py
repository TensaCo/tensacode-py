from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import BaseOp


@BaseEngine.register_op_class_for_all_class_instances
class ChoiceOp(BaseOp):
    op_name: ClassVar[str] = "choice"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

    def _execute(self, *args, engine: BaseEngine, **kwargs):
        """Choose from one of a finite collection of options"""
        # Implementation goes here
        pass
