from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BasePredictOp(Op):
    """Docstring for BasePredictOp"""

    name: ClassVar[str] = "predict"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BasePredictOp.create_subclass(name="predict")
def Predict(
    self,
    *inputs: list[Any],
    engine: BaseEngine,
    **kwargs: Any,
) -> Any:
    """Existing docstring moved here"""
    # Existing implementation
