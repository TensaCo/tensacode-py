from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseTransformOp(Op):
    """Docstring for BaseTransformOp"""

    name: ClassVar[str] = "transform"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseTransformOp.create_subclass(name="transform")
def Transform(
    engine: BaseEngine,
    *inputs: list[Any],
    **kwargs: Any,
) -> Any:
    """Transform operation"""
    raise NotImplementedError("Subclass must implement this method")
