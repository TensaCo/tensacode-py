from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op

class BaseRunOp(Op):
    """Docstring for BaseRunOp"""

    name: ClassVar[str] = "run"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine

@BaseEngine.register_op_class_for_all_class_instances
@BaseRunOp.create_subclass(name="run")
def Run(
    self,
    *inputs: list[Any],
    engine: BaseEngine,
    **kwargs: Any,
) -> Any:
    """Existing docstring moved here"""
    # Existing implementation
