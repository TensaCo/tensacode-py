from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseCallOp(Op):
    """
    Get or create values to call a function,
    conditioned by curried and direct invocation arguments
    """

    name: ClassVar[str] = "call"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseCallOp.create_subclass(name="call")
def Call(
    engine: BaseEngine,
    *inputs: list[Any],
    **kwargs: Any,
) -> Any:
    """
    Get or create values to call a function,
    conditioned by curried and direct invocation arguments
    """
    # Existing implementation
