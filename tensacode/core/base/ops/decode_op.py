from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseDecodeOp(Op):
    name: ClassVar[str] = "decode"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseDecodeOp.create_subclass(name="decode")
def Decode(
    self,
    *inputs: list[Any],
    engine: BaseEngine,
    **kwargs: Any,
) -> Any:
    """Decode operation"""
    # Existing implementation
