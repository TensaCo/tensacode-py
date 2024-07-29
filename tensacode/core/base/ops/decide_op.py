from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.core.base.ops.decode_op import DecodeOp


class BaseDecideOp(Op):
    name: ClassVar[str] = "decide"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseDecideOp.create_subclass(name="decide")
def Decide(
    engine: BaseEngine,
    latent: LatentType,
    *inputs: list[Any],
    **kwargs: Any,
) -> Any:
    return engine.decode(latent=latent, type=bool, **kwargs)
