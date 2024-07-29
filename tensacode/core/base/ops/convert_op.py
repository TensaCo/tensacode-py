from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class BaseConvertOp(Op):
    name: ClassVar[str] = "convert"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine


@BaseEngine.register_op_class_for_all_class_instances
@BaseConvertOp.create_subclass(name="convert")
def Convert(
    engine: BaseEngine,
    /,
    origin_value: Any,
    target_type: type[Any],
    modify_rounds=2,
    **kwargs: Any,
) -> Any:
    """Convert operation"""
    origin_latent = engine.encode(origin_value, **kwargs)
    target_value = engine.decode(latent=origin_latent, type=target_type, **kwargs)
    for _ in range(modify_rounds):
        target_value = engine.modify(target_value, origin=origin_latent, **kwargs)
    return target_value
