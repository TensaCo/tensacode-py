from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.core.base.ops.decode_op import DecodeOp


@BaseEngine.register_op()
def decide(
    engine: BaseEngine,
    latent: LatentType,
    *inputs: list[Any],
    **kwargs: Any,
) -> bool:
    return engine.decode(latent=latent, type=bool, **kwargs)
