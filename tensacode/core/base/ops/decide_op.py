from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.core.base.ops.decode_op import DecodeOp


class DecideOp(Op):
    name: ClassVar[str] = "decide"
    object_type: ClassVar[type[object]] = bool
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine
