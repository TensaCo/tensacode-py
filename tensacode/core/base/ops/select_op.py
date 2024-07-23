from typing import ClassVar
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op

class SelectOp(Op):
    name: ClassVar[str] = "select"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine
