from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.protocols.latent import LatentType
from tensacode.core.base.ops.base_op import Op


class CallOp(Op):
    """
    Get or create values to call a function,
    conditioned by curried and direct invocation arguments
    """

    op_name: ClassVar[str] = "call"
    latent_type: ClassVar[LatentType] = LatentType
    engine_type: ClassVar[type[BaseEngine]] = BaseEngine
