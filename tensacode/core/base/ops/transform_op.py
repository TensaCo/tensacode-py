from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


@BaseEngine.register_op()
def Transform(
    engine: BaseEngine,
    *inputs: list[Any],
    **kwargs: Any,
) -> Any:
    """Transform operation"""
    raise NotImplementedError("Subclass must implement this method")
