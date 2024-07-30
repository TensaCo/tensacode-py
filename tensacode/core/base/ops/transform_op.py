from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


# TODO: register the ops directly in the engine class file
# TODO: crate a separate `schema.py` file to remove cyclic dep issues for the Base* interface classes


@BaseEngine.register_op()
def transform(
    engine: BaseEngine,
    *inputs: list[Any],
    **kwargs: Any,
) -> Any:
    """Transform operation"""
    raise NotImplementedError("Subclass must implement this method")
