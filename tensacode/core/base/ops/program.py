from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.ops.base_op import Op


@BaseEngine.register_op()
def program(
    engine: BaseEngine,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """Program operation"""
    raise NotImplementedError("Subclass must implement this method")
