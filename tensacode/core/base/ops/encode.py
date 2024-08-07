from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


@BaseEngine.register_op()
def encode(
    engine: BaseEngine,
    *inputs: list[Any],
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Encode one or more inputs into a latent representation.

    This operation uses the engine to convert input objects into a latent representation.

    Args:
        engine (BaseEngine): The engine used for encoding.
        *inputs (list[Any]): The input objects to be encoded.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the encoding process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The resulting latent representation.

    Raises:
        NotImplementedError: This method must be implemented by a subclass.

    Examples:
        >>> text = "Hello, world!"
        >>> latent = encode(engine, text)
        >>> print(type(latent))
        <class 'tensacode.internal.latent.LatentType'>

        >>> obj1 = {"name": "Alice", "age": 30}
        >>> obj2 = ["apple", "banana", "orange"]
        >>> latent = encode(engine, obj1, obj2)
        >>> print(type(latent))
        <class 'tensacode.internal.latent.LatentType'>
    """
    raise NotImplementedError("Subclass must implement this method")
