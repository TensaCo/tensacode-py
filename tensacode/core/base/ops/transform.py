from typing import ClassVar, Any
from tensacode.core.base.base_engine import BaseEngine
from tensacode.core.base.ops.base_op import Op


@BaseEngine.register_op()
def transform(
    engine: BaseEngine,
    *inputs: list[Any],
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Transform the input(s) into a new form.

    This operation applies a transformation to the given inputs, guided by the engine and optional prompt.

    Args:
        engine (BaseEngine): The engine used for transformation.
        *inputs (list[Any]): The inputs to be transformed.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the transformation. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The transformed result.

    Examples:
        >>> text = "Hello, world!"
        >>> result = transform(engine, text, prompt="Translate to French")
        >>> print(result)
        Bonjour, monde!

        >>> numbers = [1, 2, 3, 4, 5]
        >>> result = transform(engine, numbers, prompt="Calculate the sum and average")
        >>> print(result)
        {'sum': 15, 'average': 3.0}
    """
    raise NotImplementedError("Subclass must implement this method")
