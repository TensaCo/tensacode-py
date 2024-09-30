from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base_engine import Engine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


@Engine.register_op()
def convert(
    engine: Engine,
    /,
    origin_value: Any,
    target_type: type[Any],
    prompt: Optional[Encoded[str]] = None,
    modify_rounds=2,
    **kwargs: Any,
) -> Any:
    """
    Convert a value from its original type to a target type.

    This operation uses the engine to encode the original value, decode it to the target type,
    and then modify it for a specified number of rounds.

    Args:
        engine (Engine): The engine used for conversion.
        origin_value (Any): The original value to be converted.
        target_type (type[Any]): The target type to convert the value to.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the conversion. Defaults to None.
        modify_rounds (int, optional): Number of modification rounds. Defaults to 2.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The converted value of the target type.

    Examples:
        >>> original = "forty two"
        >>> result = convert(engine, original, int)
        >>> print(result)
        42

        >>> original = {"name": "Alice", "age": "30"}
        >>> class Person:
        ...     def __init__(self, name: str, age: int):
        ...         self.name = name
        ...         self.age = age
        >>> result = convert(engine, original, Person)
        >>> print(f"{result.name}, {result.age}")
        Alice, 30
    """
    origin_latent = engine.encode(origin_value, **kwargs)
    target_value = engine.decode(latent=origin_latent, type=target_type, **kwargs)
    for _ in range(modify_rounds):
        target_value = engine.modify(target_value, origin=origin_latent, **kwargs)
    return target_value
