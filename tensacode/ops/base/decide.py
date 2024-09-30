from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base_engine import Engine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op
from tensacode.core.base.ops.decode_op import DecodeOp


@Engine.register_op_on_class()
def decide(
    engine: Engine,
    latent: LatentType,
    *inputs: list[Any],
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> bool:
    """
    Make a boolean decision based on the given latent representation and inputs.

    This operation uses the engine to decode a latent representation into a boolean decision,
    optionally considering additional inputs and a guiding prompt.

    Args:
        engine (Engine): The engine used for decision-making.
        latent (LatentType): The latent representation to be used for making the decision.
        *inputs (list[Any]): Additional inputs that may influence the decision.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the decision-making process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        bool: The boolean decision.

    Examples:
        >>> latent = engine.encode("Is it raining?")
        >>> weather_data = {"temperature": 15, "humidity": 80, "cloud_cover": "overcast"}
        >>> result = decide(engine, latent, weather_data)
        >>> print(result)
        False
    """
    latent = engine.transform(latent)
    return engine.decode(latent=latent, type=bool, prompt=prompt, **kwargs)
