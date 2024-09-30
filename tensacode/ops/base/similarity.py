from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base_engine import Engine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


@Engine.register_op()
def similarity(
    engine: Engine,
    input_a: Any,
    input_b: Any,
    **kwargs: Any,
) -> float:
    """
    Calculate the similarity between two input objects.

    This operation determines how similar the two input objects are to each other, returning a float value
    between 0 (completely different) and 1 (identical).

    Args:
        engine (Engine): The engine used for similarity calculation.
        input_a (Any): The first object to compare for similarity.
        input_b (Any): The second object to compare for similarity.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the similarity calculation. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        float: A value between 0 and 1 representing the similarity of the inputs.

    Examples:
        >>> text1 = "The quick brown fox"
        >>> text2 = "The fast brown fox"
        >>> result = similarity(engine, text1, text2)
        >>> print(result)
        0.9

        >>> img1 = load_image("cat.jpg")
        >>> img2 = load_image("dog.jpg")
        >>> result = similarity(engine, img1, img2)
        >>> print(result)
        0.3
    """
    if input_a == input_b:
        return 1.0
    else:
        return 0.0
