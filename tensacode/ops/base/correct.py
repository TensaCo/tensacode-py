from typing import Any, ClassVar
from typing_extensions import Self

from tensacode.core.base_engine import Engine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


@Engine.register_op_on_class()
def correct(
    engine: Engine,
    input: Any,
    correct_examples: list[Any],
    latent: Optional[LatentType] = None,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Correct an input value based on provided correct examples.

    This operation uses the engine to modify the input value to align it with the given correct examples.

    Args:
        engine (Engine): The engine used for correction.
        input (Any): The input value to be corrected.
        correct_examples (list[Any]): A list of correct examples to guide the correction process.
        latent (Optional[LatentType], optional): The latent type to use for correction. Defaults to None.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the correction. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The corrected value.

    Examples:
        >>> incorrect_text = "The quik brown fox jumps over the lasy dog."
        >>> correct_examples = ["The quick brown fox jumps over the lazy dog."]
        >>> result = correct(engine, incorrect_text, correct_examples)
        >>> print(result)
        The quick brown fox jumps over the lazy dog.

        >>> incorrect_data = {"name": "John", "age": "thirty"}
        >>> correct_examples = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
        >>> result = correct(engine, incorrect_data, correct_examples)
        >>> print(result)
        {'name': 'John', 'age': 30}
    """
    # Use the provided latent or encode the input
    input_latent = latent if latent is not None else engine.encode(input, **kwargs)
    
    # Encode the correct examples
    correct_latents = [engine.encode(example, **kwargs) for example in correct_examples]
    
    # Proceed with the correction using the latents
    corrected_latent = engine.modify(input_latent, target_latents=correct_latents, **kwargs)
    
    # Decode the corrected latent
    corrected_value = engine.decode(latent=corrected_latent, type=type(input), **kwargs)
    
    return corrected_value
