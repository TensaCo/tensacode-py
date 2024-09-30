from tensacode.core.base_engine import Engine
# ... existing imports ...

@Engine.register_op_on_class()
def encode(
    engine: Engine,
    *inputs: list[Any],
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> LatentType:
    """
    Encode one or more inputs into a latent representation using LLM-based methods.

    This implementation uses the engine's language model to generate embeddings or latent representations
    of the input data. It concatenates the inputs into a textual form and processes them through the LLM.

    Args:
        engine (Engine): The engine used for encoding.
        *inputs (list[Any]): The input objects to be encoded.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the encoding process.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        LatentType: The resulting latent representation.
    """
    # Convert inputs to strings and concatenate
    input_text = ' '.join(str(input_item) for input_item in inputs)
    # If a prompt is provided, prepend it
    if prompt:
        input_text = f"{prompt}\n\n{input_text}"
    # Use the engine's language model to encode the input text
    latent = engine.llm.encode(input_text, **kwargs)
    return latent