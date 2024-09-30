from tensacode.core.base_engine import Engine
# ... existing imports ...

@Engine.register_op_on_class()
def transform(
    engine: Engine,
    *inputs: list[Any],
    prompt: Optional[Encoded[str]] = None,
    output_type: type = str,
    **kwargs: Any,
) -> Any:
    """
    Transform the input(s) into a new form using LLM-based methods.

    This implementation utilizes the engine's language model to apply a transformation
    described by the prompt to the inputs.

    Args:
        engine (Engine): The engine used for transformation.
        *inputs (list[Any]): The inputs to be transformed.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the transformation.
        output_type (type, optional): The desired output type. Defaults to str.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The transformed result.
    """
    # Concatenate inputs into a single text
    input_text = '\n'.join(str(input_item) for input_item in inputs)
    # Ensure a prompt is provided
    if not prompt:
        raise ValueError("A prompt is required for transformation.")
    # Combine prompt and inputs
    full_prompt = f"{prompt}\n\n{input_text}"
    # Use the LLM to perform the transformation
    transformed_output = engine.llm.generate(text=full_prompt, **kwargs)
    # Optionally convert to the desired output type
    if output_type != str:
        transformed_output = engine.convert(transformed_output, output_type)
    return transformed_output