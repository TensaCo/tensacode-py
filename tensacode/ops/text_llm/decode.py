from tensacode.core.base_engine import Engine
# ... existing imports ...

@Engine.register_op_on_class(score_fn=score_node_inheritance_distance(type_=AtomicValueNode))
def decode_atomic(
    engine: Engine,
    /,
    type_: type[Any] = Any,
    latent: LatentType = None,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Decode a latent representation into an atomic value of the specified type using LLM-based methods.

    Args:
        engine (Engine): The engine used for decoding.
        type_ (type[Any], optional): The target type for decoding.
        latent (LatentType, optional): The latent representation to decode.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the decoding process.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The decoded atomic value.
    """
    # Ensure latent is provided
    if latent is None:
        raise ValueError("A latent representation is required for decoding.")
    # If a prompt is provided, use it to guide the decoding
    if prompt:
        decode_prompt = f"{prompt}\n\nDecode the following latent representation into {type_.__name__}: {latent}"
    else:
        decode_prompt = f"Decode the following latent representation into {type_.__name__}: {latent}"
    # Use the LLM to generate the decoded value
    decoded_text = engine.llm.generate(text=decode_prompt, **kwargs)
    # Convert the decoded text to the specified type
    decoded_value = engine.convert(decoded_text, type_)
    return decoded_value