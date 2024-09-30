from tensacode.core.base_engine import Engine
# ... existing imports ...

@Engine.register_op_on_class()
def program(
    engine: Engine,
    prompt: Optional[Encoded[str]] = None,
    language: str = "python",
    **kwargs: Any,
) -> str:
    """
    Generate a program based on the provided prompt using LLM-based methods.

    This implementation uses the engine's language model to create code in the specified language.

    Args:
        engine (Engine): The engine used for program generation.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the program generation.
        language (str, optional): The programming language for the generated code. Defaults to "python".
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        str: The generated program code.
    """
    # Ensure a prompt is provided
    if not prompt:
        raise ValueError("A prompt is required for program generation.")
    # Build the full prompt
    program_prompt = f"Write a {language} program that does the following:\n{prompt}"
    # Use the LLM to generate the program
    code = engine.llm.generate(text=program_prompt, **kwargs)
    return code