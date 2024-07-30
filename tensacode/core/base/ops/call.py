import inspect
from typing import Any, Callable, ClassVar
from typing_extensions import Self

from tensacode.core.base.base_engine import BaseEngine
from tensacode.internal.latent import LatentType
from tensacode.core.base.ops.base_op import Op


@BaseEngine.register_op()
def call(
    engine: BaseEngine,
    func: Callable,
    prompt: Optional[Encoded[str]] = None,
    **kwargs: Any,
) -> Any:
    """
    Call a function with arguments determined by the engine.

    This operation uses the engine to determine appropriate arguments for the given function
    and then calls the function with those arguments.

    Args:
        engine (BaseEngine): The engine used to determine function arguments.
        func (Callable): The function to be called.
        prompt (Optional[Encoded[str]], optional): A prompt to guide the engine. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the engine.

    Returns:
        Any: The result of calling the function with the determined arguments.

    Examples:
        >>> def greet(name: str, age: int):
        ...     return f"Hello, {name}! You are {age} years old."
        >>> result = call(engine, greet)
        >>> print(result)
        Hello, Alice! You are 30 years old.

        >>> def add(a: int, b: int):
        ...     return a + b
        >>> result = call(engine, add)
        >>> print(result)
        7
    """
    # Get the function signature
    signature = inspect.signature(func)

    # Prepare arguments for the function call
    all_args = {}
    arg_count = 0

    for param_name, param in signature.parameters.items():
        engine.info(**{param_name: param})
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # Handle *args
            var_args = engine.query_or_create(
                query=f"Provide values for *{param_name}",
                type=(
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else None
                ),
            )
            for arg in var_args:
                all_args[str(arg_count)] = arg
                arg_count += 1
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            # Handle **kwargs
            var_kwargs = engine.query_or_create(
                query=f"Provide key-value pairs for **{param_name}",
                type=(
                    param.annotation
                    if param.annotation != inspect.Parameter.empty
                    else None
                ),
            )
            all_args.update(var_kwargs)
        else:
            # Handle regular parameters
            param_type = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else None
            )
            if param.default == inspect.Parameter.empty:
                value = engine.query_or_create(
                    query=f"Provide a value for parameter '{param_name}'",
                    type=param_type,
                )
            else:
                value = engine.query_or_create(
                    query=f"Provide a value for parameter '{param_name}' (default: {param.default})",
                    type=param_type,
                )

            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                all_args[str(arg_count)] = value
                arg_count += 1
            else:
                all_args[param_name] = value

    # Call the function with the prepared arguments
    args = [all_args[str(i)] for i in range(len(all_args))]
    kwargs = {k: v for k, v in all_args.items() if k not in args}
    return func(*args, **kwargs)
