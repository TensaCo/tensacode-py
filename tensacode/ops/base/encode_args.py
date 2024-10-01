from dataclasses import dataclass, field
from typing import Any, TypeVar, Callable, Union, Annotated
from functools import wraps
import inspect
from tensacode.internal.param_tags import EncodeTag
from tensacode.internal.latent import LatentType
from tensacode.core.base.engine import Engine


def encode_args(engine: Engine, *default_encode_args: Any, latent: Optional[LatentType] = None, **default_encode_kwargs: Any):
    """
    Decorator that automatically encodes function arguments based on type annotations.

    This decorator processes function arguments, encoding them using the provided engine
    if they are annotated with the Encoded type or EncodeTag. It supports both positional
    and keyword arguments, as well as variable-length argument lists.

    Args:
        engine (Engine): The engine used for encoding arguments.
        *default_encode_args: Default positional arguments passed to the engine's encode method.
        **default_encode_kwargs: Default keyword arguments passed to the engine's encode method.

    Returns:
        Callable: A decorator function that wraps the original function with argument encoding logic.

    Examples:
        >>> @engine.encode_args()
        ... def greet(name: Encoded[str], age: int):
        ...     return f"Hello, {name}! You are {age} years old."
        >>> result = greet("Alice", 30)
        >>> print(result)
        Hello, <encoded_value>! You are 30 years old.

        >>> @engine.encode_args(model="gpt-4")
        ... def summarize(text: Encoded[str], max_length: int = 100):
        ...     return f"Summary of '{text}' with max length {max_length}"
        >>> result = summarize("Long article about AI...", max_length=50)
        >>> print(result)
        Summary of '<encoded_value>' with max length 50
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            annotations = inspect.get_annotations(func)

            def encode_if_needed(arg_name, arg_value):
                if arg_name in annotations:
                    annotation = annotations[arg_name]
                    metadata = getattr(annotation, "__metadata__", ())
                    for item in metadata:
                        if isinstance(item, EncodeTag):
                            return engine.encode(
                                arg_value,
                                *item.encode_args,
                                **item.encode_kwargs,
                            )
                        elif issubclass(item, EncodeTag):
                            return engine.encode(
                                arg_value,
                                *default_encode_args,
                                **default_encode_kwargs,
                            )
                        elif (
                            isinstance(item, EncodeTag)
                            and not item.encode_args
                            and not item.encode_kwargs
                        ):
                            return engine.encode(
                                arg_value,
                                *default_encode_args,
                                **default_encode_kwargs,
                            )
                return engine.encode(
                    arg_value,
                    *default_encode_args,
                    latent=bound_args.arguments.get('latent', None),
                    **default_encode_kwargs,
                )

            encoded_positional = []
            encoded_keyword = {}
            var_positional = ()
            var_keyword = {}

            for name, param in sig.parameters.items():
                if name in bound_args.arguments:
                    value = encode_if_needed(name, bound_args.arguments[name])
                    match param.kind:
                        case inspect.Parameter.POSITIONAL_ONLY:
                            encoded_positional.append(value)
                        case inspect.Parameter.POSITIONAL_OR_KEYWORD:
                            if len(encoded_positional) < len(args):
                                encoded_positional.append(value)
                            else:
                                encoded_keyword[name] = value
                        case inspect.Parameter.KEYWORD_ONLY:
                            encoded_keyword[name] = value
                        case inspect.Parameter.VAR_POSITIONAL:
                            var_positional = (
                                value if isinstance(value, tuple) else (value,)
                            )
                        case inspect.Parameter.VAR_KEYWORD:
                            var_keyword = value if isinstance(value, dict) else {}

            return func(
                *encoded_positional, *var_positional, **encoded_keyword, **var_keyword
            )

        return wrapper

    return decorator
