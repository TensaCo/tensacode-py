from dataclasses import dataclass, field
from typing import Any, TypeVar, Callable, Union, Annotated
from functools import wraps
import inspect

from tensacode.internal.param_tags import AutofillTag
from tensacode.internal.latent import LatentType
from tensacode.core.base_engine import Engine


def autofill_args(
    engine: Engine, *default_autofill_args, **default_autofill_kwargs
):
    """
    Decorator that automatically fills missing function arguments using query and convert operations.

    This decorator processes function arguments, querying for missing values and converting them
    to the appropriate type based on the function's type annotations.

    Args:
        engine (Engine): The engine used for querying and converting arguments.
        *default_autofill_args: Default positional arguments passed to the engine's query method.
        **default_autofill_kwargs: Default keyword arguments passed to the engine's query method.

    Returns:
        Callable: A decorator function that wraps the original function with argument autofilling logic.

    Examples:
        >>> @engine.autofill_args()
        ... def greet(name: Autofilled[str], age: int):
        ...     return f"Hello, {name}! You are {age} years old."
        >>> result = greet(age=30)
        >>> print(result)
        Hello, Alice! You are 30 years old.

        >>> @engine.autofill_args(context="user profile")
        ... def create_profile(name: Autofilled[str], age: Autofilled[int], city: Autofilled[str] = "Unknown"):
        ...     return f"Profile: {name}, {age} years old, from {city}"
        >>> result = create_profile()
        >>> print(result)
        Profile: John Doe, 25 years old, from New York
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()
            annotations = inspect.get_annotations(func)

            all_args = {}
            for i, arg in enumerate(args):
                all_args[f"_{i}"] = arg
            all_args.update(kwargs)

            def autofill_if_needed(arg_name, param):
                if arg_name not in all_args:
                    query_kwargs = {
                        **default_autofill_kwargs,
                        "context": all_args,
                        "query": f"Provide a value for the '{arg_name}' argument",
                    }
                    query_result = engine.query(*default_autofill_args, **query_kwargs)
                    if param.annotation != inspect.Parameter.empty:
                        return engine.convert(query_result, param.annotation)
                    return query_result
                return all_args[arg_name]

            autofilled_args = []
            autofilled_kwargs = {}
            for name, param in sig.parameters.items():
                value = autofill_if_needed(name, param)
                if name in kwargs or (name.startswith("_") and name[1:].isdigit()):
                    autofilled_args.append(value)
                else:
                    autofilled_kwargs[name] = value

            return func(*autofilled_args, **autofilled_kwargs)

        return wrapper

    return decorator
