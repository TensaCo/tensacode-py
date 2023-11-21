import inspect
from typing import Generator, Literal


def call_with_applicable_args(func, args_list, kwargs_dict):
    """
    Calls the given function 'func' using as many arguments from 'args_list' and
    'kwargs_dict' as possible based on the function's signature.

    The function attempts to match the provided positional and keyword arguments
    with the parameters of 'func'. It respects the function's requirements for
    positional-only, keyword-only, and variable arguments. Extra arguments are
    ignored if they do not fit the function's signature.

    Parameters:
    func (Callable): The function to be called.
    args_list (list): A list of positional arguments to try to pass to 'func'.
    kwargs_dict (dict): A dictionary of keyword arguments to try to pass to 'func'.

    Returns:
    The return value of 'func' called with the applicable arguments from 'args_list'
    and 'kwargs_dict'.
    """

    sig = inspect.signature(func)
    bound_args = {}

    # Create a mutable copy of args_list
    args = list(args_list)

    for param_name, param in sig.parameters.items():
        # Handle positional and keyword arguments
        if args and param.kind in [
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ]:
            bound_args[param_name] = args.pop(0)
        elif param_name in kwargs_dict and param.kind in [
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ]:
            bound_args[param_name] = kwargs_dict[param_name]

        # Handle variable arguments
        if param.kind == inspect.Parameter.VAR_POSITIONAL and args:
            bound_args[param_name] = args
            args = []
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            bound_args.update(kwargs_dict)

    return func(**bound_args)


def get_keys(
    object,
    visibility: Literal["public", "protected", "private"] = "public",
    inherited_members=True,
) -> Generator[str, None, None]:
    keys = dir(object)

    if isinstance(object, type) and inherited_members is False:
        for base in object.__bases__:
            keys = [k for k in keys if k not in dir(base)]

    for k in keys:
        match visibility:
            case "public":
                if k.startswith("_"):
                    continue
            case "protected":
                if k.startswith("__"):
                    continue
            case "private":
                pass
            case _:
                raise ValueError(f"Invalid visibility: {visibility}")
        yield k


def try_(lmbda, *args, **kwargs):
    try:
        return lmbda(*args, **kwargs)
    except:
        return None
