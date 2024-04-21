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


def inline_try(_lambda, /, *args, **kwargs):
    try:
        return _lambda(*args, **kwargs)
    except:
        return None

_DEFAULT_GLOBAL_NAMESPACE = {}


def unique(name: str, namespace=_DEFAULT_GLOBAL_NAMESPACE, scope_delimiter="/"):
    if name not in namespace:
        namespace[name] = 0
        return name
    else:
        namespace[name] += 1
        return name + scope_delimiter + str(namespace[name])
