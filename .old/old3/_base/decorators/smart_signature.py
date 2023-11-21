from functools import wraps


def smart_signature(search_for_args=True):
    """Intelligently adjusts the arguments to a function based on its signature.

    Args:
        search_for_args (bool, optional): Whether to look for missing args by climbing the stack. Defaults to True.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # TODO: parse args and call fn
            pass

        return wrapper

    return decorator
