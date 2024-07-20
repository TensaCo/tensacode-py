from functools import wraps, cache
from typing import Callable, Any, TYPE_CHECKING
import inspect
import threading


def call_with_appropriate_args(fn, *args, **kwargs):
    """
    Call a function with only the arguments it can accept.

    This function inspects the signature of the given function and calls it with
    only the arguments that match its parameters. It filters out any excess
    arguments that are not part of the function's signature.

    Args:
        fn (callable): The function to be called.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.

    Returns:
        The result of calling the function with the filtered arguments.

    Example:
        def example_func(a, b):
            return a + b

        result = call_with_appropriate_args(example_func, a=1, b=2, c=3)
        # result will be 3, and 'c' will be ignored
    """
    sig = inspect.signature(fn)
    bound_args = sig.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()

    # Filter out excess arguments
    filtered_args = {
        k: v for k, v in bound_args.arguments.items() if k in sig.parameters
    }
    return fn(**filtered_args)


def polymorphic(fn):
    """
    A decorator for creating polymorphic functions.

    This decorator allows you to define a base function and register multiple
    implementations for different conditions. When the decorated function is called,
    it will execute the appropriate implementation based on the registered conditions.

    The decorator adds a 'register' method to the wrapped function, which can be used
    to register new implementations with their corresponding condition functions.

    Args:
        fn (callable): The base function to be decorated.

    Returns:
        callable: A wrapper function that handles the polymorphic behavior.

    Example:
        @polymorphic
        def process(obj):
            return "Default processing"

        @process.register(lambda obj: isinstance(obj, int))
        def process_int(obj):
            return f"Processing integer: {obj}"

        @process.register(lambda obj: isinstance(obj, str))
        def process_str(obj):
            return f"Processing string: {obj}"

        @process.register(lambda obj: isinstance(obj, list))
        def process_list(obj):
            return f"Processing list: {obj}"

        print(process(10))  # Output: "Processing integer: 10"
        print(process("hello"))  # Output: "Processing string: hello"
        print(process([1, 2, 3]))  # Output: "Default processing"
    """
    Override = tuple[int, Callable, Callable]
    overrides: list[Override] = []

    class PolymorphicDecorator:
        # just for typing
        overrides: list[Override]

        def register(self, condition_fn: Callable, /, priority: int = 0):
            def decorator(override_fn):
                overrides.append(Override(priority, condition_fn, override_fn))
                return override_fn

            return decorator

        __call__: Callable[..., Any]

    @cache
    def overrides_sorted_by_priority(overrides):
        # highest priority overrides are ordered at the beginning of the list.
        return sorted(overrides, key=lambda x: x[0], reverse=True)

    @wraps(fn, updated=["__annotations__"])
    def wrapper(cls, *args, **kwargs):
        for _, condition_fn, override_fn in overrides_sorted_by_priority(overrides):
            if call_with_appropriate_args(condition_fn, cls, *args, **kwargs):
                return override_fn(cls, *args, **kwargs)
        return fn(cls, *args, **kwargs)

    setattr(wrapper, "overrides", overrides)

    def register(condition_fn, /, priority: int = 0):
        def decorator(override_fn):
            overrides.append(Override(priority, condition_fn, override_fn))
            return override_fn

        return decorator

    setattr(wrapper, "register", register)

    typed_wrapper: PolymorphicDecorator = wrapper
    return typed_wrapper


def cached_with_key(key_func=lambda input: input):
    """
    A decorator that caches the result of a method based on a key function.

    This decorator is thread-safe and caches the result of the decorated method.
    The cache is invalidated when the key returned by key_func changes.

    Args:
        key_func (callable): A function that returns a cache key for the instance.

    Returns:
        callable: A decorator function.

    Example:
        >>> import time
        >>> class Example:
        ...     def __init__(self):
        ...         self.value = 0
        ...
        ...     @cached_with_key(lambda self: self.value)
        ...     def expensive_operation(self):
        ...         time.sleep(0.1)  # Simulate expensive operation
        ...         return f"Result: {self.value}"
        ...
        >>> obj = Example()
        >>> start = time.time()
        >>> print(obj.expensive_operation)
        Result: 0
        >>> print(f"Time taken: {time.time() - start:.2f} seconds")
        Time taken: 0.10 seconds
        >>> start = time.time()
        >>> print(obj.expensive_operation)
        Result: 0
        >>> print(f"Time taken: {time.time() - start:.2f} seconds")
        Time taken: 0.00 seconds
        >>> obj.value = 1
        >>> start = time.time()
        >>> print(obj.expensive_operation)
        Result: 1
        >>> print(f"Time taken: {time.time() - start:.2f} seconds")
        Time taken: 0.10 seconds
    """

    def decorator(func):
        cache_name = f"_cached_{func.__name__}"
        key_name = f"_cached_key_{func.__name__}"
        lock_name = f"_lock_{func.__name__}"

        @wraps(func)
        def wrapper(self):
            if not hasattr(self, lock_name):
                setattr(self, lock_name, threading.Lock())

            with getattr(self, lock_name):
                if not hasattr(self, cache_name) or getattr(self, key_name) != key_func(
                    self
                ):
                    setattr(self, cache_name, func(self))
                    setattr(self, key_name, key_func(self))
                return getattr(self, cache_name)

        return property(wrapper)

    return decorator
