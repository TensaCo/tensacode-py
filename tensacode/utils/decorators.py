import functools
import inspect
from typing import Annotated, Any, Callable, get_type_hints
from tensacode.utils.internal_types import Predicate

from tensacode.utils.misc import call_with_applicable_args
from typingx import isinstancex, issubclassx
import pydantic, sqlalchemy, dataclasses, attr, typing

py_object = object


class Default:
    get: Callable[..., Any]  # should take in all the args supplied to the function

    def __init__(self, get: Callable[..., Any]):
        self.get = get


class Decorator(callable):
    prologue: Callable[..., None] = lambda *a, **kw: (a, kw)
    epilogue: Callable[..., None] = lambda retval, *a, **kw: retval
    fn: Callable[..., Any] = None

    def __call__(self, fn):
        self.fn = fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            args, kwargs = self.prologue(*args, **kwargs)
            retval = fn(*args, **kwargs)
            retval = self.epilogue(retval, *args, **kwargs)
            return retval

        return wrapper


class dynamic_defaults(Decorator):
    default_type: type[Default] = Default

    def prologue(self, *a, **kw):
        # if any args are depends, replace them with their invocation result
        args = [arg() if isinstance(arg, self.default_type) else arg for arg in a]
        kwargs = {
            k: v() if isinstance(v, self.default_type) else v for k, v in kw.items()
        }
        return args, kwargs

    def epilogue(self, retval, *a, **kw):
        """if retval is annotated with a depends, replace it with its invocation result"""
        if retval:
            return retval  # this returns the retval if it is not annotated

        return_annotation = get_type_hints(self.fn).get("return")

        if isinstance(return_annotation, self.default_type):
            return retval()

        if isinstance(return_annotation, Annotated):
            for arg in return_annotation.__args__:
                if isinstance(arg, self.default_type):
                    return_annotation = arg()
                    return retval()

        return retval  # this returns None if retval is None and not annotated


class overloaded(Decorator):
    """
    A decorator that allows multiple versions of a function to be defined,
    each with different behavior based on specified conditions.

    The `overloaded` decorator should be used to decorate a base function.
    Additional versions of the function can be defined using the `.overload`
    method of the decorated function. Each overloaded version has an associated
    condition - a lambda or function that takes the same arguments as the base
    function and returns a boolean. When the decorated function is called,
    `overloaded` checks each condition in the order the overloads were defined.
    It calls and returns the result of the first overload whose condition
    evaluates to True. If no conditions are met, the base function is called.

    Attributes:
    overloads (list): A list of tuples, each containing a condition function
                      and the associated overloaded function.

    Example:
        @overloaded
        def my_fn(a, b, c):
            return "base function", a, b, c

        @my_fn.overload(lambda a, b, c: a == 3)
        def _overload(a, b, c):
            return "overloaded function", a, b, c

        # Test calls
        print(my_fn(1, 2, 3))  # Calls the base function
        print(my_fn(3, 2, 1))  # Calls the overloaded function

    Note:
    - The base function is decorated normally.
    - Overloads are defined using '@<function_name>.overload(<condition>)'.
    - The order of overload definitions matters. The first overload to match
      its condition is the one that gets executed.
    - If no overload conditions are met, the base function is executed.
    """

    overloads: tuple[Predicate, Callable | None, Callable] = []

    def __call__(self, fn):
        self.fn = fn
        self.base_fn = super().__call__(fn)
        return self._overload_dispatcher

    def _overload_dispatcher(self, *args, **kwargs):
        for condition, func in self.overloads:
            if call_with_applicable_args(condition, args, kwargs):
                return func(*args, **kwargs)
        return self.base_fn(*args, **kwargs)

    def overload(self, condition, transform=None):
        def decorator(overload):
            self.overloads.append((condition, transform, overload))
            return overload

        return decorator
