from abc import ABC
import functools
import inspect
import threading
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    get_type_hints,
    runtime_checkable,
)
from annotated_types import Predicate
from attrs import define, field
from typingx import isinstancex, issubclassx
import pydantic, sqlalchemy, dataclasses, attr, typing

from tensacode._internal.misc import HasPostInit, call_with_applicable_args


@define
class decorator(callable, HasPostInit):
    """
    A base class for creating decorators that can modify the behavior of functions or methods.

    This class provides a structure for decorators to modify the behavior of the decorated function
    by providing `prologue` and `epilogue` methods that are executed before and after the decorated
    function, respectively. The `decorate` method is used to attach the decorator to a function.

    The `Decorator` class is designed to be subclassed to create custom decorators. Subclasses should implement
    the `prologue` and `epilogue` methods to define behavior to execute before and after the decorated function.
    The `decorate` method is not explicitly defined but is conceptually represented by the `__call__` method,
    which wraps the decorated function with the `prologue` and `epilogue` behavior.

    Attributes:
        prologue (Callable[..., None]): A callable that is executed before the decorated function. It should
                                        accept any arguments and keyword arguments and return a tuple of args
                                        and kwargs to pass to the decorated function.
        epilogue (Callable[..., None]): A callable that is executed after the decorated function. It should
                                        accept the return value of the decorated function followed by any
                                        arguments and keyword arguments the function was called with, and
                                        return the (potentially modified) return value.
        fn (Callable[..., Any]): The function that is being decorated. This attribute is set when the decorator
                                 is called.

    Example Usage:
        class MyDecorator(Decorator):
            def prologue(self, *args, **kwargs):
                print("Before the decorated function")
                return args, kwargs

            def epilogue(self, result, *args, **kwargs):
                print("After the decorated function")
                return result

        @MyDecorator()
        def my_function(x, y):
            print(f"Function body with x={x} and y={y}")
            return x + y

        my_function(1, 2)
    """

    prologue: Callable[..., None] = field(init=True, default=lambda *a, **kw: (a, kw))
    epilogue: Callable[..., None] = field(
        init=True, default=lambda result, *a, **kw: result
    )
    fn: Callable[..., Any] = field(init=False)

    _built = False

    def __call__(self, *args, **kwargs):
        if not self._built:
            decorated_fn = self._build(*args, **kwargs)
            return decorated_fn
        else:
            return self._run(*args, **kwargs)

    def _run(self, *args, **kwargs):

        if self.prologue:
            changes = self.prologue(*args, **kwargs)

            # apply changes to args and kwargs
            if isinstancex(changes, (tuple, dict)):
                if isinstancex(changes, tuple[tuple[Any], dict[str, Any]]):
                    arg_updates, kwarg_updates = changes
                elif isinstance(changes, dict):
                    arg_updates, kwarg_updates = [], changes
                else:
                    raise ValueError(
                        "Prologue must return a tuple of (args, kwargs) or a kwargs dict"
                    )

                # bind args and kwargs to signature
                sig = inspect.signature(self.fn)
                bound_args = sig.bind_partial(*args, **kwargs)

                # arg_updates contains the positional arguments to update
                for i, argv in enumerate(arg_updates):
                    if i < len(bound_args.args):
                        bound_args.arguments[i] = argv

                # kwarg_updates contains the keyword arguments to update
                for kwarg_name, kwarg_value in kwarg_updates.items():
                    if kwarg_name in bound_args.kwargs:
                        bound_args.arguments[kwarg_name] = kwarg_value

                bound_args.apply_defaults()

                args = bound_args.args
                kwargs = bound_args.kwargs

            else:
                raise ValueError(
                    "Prologue must return the updated posargs and/or the names of specific parameters to modify"
                )

        result = self.fn(*args, **kwargs)

        if self.epilogue:
            updated_result = self.epilogue(result, *args, **kwargs)

            if updated_result:
                # if the epilogue returns a value, use it as the return value
                result = updated_result

        return result

    def _build(self, fn):
        self.fn = fn
        self._built = True


class overloaded(decorator):
    """
    A decorator that allows multiple versions of a function to be defined,
    each with different behavior based on specified conditions.

    The `overloaded` decorator should be used to decorate a base function.
    Additional versions of the function can be defined using the `.overload`
    method of the decorated function. Each overloaded version has an associated
    condition that accepts any or all of the base function's arguments and
    returns a boolean. When the decorated function is called, `overloaded` checks
    each condition in the reverse order that the overloads were defined in. This means that if
    multiple conditions are met, the most recent overload to be registered whose condition evaluates to
    True will be executed. If no conditions are met, defaults to the base function.

    Attributes:
    overloads (list): A list of tuples, each containing a condition function
                      and the associated overloaded function.

    Example:
        @overloaded
        def my_fn(a, b, c):
            print("base function")

        @my_fn.overload(lambda a, b, c: a == b and b == c)
        def _(a, b, c):
            print("overloaded function: a=b=c")

        (lambda b, c: b + c == 2)
        def _(a, b, c):
            print("overloaded function: b+c=2")

        # Test calls
        print(my_fn(1, 2, 3))  # Dispatches the base function
        print(my_fn(3, 3, 3))  # Dispatches the 1st overload
        print(my_fn(2, 1, 1))  # Dispatches the 2nd overload

    TL;DR:
    - The base function is decorated normally.
    - Overloads are defined using '@<function_name>.overload(<condition>)'.
    - The order of overload definitions matters. The first overload to match
      its condition is the one that gets executed.
    - If no overload conditions are met, the base function is executed.
    """

    fn: Callable

    def __init__(self, fn, /):
        super().__init__()
        self.fn = fn
        self.__call__ = functools.wraps(self.fn)(self.__call__)

    @define
    class overload(decorator):
        condition: Predicate = field()
        transform: Optional[Callable] = field()

        def prologue(self, *args, **kwargs):
            if self.transform:
                return call_with_applicable_args(self.transform, args, kwargs)
            return args, kwargs

    overloads: list[overload] = []

    def _build(self, fn):
        default_case_fn = fn

        @functools.wraps(fn)
        def fn(*args, **kwargs):
            for overload in reversed(self.overloads):
                if overload.condition(*args, **kwargs):
                    return call_with_applicable_args(overload.fn, args, kwargs)
            else:
                return call_with_applicable_args(default_case_fn, args, kwargs)

        return super._build(fn)
