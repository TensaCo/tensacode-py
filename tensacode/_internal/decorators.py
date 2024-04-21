from abc import ABC
import functools
import inspect
from typing import Annotated, Any, Callable, Generic, Optional, TypeVar, get_type_hints, runtime_checkable
from attrs import define, field
from typingx import isinstancex, issubclassx
import pydantic, sqlalchemy, dataclasses, attr, typing



@define
class Decorator(callable):
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

            def epilogue(self, retval, *args, **kwargs):
                print("After the decorated function")
                return retval

        @MyDecorator()
        def my_function(x, y):
            print(f"Function body with x={x} and y={y}")
            return x + y

        my_function(1, 2)
    """
    
    
    prologue: Callable[..., None] = field(init=True, default=lambda *a, **kw: (a, kw))
    epilogue: Callable[..., None] = field(init=True, default=lambda retval, *a, **kw: retval)
    fn: Callable[..., Any] = field(init=False)

    def __call__(self, fn):
        self.fn = fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if self.prologue:
                changes = self.prologue(*args, **kwargs)
                
                # apply changes to args and kwargs
                if isinstancex(changes, (tuple, dict)):
                    if isinstancex(changes, tuple[tuple[Any], dict[str, Any]]):
                        arg_updates, kwarg_updates = changes
                    elif isinstance(changes, dict):
                        arg_updates, kwarg_updates = [], changes
                    else:
                        raise ValueError("Prologue must return a tuple of (args, kwargs) or a kwargs dict")
                    
                    # bind args and kwargs to signature
                    sig = inspect.signature(fn)
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
                    raise ValueError("Prologue must return the updated posargs and/or the names of specific parameters to modify")
            
            retval = fn(*args, **kwargs)
            
            if self.epilogue:
                updated_retval = self.epilogue(retval, *args, **kwargs)
                
                if updated_retval:
                    # if the epilogue returns a value, use it as the return value
                    retval = updated_retval
                
            return retval
        
        wrapper.__annotations__.get('decorators', []).append(self)
        return wrapper


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

