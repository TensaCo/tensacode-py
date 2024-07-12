"""
Dependency system for instance-aware default factories.

Example:
    ```
    >>> @InjectDefaults
        class MyClass:
            default_a = 0
        
            @inject
            def hello1(self, a: Annotated[int, depends(lambda self, *a, **kw: self.default_a)]):
                print(a)
                
            @inject
            def hello2(self, a: int = depends(lambda self, *a, **kw: self.default_a)):
                print(a)


    >>> m = MyClass()
    >>> m.hello1()
    ... 0
    >>> m.hello2()
    ... 0
    >>> m.hello1(1)
    ... 1
    >>> m.hello2(2)
    ... 2
    ```
"""

from abc import ABC
from attrs import define, field

from typingx import isinstancex


T_class = TypeVar("T_class")
T_argval = TypeVar("T_argval")


class factory_function(Generic[T_class, T_argval], runtime_checkable):
    # may take in all the args supplied to the function
    def __call__(self: T_class, *a, **kw) -> T_argval:
        raise NotImplementedError


class depends(Generic[T_class, T_argval], callable):
    # may take in all the args supplied to the function
    factory: factory_function[T_class, T_argval]

    def __init__(self, factory: factory_function[T_class, T_argval]):
        self.factory = factory


class InjectDefaultsBase(ABC):
    def __getattribute__(self, __name: str) -> Any:
        val = super().__getattribute__(__name)
        if any(isinstancex(x, inject) for x in get_args(val)):
            assert (
                getattr(val, "object_instance", self) is self
            ), "Decorator must be called on the same instance but it has already been called on a different instance."
            val.object_instance = self
            # now the instance is set, so we can use it in the factory :-)
        return val


def InjectDefaults(cls: type):
    # gets in front of the getattr method to make sure
    # a copy of the class is sent to each Depend invocation.
    if InjectDefaultsBase in cls.__bases__:
        # remove InjectDefaultsBase from the bases
        cls.__bases__ = tuple(b for b in cls.__bases__ if b != InjectDefaultsBase)
    cls.__bases__ = (InjectDefaultsBase,) + cls.__bases__
    return cls


import inspect
from typing import (
    Annotated,
    Any,
    Generic,
    TypeVar,
    get_origin,
    get_args,
    runtime_checkable,
)

from tensacode._internal.decorators import decorator


@define
class inject(Generic[T_class], decorator):
    object_instance: T_class = field(
        init=False
    )  # set by InjectDefaultsBase.__getattribute__ prior to calling the function

    def prologue(self, *a, **kw):
        if self.object_instance is None:
            raise RuntimeError(
                "`object_instance` is None. Did you remember to wrap the class with @InjectDefaults?"
            )

        # Retrieve the signature of the function to be decorated.
        sig = inspect.signature(self.fn)
        # Bind the provided arguments to the parameters of the function signature.
        # This does not require all arguments to be provided (partial binding).
        bound_args = sig.bind_partial(*a, **kw)
        # Apply default values for any parameters that were not provided in the call.
        bound_args.apply_defaults()

        # Iterate through the bound arguments.
        for name, value in bound_args.arguments.items():
            # If the argument value is a generic type that originates from `depends`,
            # replace it with the result of the factory method specified in its type arguments.
            if get_origin(value) is Annotated:
                _, *metadata = get_args(value)
                idx = next(
                    (i for i, x in enumerate(metadata) if isinstance(x, depends)), None
                )
                if idx is not None:
                    depends_instance = metadata[idx]
                    bound_args.arguments[name] = depends_instance.factory(
                        self.object_instance, *a, **kw
                    )
                    continue  # Annotations take precedence over argval depends, but only continue AFTER you found the factory
            # If the argument value is an instance of `depends`, replace it with the result of its factory method.
            if isinstance(value, depends):
                bound_args.arguments[name] = value.factory(
                    self.object_instance, *a, **kw
                )

        # Return the modified arguments and keyword arguments for further processing.
        return bound_args.args, bound_args.kwargs
