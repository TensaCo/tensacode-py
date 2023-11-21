import types
from typing import Any, Type
from context_managed_singleton import ContextManagedSingleton


class ContextManagedSingletonProxy:
    """Wraps an object and allows dot notation get/set the wrapped object's attributes.

    Example:
    >>> builtins = DotNotationProxyWrapper(__builtins__)
    >>> builtins.print("Hello, world!")
    ... Hello, world!
    """

    __obj: ContextManagedSingleton

    def __init__(self, obj: ContextManagedSingleton):
        self.__obj = obj

    def __getattribute__(self, __name: str) -> Any:
        # maybe return a hidden attribute (eg, from this class)
        if __name.startswith(_HIDDEN_ATTR_NAME_PREFIX) or (
            __name.startswith("__") and __name.endswith("__")
        ):
            return super().__getattribute__(__name)
        # get the attribute from the wrapped object
        return getattr(self.__obj.current, __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__obj.current.__setattr__(__name, __value)


_HIDDEN_ATTR_NAME_PREFIX = f"_{ContextManagedSingletonProxy.__name__}__"


class LazyContextManagedSingletonProxy(ContextManagedSingletonProxy):
    """Initializes the wrapped type when it is first accessed."""

    __type: Type
    __args: tuple
    __kwargs: dict
    __obj: ContextManagedSingleton = None

    def __init__(self, type: Type, args: tuple, kwargs: dict):
        self.__type = type
        self.__args = args
        self.__kwargs = kwargs

    def __getattribute__(self, __name: str) -> Any:
        # maybe return a hidden attribute (eg, from this class)
        if __name.startswith(_HIDDEN_ATTR_NAME_PREFIX) or (
            __name.startswith("__") and __name.endswith("__")
        ):
            return super().__getattribute__(__name)
        # maybe initialize the wrapped object
        if self.__obj is None:
            self.__obj = ContextManagedSingleton(
                self.__type(*self.__args, **self.__kwargs)
            )
        # get the attribute from the wrapped object
        return getattr(self.__obj, __name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__obj.__setattr__(__name, __value)
