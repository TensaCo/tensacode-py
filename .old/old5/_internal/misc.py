from copy import copy
import inspect
from typing import Generator, Literal, TypeVar

from abc import ABC
from contextlib import contextmanager
import inspect
from threading import Lock
from typing import Any, ClassVar, Generic, Literal, Self
from uuid import uuid4
from attrs import define, field

T = TypeVar("T")


class HasPostInit(ABC):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.__post_init__()

    def __post_init__(self):
        pass


@define
class InstanceView:
    _instance_view__obj: T
    _instance_view__overrides: dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_instance_view__"):
            return super().__getattr__(name)
        if name in self._instance_view__overrides:
            return self._instance_view__overrides[name]
        return getattr(self._instance_view__obj, name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name.startswith("_instance_view__"):
            return super().__setattr__(__name, __value)
        if __name in self._instance_view__overrides:
            self._instance_view__overrides[__name] = __value
        else:
            setattr(self._instance_view__obj, __name, __value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_instance_view__"):
            return super().__delattr__(name)
        if name in self._instance_view__overrides:
            del self._instance_view__overrides[name]
        else:
            delattr(self._instance_view__obj, name)

    @classmethod
    def create(cls, obj: T, overrides: dict[str, Any]):
        return cls(_instance_view__obj=obj, _instance_view__overrides=overrides)


class HasDefault(ABC):
    _current_stack: ClassVar[list[Self]] = []

    @classmethod
    def get_current(cls) -> Self:
        return cls._current_stack[-1]

    @contextmanager
    def as_default(self):
        self._current_stack.append(self)
        yield
        self._current_stack.pop()


T_scope = TypeVar("T_scope")


@define
class Scoped(Generic[T_scope], ABC):
    _scope_lock = field(factory=Lock, init=False)
    _scope_stack: list[T_scope] = field(factory=list, init=False)

    @property
    def full_scope(self) -> list[T_scope]:
        with self._scope_lock:
            return self._scope_stack.copy()

    @property
    def current_scope(self) -> T_scope:
        with self._scope_lock:
            return self._scope_stack[-1]

    SCOPE_QUALPATH_SEPARATOR = "."

    @property
    def scope_qualpath(self) -> str:
        return self.SCOPE_QUALPATH_SEPARATOR.join(
            str(scope) for scope in self.current_action_scope
        )

    @contextmanager
    def scoped(self, next_scope: T_scope, keys_to_detach: list[str] = None) -> Self:
        if not keys_to_detach:
            keys_to_detach = []
        keys_to_detach.append("_scope_stack")
        scoped_self = InstanceView.create(
            obj=self,
            overrides={
                k: copy(v) for k, v in self.__dict__.items() if k in keys_to_detach
            },
        )
        with self._scope_lock:
            scoped_self._scope_stack.append(next_scope)
        yield scoped_self
        with self._scope_lock:
            _last_scope = scoped_self._scope_stack.pop()
            assert (
                _last_scope == next_scope
            ), f"Corrupted scope stack: stack imbalance.\nExpected: {next_scope}\nGot: {_last_scope}"


class Namespaced(Scoped[str]):
    @property
    def qualname(self) -> str:
        return self.full_scope

    @property
    def current_namespace(self) -> str:
        return self.current_scope

    @contextmanager
    def namespaced(self, name: str = None) -> Self:
        if not name:
            name = uuid4().hex
        with self.scoped(name):
            yield self


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
