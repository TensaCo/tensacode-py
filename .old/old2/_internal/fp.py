from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar, runtime_checkable
from functools import partial

_T = TypeVar("_T")
Number = TypeVar("Number", int, float)

@runtime_checkable
class MapFn(Protocol):
  def __call__(self, *args, **kwargs) -> _T:
    pass

class IdentityMapFn(MapFn):
  def __call__(self, x: _T, *args, **kwargs) -> _T:
    return x

class ConstFn(MapFn):
  def __init__(self, value: _T) -> None:
    super().__init__()
    self.value = value
  def __call__(self, *args, **kwargs) -> _T:
    return self.value

class ScoringFn(MapFn):
  def __call__(self, x: _T, *args, **kwargs) -> Number:
    pass

@runtime_checkable
class Predicate(Protocol):
  def __call__(self, *args, **kwargs) -> bool:
    pass

class AlwaysTruePredicate(Predicate):
  def __call__(self, *args, **kwargs) -> bool:
    return True

def make_decorator(dataclass):
    """ Turns dataclass into a decorator which copies all dataclass' attrs onto
    the sole supplied object during initialization.

    Use:
    1. Define the dataclass
    ```
    class named:
        name: str
    ```
    2. Decorate a dataclass-like with `make_decorator`
    ```
    @make_decorator
    class named:
        name: str
    ```
    3. Decorate objects with your decorator
    ```
    @named class Team: ...
    @named class Company:
        teams: list[Team]
        @named brand: any
    @named def command1(...): ...
    ```
    4. Access decorator attributes
    """
    def apply(source, target):
        # applies attrs from source onto target
        # copies default values if they exist
        # it should also copy superclass' keys and slots
        for attr in inspect.getmembers(source, predicate=X): # TODO
            pass
            # TODO: maybe one-liner for this
    return partial(apply, source=dataclass)

class decorate(fn, *, prolog=None, epilog=None):
  @wraps
  def _fn(*args, **kwargs):
    prolog?(*args, **kwargs)
    fn(*args, **kwargs)
    epilog?(*args, **kwargs)
