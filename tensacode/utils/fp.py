from typing import Protocol, TypeVar, runtime_checkable

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