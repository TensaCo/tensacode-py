from __future__ import annotations

from math import inf
import logging
from typing import Callable, Hashable, List, MutableMapping, Optional, Tuple, TypeVar

from tensacode.utils.assertions import exactly_one
from tensacode.utils.fp import MapFn, Number, Predicate, ScoringFn


_V = TypeVar("_V")
T_Callable = TypeVar("T_Callable", Callable)
KV_PAIR = Tuple[T_Callable, _V]

class Registry:

  registry: dict[T_Callable, _V]

  def __init__(self, strategy: LookupStrategy, dict) -> None:
    super().__init__()
    self.strategy = strategy
    self.registry = dict or {}

  # Registration methods

  def register(self,
      key_or_fn: Hashable|T_Callable,
      value: _V):
    if isinstance(key_or_fn, T_Callable):
      self.register_fn(key_or_fn, value)
    else:
      self.register_key(key_or_fn, value)

  def register_key(self, key: Hashable, value: _V):
    self.register_fn(lambda x: x == key, value)

  def register_fn(self, fn: T_Callable, value: _V):
    self.registry[fn] = value

  # Retrieval methods

  def first(self, *args, **kwargs) -> Optional[KV_PAIR]:
    first = self._first_pair(*args, **kwargs)
    if first is None:
      return None
    else:
      return first[1]

  def _first_pair(self, *args, **kwargs) -> Optional[KV_PAIR]:
    return self.strategy.first(self.registry, args, kwargs)

  def all(self, *args, **kwargs) -> List[KV_PAIR]:
    return [val for _, val in self._all_pairs(*args, **kwargs)]

  def _all_pairs(self, *args, **kwargs) -> List[KV_PAIR]:
    return self.strategy.all(self.registry, args, kwargs)

  def _all_iter(self, fn_args, fn_kwargs):
    raise NotImplementedError("Subclasses must implement this method")

  #### This is not pythonic, so I want to get rid of it
  #### # Mapping methods
  ####
  #### def __getitem__(self, __key: Hashable|T_Callable) -> Optional[_V]:
  ####   return self.first(__key)
  ####
  #### def __setitem__(self, __key: Hashable|T_Callable, __value: _V) -> None:
  ####   match = self.first(__key)
  ####   # If match not found, register new trigger for this key
  ####   if match is None:
  ####     self.register(__key, value=__value)
  ####   # Otherwise, update the existing trigger
  ####   else:
  ####     fn, _ = match
  ####     self.registry[fn] = __value
  ####
  #### def __delitem__(self, __key: Hashable|T_Callable) -> None:
  ####   match = self.first(__key)
  ####   # See if a trigger already matches or equals this key
  ####   if match is not None:
  ####     fn, _ = match
  ####     del self.registry[fn]
  ####   # Otherwise, the trigger is already absent. Raise KeyError
  ####   else:
  ####     raise KeyError(f"Key {__key} not found")


class LookupStrategy:

  def first(self, dict, fn_args, fn_kwargs) -> Optional[KV_PAIR]:
    try:
      return next(self.all_iter(dict, fn_args, fn_kwargs))
    except StopIteration:
      return None

  def all(self, dict, fn_args, fn_kwargs) -> List[KV_PAIR]:
    return list(self.all_iter(dict, fn_args, fn_kwargs))

  def all_iter(self, dict, fn_args, fn_kwargs):
    raise NotImplementedError("Subclasses must implement this method")

class FirstMatch(LookupStrategy):
  """The first predicate to return True given the supplied args wins, and its
  corresponding value is returned. If no predicate evaluates to True, look for
  an exact match, and if that fails, return None.
  """

  def _all_iter(self, dict, fn_args, fn_kwargs):
    for fn, val in dict.items():
      try:
        if fn(*fn_args, **fn_kwargs):
          yield fn, val
      except SyntaxError:
        logging.warning(f"Trigger function {fn} does not coform to `Predicate` protocol. Skipping.")

class HighestScore(LookupStrategy):
  """The highest scoring function wins, and its corresponding value is returned.
  If all scores are -inf, look for an exact match, and if that fails, return None.
  """

  def _all_iter(self, dict, fn_args, fn_kwargs):
    bucket = []
    for fn, val in dict.items():
      try:
        score = fn(*fn_args, **fn_kwargs)
        if score == -inf or score is False or score is None:
          continue
        if not isinstance(score, Number):
          score = 1
        bucket.append(((fn, val), score))
      except SyntaxError:
        logging.warning(f"Scoring function {fn} does not coform to `Predicate` protocol. Skipping.")
    sorted = bucket.sort(key=lambda x: x[1], reverse=True)
    return iter(sorted)