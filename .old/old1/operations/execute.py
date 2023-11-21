from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable, Union
from typing_extensions import Self

Enumerator, MembershipFn, Recommender = Callable, Callable, Callable



class GetOptions:
  enumerator: Callable[[], List[object]] # returns the objects referenced by this option, may overlap other options
  membershipFn: Callable[[object], bool] # determines if an object is under the influence of this option
  recommender: Callable[[object], float] # recommends for or against a given object that it recognizes as under its purview

  def __init__(self, enumerator, membershipFn, recommender):
    self.enumerator = enumerator
    self.membershipFn = membershipFn
    self.recommender = recommender

  @staticmethod
  def foo():
    pass

# TODO: finish this
GetOptions.SELECT.LOCALS = GetOptions(...)
GetOptions.SELECT.GLOBALS = GetOptions(...)
GetOptions.SELECT.OBJECTS = GetOptions(...)
GetOptions.SELECT.CALLABLES = GetOptions(...)
GetOptions.SELECT.TYPES = GetOptions(...)
GetOptions.SELECT.ATTRIBUTES = GetOptions(...)
GetOptions.SELECT.METHODS = GetOptions(...)
GetOptions.SELECT.NON_METHOD_FUNCTIONS = GetOptions(...)
GetOptions.SELECT.INDEXED_ITEMS = GetOptions(...)
GetOptions.SELECT.OTHER_OBJECTS = GetOptions(...)
GetOptions.SELECT.FUNCTION_RESULT = GetOptions(...)
GetOptions.SELECT.METHOD_RESULT = GetOptions(...)
GetOptions.SELECT.CONSTRUCTOR_RESULT = GetOptions(...)
GetOptions.SELECT.ANY = GetOptions(...)

# special functions TODO: add these NO ACTUALLY LATER
# functionalize, eg, say f is available but you're cwd is x, then call x.functionalize(f) to cwd to f

  # all possible 'query' statements
  # the options referenced by these flags cannot be determined in advance since
  # 1) some functions will mutate the state of the program and 2) it would be too
  # expensive to enumerate all possible options in advance
  # these tags are not necessarily mutually exclusive or in a tree structure

  # They are all converted into an IntMap with negative values disallowed and
  # positive ones allowed. Allow rules override disallow rules if they are more
  # positive than the disallow rule is negative.


T_Rules = List[Union[GetOptions,type,object]]

def get(
    Type: type,
    num=1,
    allowed:T_Rules=[GetOptions.ALL],
    disallowed:T_Rules=[],
    rules: List[Tuple[Enumerator, MembershipFn, Recommender]] = None,
    depth_limit=None,
    default=None,
    default_threshold=0.1,
    ctx: dict=None) -> object:
  """retrieves an object of a given type
    a) from existing,
    b) from calling a function (including constructor),
    c) from a default value"""
  pass