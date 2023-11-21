from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable, Union
from typing_extensions import Self

# TODO: skeleton the `execute` interface first

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