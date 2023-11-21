from typing import List

NoneType = type(None)


def exactly_one(*args):
  return exactly_n(*args, n=1)

def exactly_n(*args: List[object|NoneType], n: int):
  return sum(int(bool(arg)) for arg in args) == n