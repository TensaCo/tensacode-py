from __future__ import annotations

from typing import Mapping
from typing_extensions import Self


class Namespaced:
  """Namespaced

  Seemlessly hijacks the `__new__` method of subclasses to ensure
  instances of `Namespaced` with identical `name`s are identical.
  """

  ALL: Mapping[str, object] = dict() # {name, obj}

  def __new__(cls: type[Self], name: str) -> Self:
    if name in Namespaced.ALL:
      self = Namespaced.ALL[name]
    else:
      self = super.__new__()
      Namespaced.ALL[name] = self
    return self