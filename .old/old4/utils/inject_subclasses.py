from typing import Callable, Dict, List, Optional, Tuple

INJECTED: Dict[Tuple[type, str], List[type]] = dict()

def inject_into_subclasses_epilog(owner: type, method: str, fn: Callable):
  """Attaches `fn` to the end of each `method` in direct subclasses of `owner`"""
  inject_subclasses(owner=owner, method=method ,post=fn)

def inject_into_subclass_prolog(owner: type, method: str, fn: Callable):
  """Attaches `fn` to the start of each `method` in direct subclasses of `owner`"""
  inject_subclasses(owner=owner, method=method, post=fn)

def inject_subclasses(
  owner: type,
  method: str,
  pre: Optional[Callable],
  post: Optional[Callable]):
  """Attaches `injection` to the start and/or end of each `method`
  for all direct subclasses of `owner`"""

  if (owner, method) not in INJECTED:
    INJECTED[(owner, method)] = list()

  for subclass in owner.__subclasses__:
    if subclass not in INJECTED[(owner, method)]:
      if method in subclass.__dict__:
        old = subclass.__dict__[method]
        def new(*args, **kwargs):
          pre(*args, **kwargs)
          old(*args, **kwargs)
          post(*args, **kwargs)
        setattr(subclass, method, new)

        INJECTED[owner].append(subclass)