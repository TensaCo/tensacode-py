_NAMESPACE = {}
def unique(name: str, namespace=_NAMESPACE, scope_delimitor='/'):
  if name not in _NAMESPACE:
    _NAMESPACE[name] = 0
    return name
  else:
    _NAMESPACE[name] += 1
    return name + scope_delimitor + _NAMESPACE[name]

import inspect

def TODO(reason=None):
  error = NotImplementedError("TODO" + (reason ?? "This is a TODO"))
  error.add_note(inspect.currentframe().f_back |> inspect.get_source)
  raise error

def subclasses_should_implement(cls):
  calling_class = inspect.currentframe().f_back.f_locals['self'].__class__
  raise NotImplementedError("Not implemented in {calling_class}. Subclasses should implement this method")

def unscramble_optional_first(first_or_second, definitely_second, default=None):
  if definitely_second:
    return first_or_second, definitely_second
  else:
    return default, first_or_second