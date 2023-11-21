_NAMESPACE = {}
def unique(name: str, namespace=_NAMESPACE, scope_delimitor='/'):
  if name not in _NAMESPACE:
    _NAMESPACE[name] = 0
    return name
  else:
    _NAMESPACE[name] += 1
    return name + scope_delimitor + _NAMESPACE[name]
