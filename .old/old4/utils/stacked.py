def stacked(_dict):
  """Manages a stack of dict assignments.
  Useful for stacking class attributes or module exports."""

  stack = dict()

  def push(key, val):
    if key not in stack: stack[key] = list()
    stack[key].push(val)
    _dict[key] = val

  def pop(key):
    if key not in stack: return
    val = stack[key].pop()
    if len(stack[key]) == 0: _dict.pop(key)
    else: _dict[key] = stack[key][-1]
    return val

  return push, pop, stack