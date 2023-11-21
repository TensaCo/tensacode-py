from tensacode.utils.stacked import stacked

def make_export_helpers(__all__, __dict__):
  """Used to push and pop variables to the top-level module namespace"""

  _push, _pop, _stack = stacked(__dict__)

  def push(key, val):
    _push(key, val)
    if key not in __all__: __all__.append(key)
  def pop(key):
    _pop(key)
    __all__.remove(key)

  return push, pop, _stack