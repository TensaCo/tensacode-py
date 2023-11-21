from typing import Callable, Dict
from tensorcode.utils.fp import ConstFn

from tensorcode.utils.registry import FirstMatch, HighestScore, LookupStrategy, Registry


class Operation:

  OP_NAME: str # affirmative command verb, eg, "encode", "decode", "select"

  handlers = Registry(strategy=HighestScore)

  def __init__(self) -> None:
    self.handlers.register(ConstFn(0.0), self.default_handler)

  def __call__(self, *args, **kwargs):
    # args and kwargs can help determine which handler to use
    handler = self.handlers.first(*args, **kwargs)
    # but their primary purpose is for the handler itself
    return handler(*args, **kwargs)

  def default_handler(self, *args, **kwargs):
    raise NotImplementedError(f"Default {self.OP_NAME} operation is not implemented")

  def add_exception(self, type_or_trigger: Callable, handler: Callable):
    if type_or_trigger is type:
      type_or_trigger = lambda obj, *args, **kwargs: isinstance(obj, type_or_trigger)
    self.handlers[type_or_trigger] = handler
