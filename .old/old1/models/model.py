from enum import Enum
from typing import List

from tensacode import pop_export, push_export
from tensacode.operations.operation import Operation
from tensacode.utils.inject_subclasses import inject_into_subclasses_epilog
from tensacode.utils.unique import unique


class Model:

  @property
  def operations(self) -> List[Operation]:
    return []

  def __init__(self, fn, input_spec, output_spec, context_name):

    self.fn = fn
    self.input_spec = input_spec
    self.output_spec = output_spec
    self.context_name = unique(context_name)

    # FIXME: do we even want Models to hijack the current_model as soon as they are created?
    # if I'm being subclassed, inject my context manager into all subclass methods
    @inject_into_subclasses_epilog(self.__name__, '__init__')
    def with_context(method, *args, **kwargs):
      with self:
        method(*args, **kwargs)

  def __enter__(self):
    for operation in self.operations:
      push_export(operation.OP_NAME, operation.handler)
    push_export(Model.CURRENT_MODEL, self)

    return self

  def __exit__(self, exc_type, exc_value, traceback):
    for operation in self.operations:
      pop_export(operation.OP_NAME, operation.handler)
      toplevel_namespace_export(openation.handler)

  @toplevel_namespace_export
  def train(self, path):
    pass

  @toplevel_namespace_export
  def reward(self, path):
    pass

  @toplevel_namespace_export
  def add_loss(self, path):
    pass

  @toplevel_namespace_export
  def load(self, path):
    pass

  @toplevel_namespace_export
  def load(self, path):
    pass