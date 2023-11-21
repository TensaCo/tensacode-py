from tensacode.operations.operation import Operation


class Decode(Operation):

  OP_NAME: str = "decode"

  def __call__(self, tensor, Type, context, *args, **kwargs):
    # recursively initialize object like Type

    super().__call__(tensor, Type, context, *args, **kwargs)