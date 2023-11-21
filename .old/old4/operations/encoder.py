from tensorcode.operations.operation import Operation


class Encoder(Operation):

  OP_NAME: str = "decode"

  def __init__(self) -> None:
    super().__init__()

  def __call__(self, tensor, context, *args, **kwargs):
    # recursively initialize object like Type
    super().__call__(tensor, Type, context, *args, **kwargs)