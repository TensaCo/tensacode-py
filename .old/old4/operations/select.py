from tensorcode.operations.operation import Operation


class Select(Operation):
  
  def __call__(self, list, context):
    super.__call__(list, context, type=type(list)) 
    # fixme: how should list typing be handled? homogenous/heterogenous x item type