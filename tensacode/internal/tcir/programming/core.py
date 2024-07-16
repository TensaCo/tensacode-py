from tensacode.internal.tcir.tcir_values import Function


class Call()

class Assign(Function):
    name = "assign"
    arity = 2
    apply = lambda x, y: y

