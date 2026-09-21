from ..base import Operation
from .representation import Graph


class Transform(Operation):
    def __init__(self, function, *, replayable=False):
        self.function = function
        self.replayable = replayable

    def forward(self, value, *, context=None):
        if not isinstance(value, Graph):
            raise TypeError('Expected a Graph')
        return self.function(value, context or {})
