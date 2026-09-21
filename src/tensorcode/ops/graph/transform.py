from ..base import Operation
from .representation import Graph


class Transform(Operation):
    def __init__(self, function, *, identity=None, replayable=False):
        if not callable(function):
            raise TypeError('Transform requires a callable')
        self.function = function
        self.identity = identity
        self.replayable = replayable

    def forward(self, value, *, context=None):
        if not isinstance(value, Graph):
            raise TypeError('Expected a Graph')
        return self.function(value, context or {})

    def configuration(self):
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError('An explicit callback identity is required for persistence')
        return {
            'operation': 'graph.transform',
            'identity': self.identity,
            'replayable': bool(self.replayable),
        }
