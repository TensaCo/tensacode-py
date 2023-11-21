from copy import deepcopy
from functools import wraps

from tensacode._utils.helpers import join, Lockable, if_unlocked
from tensacode._utils.registry import Registry, HighestScore

class Ledger(Lockable):
    """Provides method wrapper `ledgered` to track the actions
    performed by a `Model` and inject them into future relevant
    method calls.
    """

    _ledger: Statement[] = []
    _active_statements: Statement[] = []

    def __init__(self):
        self._active_statements.append(Statement()) # for convenience

    @if_unlocked
    def record(self, statement: Statement):
        self._ledger.append(statement)

    @if_unlocked
    def read(self):
        return self.substatements(self._active_statements[-1])

    def substatements(self, statement: Statement) -> Statement[]:
        return self._ledger |> filter$((s)->s.parent==self._active_statements[-1], ?)

    def ledgered(self, fn):
        @wraps
        def _fn(model, *args, **kwargs):
            event = Event(fn, args, kwargs, parent=self._active_statements[-1])
            self.record(event)
            self._active_statements.insert(event, -1)

            # TODO: find out which arg or kwarg maps to the `context` param (if any)
            # Then append self.read() onto that list (or create it) becaore passing to `fn`
            raise NotImplemented()
            fn(self, *args, **kwargs)

            self._active_statements.remove(event, -1)
        return _fn

    data Statement(description: str, parent: Statement = None)
    data Value(variable: object, description: str = None, parent: Statement = None) from Statement:
        def __new__(variable, description=None, parent: Statement = None):
            return makedata(Value, deepcopy(variable),
                description ?? f'{variable.__name__}: {variable}', parent)
    data Event(fn, argvals: list = [], kwargvals: dict = {}, description: str = None, parent: Statement = None) from Statement:
        def __new__(fn, argvals, kwargvals, description=None, parent: Statement = None):
            return makedata(Event, fn, argvals, kwargvals, 
                description ?? f'Running {fn.__name__}({join(argvals)}, {join(kwargvals)})', parent)
        def replay(self):
            self.fn(*self.argvals, **self.kwargvals)
