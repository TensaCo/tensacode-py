from __future__ import annotations
from functools import wraps
import inspect
from typing import Callable, TypeVar


class Ledger:

    depth: int = 0
    _statements: dict[str, Statement]
    replay_mode = False

    def __init__(self):
        self._statements = []

    @property
    def statements(self):
        return list(filter(lambda s: s.scope == self.depth, self._statements))

    def ledgered(self, fn):
        # when the function is called, we will enter a deeper scope
        @wraps(fn)
        def _fn(*args, **kwargs):
            if self.replay_mode:
                if
            
            self.depth += 1
            self.add(Function(fn, args, kwargs))
            fn(*args, **kwargs)
            self.depth -= 1

    def add(self, statement: Statement):
        if tc.text.Model.remember is False:
            return
        statement.depth = self.depth
        self._statements.append(statement)
        
    def track_all_functions(module=None, max_height=5, max_nested=2, skip_last_frames=0, exclude=None, exclude_tags=None, required_tags=None):
        """Track all functions in a module and its submodules (or everywhere if module is None)

        Args:
            directions (str): What you're looking for.
            max_height (int, optional): Maximum number of frames to climb. Defaults to 5.
            max_nested (int, optional): Maximum levels of nested access to explore. Defaults to 2.
            skip_last_frames (int, optional): Skip the first `skip_last_frames` when climbing the call stack. These skipped frames do not count towards the `max_height`. Defaults to 0.
            exclude (list[str], optional): Module globs to exclude. Defaults to None.
            exclude_tags (list[str], optional): Functions with these tags in their __tensacode__.tag attr will be excluded. Defaults to None.
            required_tags (list[str], optional): Functions without these tags in their __tensacode__.tag attr will be excluded. Defaults to None.

        Returns:
            Any: The object you're looking for.

        Raises:
            ValueError: If the object cannot be found.
        """
        frames = TODO
        for frame in frames:
            dict = {**frame.f_locals, **frame.f_globals}

class Statement:
    msg: str
    substatements: list[Statement]
    scope: list[str]
    depth: int


class Value(Statement):
    var_name: str
    var_value: object

    @property
    def msg(self):
        return f"{self.var_name}={self.var_value}"

    substatements = []


class Function(Statement):
    fn: Callable
    args: list
    kwargs: dict

    @property
    def msg(self):
        return tc.text.encode(self.fn, remember=False)
