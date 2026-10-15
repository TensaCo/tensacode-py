"""Compile bounded deterministic programs and compose them with learned models."""
from .core import Builder, Program, Value, boolean, float32, float64, int32
from .frontend import compile

__all__ = ['Builder', 'Program', 'Value', 'boolean', 'float32', 'float64', 'int32', 'compile']


def __getattr__(name):
    if name == 'ProgramOperation':
        from .operation import ProgramOperation
        return ProgramOperation
    if name == 'ProgramModule':
        from .tensor import ProgramModule
        return ProgramModule
    raise AttributeError(name)

from .runtime import Runtime
__all__.append('Runtime')
from .analysis import analyze
__all__.append('analyze')
from .verification import verified_replacement
__all__.append('verified_replacement')
