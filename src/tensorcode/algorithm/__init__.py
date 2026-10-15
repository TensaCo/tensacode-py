"""Algorithms across symbolic (ALG.0), tensor (ALG.1), and attention (ALG.2) substrates.

NumPy supplies reference execution. Torch is imported only for neural substrates.
"""

from .extraction import from_attention, from_module
from .ir import Algorithm, AlgorithmBuilder, Variable
from .runtime import convert, topology
from .source import compile
from .validation import compare, validate_replacement

__all__ = [
    "Algorithm",
    "AlgorithmBuilder",
    "AlgorithmModule",
    "AttentionModule",
    "BiasedMultiheadAttention",
    "StructuralBias",
    "Variable",
    "compare",
    "compile",
    "convert",
    "from_attention",
    "from_module",
    "topology",
    "validate_replacement",
]


def __getattr__(name):
    if name in ("AlgorithmModule", "AttentionModule"):
        from . import tensor

        return getattr(tensor, name)
    if name in ("StructuralBias", "BiasedMultiheadAttention"):
        from . import structure

        return getattr(structure, name)
    raise AttributeError(name)
