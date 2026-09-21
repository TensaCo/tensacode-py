"""Symbolic graph records and unimplemented operation contracts.

Graph and SourceAnchor preserve caller-supplied structure. The operation classes
reserve the future symbolic API and always raise NotImplementedError on execution.
There is no neural, callback, or implicit semantic implementation behind them.
"""
from .representation import FrozenMap, Graph, SourceAnchor
from .encode import Encode, TextEncode
from .decode import Decode, TextDecode
from .transform import Transform
from .score import Score
from .retrieve import Retrieve
from .decide import ChoiceInput, Decide
from .classify import Classify

__all__ = [
    "FrozenMap", "Graph", "SourceAnchor", "Encode", "TextEncode", "Decode",
    "TextDecode", "Transform", "Score", "Retrieve", "ChoiceInput", "Decide",
    "Classify",
]
