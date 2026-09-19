"""Deduction over the claim schema: what must be the case, given what was stated.

The store holds competing claims without resolving them, which is right — and left the agent
with no way to *resolve* them when a text says which combinations are allowed. Asked the
three-boxes puzzle (exactly one prize, three statements, exactly one of them true) it replied
"I don't know" and "I can not explain my reasoning", because holding possibilities is not the
same as searching them.

:mod:`~tensorcode.reasoning.constraints` is that search: candidate worlds over a finite
domain of hypotheses, constraints that can speak about the truth of other propositions, and
an elimination trace, because an answer whose reasoning cannot be shown is a guess that
happened to be right.

This package takes propositions as input. Turning English into them is the readers' job
(``agent/understand.py``, ``language/``); nothing here looks at a string of the puzzle.
"""

from .constraints import (
    MAX_WORLDS,
    Constraint,
    Distinct,
    Elimination,
    MustHold,
    Puzzle,
    Solution,
    TruthCount,
    Variable,
    World,
    check_by_elimination,
    entails,
    exactly_one,
    one_of,
    solve,
)

__all__ = [
    "MAX_WORLDS",
    "Constraint",
    "Distinct",
    "Elimination",
    "MustHold",
    "Puzzle",
    "Solution",
    "TruthCount",
    "Variable",
    "World",
    "check_by_elimination",
    "entails",
    "exactly_one",
    "one_of",
    "solve",
]
