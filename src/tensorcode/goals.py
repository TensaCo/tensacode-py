"""Desired outcomes independent of a language resource or execution plan.

Interpreters and domain refiners supply these conditions. This module does not
infer a specification from an abstract task name or claim to implement planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Condition:
    """A desired state: predicate over roles, possibly negated."""

    pred: str
    args: Mapping[str, Any]
    negated: bool = False

    def describe(self) -> str:
        inner = ", ".join(f"{k}={getattr(v, 'text', v)}" for k, v in self.args.items())
        return ("not " if self.negated else "") + f"{self.pred}({inner})"


@dataclass(frozen=True)
class GoalSpec:
    """A conjunction of desired outcomes, with no prescribed means of achieving it.

    Role names currently follow the capability matcher conventions. Values can
    be domain references or descriptions resolved by a plugin. This is a first
    interface, not a quantified or temporal goal language.
    """

    conditions: tuple[Condition, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("a goal must specify at least one desired condition")
        if any(v is None or v == "addressee" for c in self.conditions for v in c.args.values()):
            raise ValueError("explicit goal roles must be bound; unresolved lexical roles are not a specification")

    def describe(self) -> str:
        return "; ".join(c.describe() for c in self.conditions)
