"""Desired outcomes independent of a language resource or execution plan.

Interpreters and domain refiners supply these conditions. This module does not
infer a specification from an abstract task name or claim to implement planning.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace
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


def normalize_goal_value(value: Any, *, path: str = "goal") -> Any:
    """Preserve supplied values, resolving only explicit linguistic bindings.

    Imports are local because language resources also import Condition. No
    language taxonomy, description lookup, or plugin participates in grounding.

    A featureless Entity carrying a Ref is an explicit reference wrapper. Its
    description is not an additional constraint. Any attached features require
    semantic projection before this boundary: identity does not establish that
    a count, modifier, or other qualification was satisfied. No feature names
    are presumed harmless. The sole scalar representation field consumed here
    is ``value`` on an unbound literal/number, as defined by ``explicit_ref``.
    Refiners must represent qualifications in conditions/invariants and supply
    the resulting domain values; this function cannot certify that projection.
    """
    from .language.semantics import Entity, explicit_ref
    from .outcomes import Unknown

    if isinstance(value, Entity):
        if value.candidates:
            raise ValueError(f"unresolved entity alternatives at {path}")
        consumed = {"value"} if value.ref is None and value.kind in ("number", "literal") else set()
        unconsumed = set(value.features) - consumed
        if unconsumed:
            names = ", ".join(sorted(repr(name) for name in unconsumed))
            raise ValueError(f"unconsumed entity features at {path}: {names}; explicit semantic projection required")
        resolved = explicit_ref(value)
        if isinstance(resolved, Unknown):
            raise ValueError(f"explicit grounding required at {path}: {resolved.detail or resolved.reason}")
        return resolved
    if isinstance(value, Unknown):
        raise ValueError(f"unknown goal value at {path}: {value.reason}")
    if isinstance(value, Mapping):
        updated = {}
        changed = False
        for key, item in value.items():
            normalized_key = normalize_goal_value(key, path=f"{path}[key]")
            normalized_item = normalize_goal_value(item, path=f"{path}[{key!r}]")
            if normalized_key in updated:
                raise ValueError(f"grounded mapping keys collide at {path}")
            updated[normalized_key] = normalized_item
            changed = changed or normalized_key is not key or normalized_item is not item
        if not changed:
            return value
        if isinstance(value, MutableMapping):
            normalized = copy(value)
            normalized.clear()
            normalized.update(updated)
            return normalized
        return type(value)(updated)
    if isinstance(value, (tuple, list)):
        updated = [normalize_goal_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
        if all(new is old for new, old in zip(updated, value)):
            return value
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*updated)
        return type(value)(updated)
    if isinstance(value, (set, frozenset)):
        return type(value)(normalize_goal_value(item, path=f"{path}[member]") for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        changes = {}
        for item in fields(value):
            original = getattr(value, item.name)
            normalized = normalize_goal_value(original, path=f"{path}.{item.name}")
            if normalized is not original:
                if not item.init:
                    raise ValueError(f"cannot normalize non-init field {path}.{item.name}")
                changes[item.name] = normalized
        return replace(value, **changes) if changes else value
    return value


@dataclass(frozen=True)
class GoalSpec:
    """A conjunction of desired outcomes, with no prescribed means of achieving it.

    Predicates and role names are exact domain data, independent of lexical
    aliases. Values are explicit domain values or bound linguistic entities,
    normalized at construction for every execution path. Descriptions without
    explicit identity and entities with unprojected qualifications are rejected.
    This is not a quantified or temporal language.
    """

    conditions: tuple[Condition, ...]
    label: str = ""
    invariants: tuple[Condition, ...] = ()
    basis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("a goal must specify at least one desired condition")
        def normalized(conditions: tuple[Condition, ...], section: str) -> tuple[Condition, ...]:
            result = []
            for index, condition in enumerate(conditions):
                values = {}
                for role, value in condition.args.items():
                    path = f"{section}[{index}].args[{role!r}]"
                    if value is None:
                        raise ValueError(f"explicit goal roles must be bound: {path}")
                    values[role] = normalize_goal_value(value, path=path)
                result.append(replace(condition, args=values))
            return tuple(result)
        object.__setattr__(self, "conditions", normalized(self.conditions, "conditions"))
        object.__setattr__(self, "invariants", normalized(self.invariants, "invariants"))

    def describe(self) -> str:
        return "; ".join(c.describe() for c in self.conditions)
