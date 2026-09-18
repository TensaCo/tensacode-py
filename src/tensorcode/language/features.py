"""Feature structures and unification: the one mechanism the grammar agrees on.

A category is a name plus a feature structure. Agreement, subcategorisation,
gaps and semantic plumbing are all expressed by unifying those structures, so
the parser needs exactly one operation and the grammar needs no side channels.

Values are deliberately small: atoms (``str``, ``int``, ``bool``, ``None``),
variables, tuples, and nested structures. No lists, no mutation, no cyclic
terms, so unification terminates without an occurs check on recursive bindings
and a bound structure can be hashed and cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

Bindings = dict[str, Any]


@dataclass(frozen=True, order=True)
class FVar:
    """A feature variable, written ``?name`` in a production string."""

    name: str

    def __str__(self) -> str:
        return f"?{self.name}"


def unify(a: Any, b: Any, bindings: Bindings | None = None) -> Bindings | None:
    """Unify two values under ``bindings``; ``None`` means they cannot agree.

    ``bindings`` is never mutated: a caller can try an alternative without
    undoing anything.
    """
    out = dict(bindings or {})
    return out if _unify_into(a, b, out) else None


def _unify_into(a: Any, b: Any, out: Bindings) -> bool:
    # ``type(x) is dict`` rather than ``isinstance(x, Mapping)``: the typing ABC check
    # cost ~2 s of a 4.8 s benchmark run, and every feature structure here is a dict.
    if type(a) is FVar:
        a = resolve(a, out)
    if type(b) is FVar:
        b = resolve(b, out)
    if type(a) is FVar:
        out[a.name] = b
        return True
    if type(b) is FVar:
        out[b.name] = a
        return True
    if type(a) is dict and type(b) is dict:
        if not a or not b:
            return True
        for key, value in a.items():
            other = b.get(key, _ABSENT)
            if other is not _ABSENT and not _unify_into(value, other, out):
                return False
        return True
    if type(a) is tuple and type(b) is tuple:
        return len(a) == len(b) and all(_unify_into(x, y, out) for x, y in zip(a, b))
    return a == b


class _Absent:
    __slots__ = ()


_ABSENT = _Absent()


def resolve(value: Any, bindings: Bindings) -> Any:
    """Follow variable bindings one value deep (chains are followed to the end)."""
    if type(value) is not FVar or not bindings:
        return value
    seen: set[str] = set()
    while type(value) is FVar and value.name in bindings and value.name not in seen:
        seen.add(value.name)
        value = bindings[value.name]
    return value


def ground(value: Any, bindings: Bindings) -> Any:
    """Apply bindings throughout a value, leaving unbound variables in place."""
    if not bindings:
        return value
    value = resolve(value, bindings)
    if type(value) is dict:
        return {k: ground(v, bindings) for k, v in value.items()}
    if type(value) is tuple:
        return tuple(ground(v, bindings) for v in value)
    return value


def merge(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """``extra`` wins; used to specialise a lexical entry or a category."""
    out = dict(base)
    out.update(extra)
    return out


def rename(value: Any, suffix: str) -> Any:
    """Freshen variables so two uses of one production do not share them."""
    if isinstance(value, FVar):
        return FVar(f"{value.name}#{suffix}")
    if isinstance(value, Mapping):
        return {k: rename(v, suffix) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(rename(v, suffix) for v in value)
    return value
