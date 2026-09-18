"""Task-stated invariants over a scene: refuse a reading that the task's own structure rejects.

A perception layer cannot know that two strings on screen must agree, or what one of them
has to look like. A task can. An invariant is a small declaration the task makes about what
it reads, and there are two kinds:

    same_id = SameIdentifier("project", r"[a-z]+-[a-z]+-(\\d{4,})", r"task-(\\d{4,})\\.txt")
    shape = WellFormed("project name", r"project called ([^\\s(]+)", r"[a-z]+-[a-z]+-\\d{4,}")
    verdict = check(scene_texts, [same_id, shape])
    if verdict.violations: ...          # the reading cannot be right: act on none of it

``check`` returns the values that agree (corroborated: the same value read in two
independent places, or a value that has the shape it must have), and the ones that fail.
Failures name the readings, so a task can escalate honestly instead of guessing.

``mask`` rewrites the offending value out of a line, so a grammar downstream abstains rather
than parsing a truncated identifier — which is the measured failure mode: a project name
read as ``atlas`` instead of ``atlas-sync-31288`` produced a whole episode of correct-looking
work in the wrong directory, and the agent then confirmed it against its own misreading.

This is deliberately not string-similarity heuristics over the whole screen (measured and
rejected: unrelated identifiers legitimately differ by one character). It only judges values
the task has said something about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SameIdentifier:
    """Two or more patterns whose captured group must be the same value wherever it appears."""

    name: str
    patterns: tuple[str, ...]

    def __init__(self, name: str, *patterns: str) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "patterns", tuple(patterns))

    def readings(self, texts: Iterable[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for text in texts:
            for pattern in self.patterns:
                for m in re.finditer(pattern, text):
                    value = m.group(1) if m.groups() else m.group(0)
                    found.setdefault(pattern, []).append(value)
        return found

    def judge(self, texts: Sequence[str]) -> tuple[str, list[str]]:
        readings = self.readings(texts)
        if len(readings) < len(self.patterns):
            return "unseen", []
        values = sorted({v for vs in readings.values() for v in vs})
        return ("ok", values) if len(values) == 1 else ("violation", values)

    def mask(self, text: str, refused: str) -> str:
        """Take the whole identifier out, so a grammar cannot parse a fragment of it."""
        for pattern in self.patterns:
            text = re.sub(pattern, refused, text)
        return text


@dataclass(frozen=True)
class WellFormed:
    """A value whose shape the task knows: read it where it appears, refuse what cannot be it.

    ``where`` locates the value (group 1) in a line of text; ``shape`` is what the value must
    match in full. This catches a misreading that no second reading contradicts — a lost
    hyphen, a fused word — which agreement between two readings cannot see.
    """

    name: str
    where: str
    shape: str

    @property
    def patterns(self) -> tuple[str, ...]:
        return (self.where,)

    def readings(self, texts: Iterable[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for text in texts:
            for m in re.finditer(self.where, text):
                found.setdefault(self.where, []).append(m.group(1))
        return found

    def judge(self, texts: Sequence[str]) -> tuple[str, list[str]]:
        values = self.readings(texts).get(self.where, [])
        if not values:
            return "unseen", []
        bad = sorted({v for v in values if not re.fullmatch(self.shape, v)})
        return ("violation", bad) if bad else ("ok", sorted(set(values)))

    def mask(self, text: str, refused: str) -> str:
        return re.sub(self.where, lambda m: m.group(0).replace(m.group(1), refused), text)


@dataclass
class Verdict:
    corroborated: dict[str, str] = field(default_factory=dict)  # invariant name -> the value it holds for
    violations: dict[str, list[str]] = field(default_factory=dict)  # invariant name -> the readings that failed
    unseen: list[str] = field(default_factory=list)  # invariants whose patterns did not all appear

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        parts = [f"{k}={v!r} (corroborated)" for k, v in self.corroborated.items()]
        parts += [f"{k} does not hold: {' vs '.join(sorted(set(v)))}" for k, v in self.violations.items()]
        parts += [f"{k}: not seen in every place" for k in self.unseen]
        return "; ".join(parts)


def check(texts: Sequence[str], invariants: Sequence[SameIdentifier | WellFormed]) -> Verdict:
    verdict = Verdict()
    for inv in invariants:
        outcome, values = inv.judge(texts)
        if outcome == "unseen":
            verdict.unseen.append(inv.name)
        elif outcome == "violation":
            verdict.violations[inv.name] = values
        else:
            verdict.corroborated[inv.name] = values[0]
    return verdict


def mask(texts: Sequence[str], invariants: Sequence[SameIdentifier | WellFormed], verdict: Verdict, refused: str = "<refused>") -> list[str]:
    """The same lines with every failing invariant's value taken out of them."""
    out = list(texts)
    for inv in invariants:
        if inv.name in verdict.violations:
            out = [inv.mask(line, refused) for line in out]
    return out
