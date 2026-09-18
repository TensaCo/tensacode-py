"""Inducing readable artifacts: decision lists, preconditions, and role types.

Three inducers, each with the discipline its source repo learned the hard way:

* :func:`decision_list` — greedy separate-and-conquer with an **MDL stop**, so a
  rule has to save more bits than it costs to state. (The algorithm is
  ``symbolic-ai-models``'s ``models/ruleinduce_001/dlist.py``, re-expressed over
  claims instead of that repo's graph facts.)
* :func:`preconditions` — the most-specific cover of the states in which an action
  fired, with optional **interventional pruning**: keep a condition only if
  removing it changes what the world does. (``synthEX``'s
  ``perception/induce_rules.py`` reports precision 0.44 → 0.74 from exactly this.)
* :func:`role_type` — the least general type covering every observed filler, with a
  **productivity gate**: a shape generalises only when many distinct fillers share
  it and almost all of them are of that kind, otherwise the forms are memorised.
  (``symbolic-ai-models``'s ``reader/learned.py`` ``Lexicon.induce``.)

Nothing here adopts what it induces. Adoption is :mod:`tensacode.learning.verify`,
which is where the controls live.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

from ..outcomes import Score, Unknown
from .literals import Case, Literal

# ------------------------------------------------------------- decision lists


@dataclass
class Rule:
    conditions: tuple[Literal, ...]
    label: Any
    support: int = 0
    correct: int = 0

    def matches(self, facts: Any) -> bool:
        return all(c.holds(facts) for c in self.conditions)

    @property
    def confidence(self) -> float:
        return self.correct / max(1, self.support)

    def cost(self) -> int:
        return 2 + sum(c.cost() for c in self.conditions)

    def __repr__(self) -> str:
        body = " ∧ ".join(repr(c) for c in self.conditions) or "true"
        return f"IF {body} THEN {self.label!r}  [{self.correct}/{self.support}]"


@dataclass
class DecisionList:
    """An ordered list of rules you can read, argue with, and price."""

    rules: list[Rule] = field(default_factory=list)
    default: Any = None
    considered: int = 0

    def predict(self, facts: Any) -> Any:
        for rule in self.rules:
            if rule.matches(facts):
                return rule.label
        return self.default

    def explain(self, facts: Any) -> tuple[Any, Rule | None]:
        for rule in self.rules:
            if rule.matches(facts):
                return rule.label, rule
        return self.default, None

    def decide(self, facts: Any) -> tuple[Any, "Rule | None", Any]:
        """Predict, and return the certificate of what the decision actually read.

        The read set includes predicates that were *absent*, so an answer is
        invalidated by a fact appearing as well as by one changing.
        """
        from .certificate import Reader

        reader = Reader(dict(facts) if isinstance(facts, frozenset) else facts, note="decision_list")
        label, rule = self.explain(reader)
        return label, rule, reader.readset()

    def score(self, facts: Any) -> Score:
        _, rule = self.explain(facts)
        return Score(rule.confidence if rule else 0.0, "uncalibrated")

    def cost(self) -> int:
        return sum(r.cost() for r in self.rules) + 2

    def accuracy(self, cases: Sequence[Case]) -> float:
        if not cases:
            return 0.0
        return sum(self.predict(facts) == label for facts, label in cases) / len(cases)

    def __repr__(self) -> str:
        return "\n".join([*(repr(r) for r in self.rules), f"ELSE {self.default!r}"])


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def decision_list(cases: Sequence[Case], literals: Sequence[Literal], *, max_conditions: int = 3,
                  min_support: int = 2, min_confidence: float = 0.6, max_rules: int = 40,
                  beam: int = 4, mdl: bool = True) -> DecisionList:
    """Induce an ordered rule list. Discrete throughout: no gradients, five knobs."""
    remaining = list(cases)
    out = DecisionList(considered=len(literals))
    labels = Counter(label for _, label in cases)
    out.default = labels.most_common(1)[0][0] if labels else None

    while remaining and len(out.rules) < max_rules:
        rule = _grow(remaining, literals, max_conditions=max_conditions, min_support=min_support, beam=beam)
        if rule is None or rule.confidence < min_confidence:
            break
        if mdl:
            covered = [(f, y) for f, y in remaining if rule.matches(f)]
            before = _entropy(Counter(y for _, y in remaining)) * len(covered)
            after = _entropy(Counter(y for _, y in covered)) * len(covered)
            if before - after < rule.cost():  # the rule must pay for its own statement
                break
        out.rules.append(rule)
        remaining = [(f, y) for f, y in remaining if not rule.matches(f)]
    if remaining:
        out.default = Counter(y for _, y in remaining).most_common(1)[0][0]
    return out


def _grow(cases: Sequence[Case], literals: Sequence[Literal], *, max_conditions: int, min_support: int,
          beam: int) -> Rule | None:
    """Beam search over conjunctions, scored by information gain weighted by purity."""
    hits = {literal: [literal.holds(facts) for facts, _ in cases] for literal in literals}
    labels = [label for _, label in cases]
    base = _entropy(Counter(labels))
    n = len(cases)

    def score(mask: Sequence[bool]) -> tuple[float, Counter, int]:
        picked = [i for i, m in enumerate(mask) if m]
        if len(picked) < min_support:
            return -1e9, Counter(), 0
        counts = Counter(labels[i] for i in picked)
        return (base - _entropy(counts)) * (len(picked) / n), counts, len(picked)

    beams: list[tuple[float, tuple[Literal, ...], list[bool]]] = [(0.0, (), [True] * n)]
    best: tuple[float, tuple[Literal, ...], Counter, int] | None = None
    for _ in range(max_conditions):
        nxt: list[tuple[float, tuple[Literal, ...], list[bool]]] = []
        for _, conditions, mask in beams:
            for literal in literals:
                if literal in conditions:
                    continue
                merged = [a and b for a, b in zip(mask, hits[literal])]
                gain, counts, support = score(merged)
                if gain <= 0:
                    continue
                adjusted = gain * (max(counts.values()) / max(1, support))
                nxt.append((adjusted, conditions + (literal,), merged))
                if best is None or adjusted > best[0]:
                    best = (adjusted, conditions + (literal,), counts, support)
        if not nxt:
            break
        nxt.sort(key=lambda row: (-row[0], tuple(repr(c) for c in row[1])))
        beams = nxt[:beam]
    if best is None:
        return None
    _, conditions, counts, support = best
    label, correct = counts.most_common(1)[0]
    return Rule(conditions, label, support, correct)


# ---------------------------------------------------------------- preconditions


@dataclass(frozen=True)
class Precondition:
    """What has to hold for an action to fire, and how it was established."""

    action: str
    conditions: tuple[Literal, ...]
    support: int
    method: str = "most-specific-cover"

    def holds(self, facts: Any) -> bool:
        return all(c.holds(facts) for c in self.conditions)

    def __repr__(self) -> str:
        body = " ∧ ".join(repr(c) for c in self.conditions) or "true"
        return f"{self.action} needs {body}  [{self.support} firings, {self.method}]"


def preconditions(action: str, fired: Sequence[frozenset], *,
                  intervene: Callable[[str, frozenset], bool] | None = None,
                  did_not_fire: Sequence[frozenset] = ()) -> Precondition:
    """Conditions true in **every** state where the action fired.

    The passive cover over-specialises: anything the world happened to keep constant
    becomes a condition. Two optional correctives, in the order they should be tried:

    * ``did_not_fire`` states drop conditions that also held when nothing happened;
    * ``intervene(action, facts)`` is asked whether the action still fires with a
      condition removed, which is the only way to tell a cause from a coincidence.
    """
    if not fired:
        return Precondition(action, (), 0, "no firings")
    common = set(fired[0])
    for state in fired[1:]:
        common &= set(state)
    conditions = [Literal(p, v) for p, v in sorted(common, key=repr)]

    if did_not_fire:
        conditions = [c for c in conditions if not all(c.holds(state) for state in did_not_fire)]
    method = "most-specific-cover" + ("+negatives" if did_not_fire else "")

    if intervene is not None:
        kept: list[Literal] = []
        for condition in conditions:
            weakened = [c for c in conditions if c is not condition]
            probe = frozenset((c.predicate, c.value) for c in weakened)
            if not intervene(action, probe):  # removing it stopped the action: it matters
                kept.append(condition)
        conditions, method = kept, method + "+intervention"
    return Precondition(action, tuple(conditions), len(fired), method)


def effects(before: Sequence[frozenset], after: Sequence[frozenset]) -> tuple[Literal, ...]:
    """What every firing added: the union of new facts, kept only where it is unanimous."""
    added: list[set] = [set(b) ^ (set(a) & set(b)) for a, b in zip(after, before)]
    gained = [set(a) - set(b) for a, b in zip(after, before)]
    if not gained:
        return ()
    common = set(gained[0])
    for facts in gained[1:]:
        common &= facts
    _ = added
    return tuple(Literal(p, v) for p, v in sorted(common, key=repr))


# ------------------------------------------------------------------ role types


@dataclass(frozen=True)
class RoleType:
    """What fills a role: a generalising shape, or a memorised set of forms."""

    role: str
    concept: str | None
    shapes: frozenset[str] = frozenset()
    forms: frozenset[str] = frozenset()
    productive: bool = False
    support: int = 0

    def admits(self, value: Any) -> bool:
        text = value if isinstance(value, str) else str(value)
        return text in self.forms or (self.productive and shape(text) in self.shapes)

    def __repr__(self) -> str:
        how = f"shape {sorted(self.shapes)}" if self.productive else f"{len(self.forms)} memorised forms"
        return f"{self.role}: {self.concept or 'unknown'} ({how}, {self.support} observations)"


def shape(text: str) -> str:
    """A token's shape: letters, digits and punctuation classes, run-length collapsed."""
    out = []
    for ch in text:
        kind = "X" if ch.isupper() else "x" if ch.islower() else "9" if ch.isdigit() else ch
        if not out or out[-1] != kind:
            out.append(kind)
    return "".join(out)


def role_type(role: str, fillers: Sequence[Any], *, subsumes: Callable[[Any], str] | None = None,
              min_forms: int = 5, purity: float = 0.9, others: Sequence[Any] = ()) -> RoleType:
    """The least general type covering every filler, generalised only if productive.

    ``subsumes`` maps a filler to its concept; the induced concept is the most
    specific one covering all fillers. A *shape* is trusted only when it covers
    ``min_forms`` distinct fillers and almost nothing else in ``others``, so
    ``region_0`` generalises to ``region_84`` while ``group`` must be memorised.
    """
    texts = [f if isinstance(f, str) else str(f) for f in fillers]
    concepts = {subsumes(f) for f in fillers} if subsumes else set()
    concept = concepts.pop() if len(concepts) == 1 else None
    by_shape: dict[str, set[str]] = defaultdict(set)
    for text in texts:
        by_shape[shape(text)].add(text)
    foreign: dict[str, set[str]] = defaultdict(set)
    for other in others:
        text = other if isinstance(other, str) else str(other)
        foreign[shape(text)].add(text)
    productive = {
        s for s, forms in by_shape.items()
        if len(forms) >= min_forms and len(forms) >= purity * (len(forms) + len(foreign.get(s, ())))
    }
    memorised = {t for t in texts if shape(t) not in productive}
    return RoleType(role, concept, frozenset(productive), frozenset(memorised), bool(productive), len(texts))
