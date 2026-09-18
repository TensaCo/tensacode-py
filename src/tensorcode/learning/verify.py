"""Whether an induced artifact may be adopted, and the controls that say what it means.

A fit is not a finding. Both source repos record the same lesson from opposite
directions: ``typed-crystallization-networks`` found a mined abstraction helping
exactly as much as a *wrong* abstraction of the same size (research §44), and
``symbolic-ai-models`` states it as a rule — "same-cardinality-but-meaningless is
this repo's most productive instrument … it corrected four separate published
results" (``symbolic_ai_lean/gate.py``). So an artifact is measured against four
controls, in the shape that file pre-registered:

``held_out``
    Accuracy on cases the inducer never saw. A rule admitted on its training fit
    alone is the failure mode TCN's §65 records.
``random``
    The same artifact with its decisions shuffled among the same labels: a
    same-cardinality, meaningless competitor. Beating it is the minimum.
``shifted``
    The artifact applied to the *wrong question* — cases relabelled from a
    different pool. Its accuracy must fall to the random level; if it does not,
    the artifact is reading something other than the question.
``renamed``
    Every symbol consistently renamed. The verdicts must be **identical**. An
    artifact that moves under renaming is reading vocabulary, which is how
    ``ensemble-001``'s headline result died (0.467 → 0.038).

Adoption also requires a **simpler competitor** to lose: a one-condition rule, or
the majority label. Nothing is adopted silently, and nothing is rejected silently
either — :class:`Verification` records the numbers and the reason.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..outcomes import Verdict
from .induce import DecisionList, Rule, decision_list
from .literals import Case, Literal, rename_facts, rename_map


@dataclass(frozen=True)
class Verification:
    """What the controls said, and whether that is enough to adopt."""

    held_out: float
    train: float
    random: float
    shifted: float
    renamed_identical: bool
    floor: float  # the best simple competitor (majority label, or a one-condition rule)
    reasons: tuple[str, ...] = ()

    @property
    def verdict(self) -> Verdict:
        return Verdict("holds" if not self.reasons else "fails", self.reasons)

    @property
    def adopted(self) -> bool:
        return not self.reasons

    def report(self) -> str:
        rows = [
            f"held-out      {self.held_out:.3f}",
            f"train         {self.train:.3f}",
            f"floor         {self.floor:.3f}  (majority or one condition)",
            f"random        {self.random:.3f}  (same labels, shuffled)",
            f"shifted       {self.shifted:.3f}  (wrong question; should fall to random)",
            f"renamed       {'identical' if self.renamed_identical else 'CHANGED — reading vocabulary'}",
            f"verdict       {'adopt' if self.adopted else 'reject: ' + '; '.join(self.reasons)}",
        ]
        return "\n".join(rows)


def verify_decision_list(
    artifact: DecisionList,
    *,
    train: Sequence[Case],
    held_out: Sequence[Case],
    literals: Sequence[Literal],
    shifted: Sequence[Case] = (),
    margin: float = 0.02,
    seed: int = 0,
) -> Verification:
    """Run the four controls plus the simpler-competitor floor over an induced list."""
    reasons: list[str] = []
    train_acc = artifact.accuracy(train)
    held = artifact.accuracy(held_out)

    # floor: the majority label, and the best single-condition rule
    labels = [label for _, label in train]
    majority = max(set(labels), key=labels.count) if labels else None
    floor = sum(label == majority for _, label in held_out) / max(1, len(held_out))
    # the "one-line hand rule" competitor: a single condition plus a default. TCN's
    # findings record learned priors being matched by exactly this.
    one = decision_list(train, literals, max_conditions=1, mdl=False, max_rules=1)
    floor = max(floor, one.accuracy(held_out))

    # random: the same rules with their labels shuffled among the same multiset. One
    # shuffle can come back as the identity, so this is the mean over several — the
    # control is an *expected* accuracy, not a single draw.
    rng = random.Random(seed)
    original = [r.label for r in artifact.rules]
    draws: list[float] = []
    for _ in range(20 if len(original) > 1 else 0):
        shuffled = list(original)
        rng.shuffle(shuffled)
        if shuffled == original:
            continue
        scrambled = DecisionList([Rule(r.conditions, label, r.support, r.correct)
                                  for r, label in zip(artifact.rules, shuffled)], artifact.default)
        draws.append(scrambled.accuracy(held_out))
    random_acc = sum(draws) / len(draws) if draws else floor

    # shifted: the right artifact, the wrong question
    shifted_acc = artifact.accuracy(shifted) if shifted else random_acc

    # renamed: the *same* artifact, run on consistently renamed cases. Renaming the
    # artifact too would hide exactly the failure this control exists to catch — an
    # artifact that is reading vocabulary rather than structure.
    mapping = rename_map(list(train) + list(held_out))
    identical = all(artifact.predict(rename_facts(facts, mapping)) == artifact.predict(facts)
                    for facts, _ in held_out)

    if held < floor + margin:
        reasons.append(f"held-out {held:.3f} does not beat the simple floor {floor:.3f}")
    if held < random_acc + margin:
        reasons.append(f"held-out {held:.3f} does not beat the same-size random control {random_acc:.3f}")
    if shifted and shifted_acc > random_acc + 0.1:
        reasons.append(f"scores {shifted_acc:.3f} on the wrong question: it is not reading the question")
    if not identical:
        reasons.append("verdicts change under renaming: it is reading vocabulary, not structure")
    return Verification(held, train_acc, random_acc, shifted_acc, identical, floor, tuple(reasons))


# ------------------------------------------------------------ concept adoption


@dataclass(frozen=True)
class Concept:
    """A proposed predicate: a name, a definition over existing claims, and its evidence."""

    name: str
    definition: tuple[Literal, ...]
    support: int
    functional: bool = False

    def holds(self, facts: Any) -> bool:
        return all(c.holds(facts) for c in self.definition)

    def __repr__(self) -> str:
        return f"{self.name} ⇔ " + " ∧ ".join(repr(c) for c in self.definition)


@dataclass(frozen=True)
class ConceptCheck:
    """Whether a proposed concept may join the vocabulary."""

    concept: Concept
    covers: int
    round_trips: bool
    renamed_identical: bool
    conflicts: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def adopted(self) -> bool:
        return not self.reasons


def check_concept(concept: Concept, *, positives: Sequence[frozenset], negatives: Sequence[frozenset],
                  functional_values: Callable[[frozenset], Any] | None = None,
                  min_support: int = 3) -> ConceptCheck:
    """A concept is adopted only if it separates, round-trips, and survives renaming.

    * it must hold of the positives and not of the negatives (it has content);
    * **round trip**: re-describing a case with the concept must not lose what the
      concept was defined from, so the original claims can still be derived;
    * renaming symbols must not change which cases it covers;
    * a functional predicate must not give one subject two values.
    """
    reasons: list[str] = []
    covers = sum(concept.holds(facts) for facts in positives)
    leaks = sum(concept.holds(facts) for facts in negatives)
    if covers < min_support:
        reasons.append(f"covers only {covers} cases (needs {min_support})")
    if leaks:
        reasons.append(f"also holds of {leaks} negative cases: it does not separate them")

    # round trip: the definition's own predicates must still be present in the case
    round_trips = all(all(c.holds(facts) for c in concept.definition) for facts in positives if concept.holds(facts))
    if not round_trips:
        reasons.append("re-describing loses the claims it was defined from")

    mapping = rename_map([(facts, None) for facts in list(positives) + list(negatives)])
    renamed_positives = [rename_facts(facts, mapping) for facts in positives]
    identical = [concept.holds(f) for f in positives] == [concept.holds(f) for f in renamed_positives]
    if not identical:
        reasons.append("coverage changes under renaming")

    conflicts: list[str] = []
    if concept.functional and functional_values is not None:
        seen: dict[Any, Any] = {}
        for facts in positives:
            if not concept.holds(facts):
                continue
            key = tuple(sorted(facts, key=repr))
            value = functional_values(facts)
            if key in seen and seen[key] != value:
                conflicts.append(f"two values for one subject: {seen[key]!r} and {value!r}")
            seen[key] = value
        if conflicts:
            reasons.append("contradiction on a functional predicate")
    return ConceptCheck(concept, covers, round_trips, identical, tuple(conflicts), tuple(reasons))


def propose_concepts(cases: Sequence[Case], literals: Sequence[Literal], *, label: Any,
                     max_conditions: int = 2, top: int = 3) -> list[Concept]:
    """Candidate definitions for "what these cases have in common", purest first.

    Grouping is by *behaviour* — which cases a definition covers — not by syntax, so
    two spellings of one concept collapse into a single candidate. TCN's research
    §46 found one real concept arriving as 8–12 syntactically different copies.
    """
    positives = [facts for facts, y in cases if y == label]
    negatives = [facts for facts, y in cases if y != label]
    if not positives:
        return []
    scored: dict[tuple, tuple[float, Concept]] = {}
    for size in range(1, max_conditions + 1):
        for combo in _combinations(literals, size):
            covered = tuple(i for i, facts in enumerate(positives) if all(c.holds(facts) for c in combo))
            if not covered:
                continue
            leaks = sum(all(c.holds(facts) for c in combo) for facts in negatives)
            purity = len(covered) / (len(covered) + leaks)
            concept = Concept(f"{label}_like", tuple(combo), len(covered))
            best = scored.get(covered)
            cost = sum(c.cost() for c in combo)
            if best is None or (purity, -cost) > (best[0], -sum(c.cost() for c in best[1].definition)):
                scored[covered] = (purity, concept)
    ranked = sorted(scored.values(), key=lambda row: (-row[0], -row[1].support))
    return [concept for _, concept in ranked[:top]]


def _combinations(items: Sequence[Literal], size: int) -> Any:
    if size == 1:
        for item in items:
            yield (item,)
        return
    for i, first in enumerate(items):
        for rest in _combinations(items[i + 1:], size - 1):
            yield (first, *rest)
