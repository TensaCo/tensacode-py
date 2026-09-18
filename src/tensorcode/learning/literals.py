"""The hypothesis language: conditions over claims, generated from what was observed.

Induction needs a space of candidate conditions, and the honest way to get one is
to read it off the data rather than to write it down: every literal below is
generated from claims that actually occurred. Nothing here names a domain.

A literal costs bits (:meth:`Literal.cost`), which is what lets an induced rule
be charged for its own description length instead of growing until it fits.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..records import Claim, Ref, Store

#: One case for induction: the claims that held, and the outcome to predict.
Case = tuple[frozenset[tuple[str, Any]], Any]


@dataclass(frozen=True)
class Literal:
    """A readable condition over one case's facts."""

    predicate: str
    value: Any = None
    negated: bool = False
    kind: str = "equals"  # "equals" | "present" | "at_least"

    def holds(self, facts: Mapping[str, Any] | frozenset) -> bool:
        table = dict(facts) if isinstance(facts, frozenset) else facts
        got = table.get(self.predicate, _MISSING)
        if self.kind == "present":
            out = got is not _MISSING
        elif self.kind == "at_least":
            out = got is not _MISSING and isinstance(got, (int, float)) and got >= self.value
        else:
            out = got == self.value
        return not out if self.negated else out

    def cost(self) -> int:
        """Bits-ish: a name, a test, and a value."""
        return 2 + (0 if self.value is None else 1) + (1 if self.negated else 0)

    def __repr__(self) -> str:
        body = {"present": f"has({self.predicate})",
                "at_least": f"{self.predicate}>={self.value}",
                "equals": f"{self.predicate}={self.value!r}"}[self.kind]
        return f"¬{body}" if self.negated else body


class _Missing:
    def __repr__(self) -> str:
        return "∅"


_MISSING = _Missing()


def facts_of(store: Store, subject: Ref, *, predicates: Sequence[str] | None = None) -> frozenset[tuple[str, Any]]:
    """One subject's live claims as a flat fact table (missing predicates stay missing)."""
    out = {}
    for record in store.claims(subject):
        if predicates is None or record.claim.predicate in predicates:
            out[record.claim.predicate] = record.claim.object
    return frozenset(out.items())


def candidate_literals(cases: Sequence[Case], *, min_count: int = 2, max_values: int = 12) -> list[Literal]:
    """Every condition worth trying, from the values that actually occur.

    A predicate seen with many distinct values contributes a presence test rather
    than one literal per value, which keeps the space from exploding on identifiers.
    """
    values: dict[str, Counter] = defaultdict(Counter)
    for facts, _ in cases:
        for predicate, value in facts:
            try:
                values[predicate][value] += 1
            except TypeError:  # unhashable objects are still worth a presence test
                values[predicate]["<unhashable>"] += 1
    out: list[Literal] = []
    for predicate, counts in sorted(values.items()):
        out.append(Literal(predicate, kind="present"))
        out.append(Literal(predicate, kind="present", negated=True))
        if len(counts) > max_values:
            continue
        for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], repr(kv[0]))):
            if count < min_count or value == "<unhashable>":
                continue
            out.append(Literal(predicate, value))
            out.append(Literal(predicate, value, negated=True))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out.append(Literal(predicate, value, kind="at_least"))
    return out


def rename_case(case: Case, mapping: Mapping[Any, Any]) -> Case:
    """A case with its symbols consistently renamed — the input to the rename control."""
    facts, label = case
    renamed = frozenset((p, mapping.get(v, v)) for p, v in facts)
    return renamed, mapping.get(label, label)


def rename_facts(facts: frozenset, mapping: Mapping[Any, Any]) -> frozenset:
    """Rename the symbols *inside* a case, leaving its label alone.

    The renaming control asks whether an artifact's output is unchanged when the
    vocabulary changes. Renaming the labels too would make every artifact fail it,
    which measures nothing.
    """
    return frozenset((p, mapping.get(v, v)) for p, v in facts)


def rename_map(cases: Sequence[Case], *, prefix: str = "sym", labels: bool = False) -> dict[Any, Any]:
    """A consistent, order-independent renaming of the symbols in the cases."""
    seen: list[Any] = []
    for facts, label in cases:
        for _, value in facts:
            if isinstance(value, str) and value not in seen:
                seen.append(value)
        if labels and isinstance(label, str) and label not in seen:
            seen.append(label)
    return {value: f"{prefix}{i}" for i, value in enumerate(sorted(seen))}
