"""Wants: an open question as a first-class object, and the standing pull toward answering it.

``Unknown`` says an operation could not answer. That is where it stops, which is why a
mind built on it goes quiet instead of going looking. A ``Want`` is the same admission with
its consequences attached: what would satisfy it, what provenance the answer must have, and
which candidate satisfiers exist — a memory to search, a modality to look at, a command to
run, a person to ask, a derivation to attempt.

The runtime can then rank wants by what an answer is worth against what it costs, and hand
the chosen one to whoever can satisfy it. Abstention stays honest at both ends: looking up
an answer refuses to pick between disagreeing claims, refuses hearsay where the want asked
for an observation, and a want nothing can satisfy simply stays open.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from .outcomes import Score, Unknown
from .records import ClaimRecord, Ref, Store

SatisfierKind = Literal["memory", "perceive", "command", "ask", "derive"]

#: which modalities count as first-hand, for a want that requires an observation
FIRSTHAND = ("pixels", "structure", "text")


@dataclass(frozen=True)
class Satisfier:
    """One way this want might be answered, with what it would cost to try."""

    kind: SatisfierKind
    detail: str  # a place to look, a command to run, a question to put to someone
    cost: float = 1.0  # caller's own units (seconds, tokens, keystrokes)
    odds: Score = field(default_factory=lambda: Score(0.5, "uncalibrated"))

    @property
    def worth(self) -> float:
        return self.odds.value / max(self.cost, 1e-6)


@dataclass(frozen=True)
class Want:
    """A question the mind is holding open."""

    question: str
    subject: Ref | None = None
    predicate: str | None = None
    expect: type | None = None  # what kind of value would satisfy it
    requires: str | None = None  # provenance demand: "observed", "not-hearsay", or a modality name
    satisfiers: tuple[Satisfier, ...] = ()
    value: float = 1.0  # how much an answer is worth
    asked_by: Ref | None = None  # who wants to know (the user, a rule, a procedure)

    @property
    def id(self) -> str:
        key = f"{self.question}|{self.subject}|{self.predicate}|{self.requires}"
        return "want:" + hashlib.sha256(key.encode()).hexdigest()[:12]

    def best_satisfier(self) -> Satisfier | None:
        return max(self.satisfiers, key=lambda s: (s.worth, s.kind), default=None)


@dataclass(frozen=True)
class Answer:
    """A want satisfied, with the claims that answer it and how they were known."""

    want_id: str
    value: Any
    claims: tuple[str, ...]
    modality: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.value!r} (from {len(self.claims)} claim{'s' * (len(self.claims) != 1)})"


def want_from(unknown: Unknown, question: str, *, subject: Ref | None = None, predicate: str | None = None,
              satisfiers: Iterable[Satisfier] = (), value: float = 1.0) -> Want:
    """Turn a dead-end ``Unknown`` into a want, keeping its reason and any candidates it had."""
    extra = tuple(Satisfier("derive", f"reconsider {cand!r}", cost=0.5, odds=score) for cand, score in unknown.candidates)
    return Want(question=question, subject=subject, predicate=predicate,
                satisfiers=tuple(satisfiers) + extra, value=value,
                asked_by=Ref(f"unknown:{unknown.reason}"))


class Wants:
    """The standing set of open questions."""

    def __init__(self) -> None:
        self.open: dict[str, Want] = {}
        self.answered: dict[str, Answer] = {}
        self.abandoned: dict[str, str] = {}

    def add(self, want: Want) -> str:
        if want.id not in self.answered:
            self.open[want.id] = want
        return want.id

    def drop(self, want_id: str, reason: str) -> None:
        self.open.pop(want_id, None)
        self.abandoned[want_id] = reason

    def satisfied(self, want_id: str, answer: Answer) -> Answer:
        self.open.pop(want_id, None)
        self.answered[want_id] = answer
        return answer

    # -- ranking

    def ranked(self) -> list[tuple[Want, Satisfier | None, float]]:
        """Open wants by what answering them is worth against what it would cost."""
        rows = []
        for want in self.open.values():
            best = want.best_satisfier()
            rows.append((want, best, want.value * (best.worth if best else 0.0)))
        return sorted(rows, key=lambda row: (-row[2], row[0].id))

    def next_to_pursue(self) -> tuple[Want, Satisfier] | None:
        for want, satisfier, _ in self.ranked():
            if satisfier is not None:
                return want, satisfier
        return None

    # -- answering from memory

    def look_up(self, want: Want, mind: Store, *, frames: Any = None) -> Answer | Unknown:
        """Try to answer from what the mind already holds. Refuses to guess."""
        if want.subject is None and want.predicate is None:
            return Unknown("want_underspecified", f"{want.question!r} names neither a subject nor a predicate")
        records = mind.claims(subject=want.subject, predicate=want.predicate)
        if not records:
            return Unknown("not_in_memory", f"nothing recorded for {want.question!r}")
        allowed = [r for r in records if _provenance_ok(r, want.requires, frames)]
        if not allowed:
            kinds = sorted({m for r in records for m in _modalities(r, frames)})
            return Unknown("provenance_unmet", f"{want.question!r} needs {want.requires}; have only {', '.join(kinds) or 'unknown'}",
                           candidates=tuple((r.claim.object, Score(0.5, "uncalibrated")) for r in records))
        if want.expect is not None:
            allowed = [r for r in allowed if isinstance(r.claim.object, want.expect)]
            if not allowed:
                return Unknown("wrong_type", f"{want.question!r} expects {want.expect.__name__}")
        distinct = {repr(r.claim.object): r for r in allowed}
        if len(distinct) > 1:
            return Unknown("disagreement", f"memory holds {len(distinct)} different answers to {want.question!r}",
                           candidates=tuple((r.claim.object, Score(1 / len(distinct), "vote_share")) for r in distinct.values()))
        record = next(iter(distinct.values()))
        return Answer(want.id, record.claim.object, tuple(sorted(r.id for r in allowed)), _modalities(record, frames))

    def pursue_from_memory(self, mind: Store, *, frames: Any = None) -> list[Answer]:
        """Answer every open want memory can already settle; leave the rest open."""
        out = []
        for want in list(self.open.values()):
            found = self.look_up(want, mind, frames=frames)
            if isinstance(found, Answer):
                out.append(self.satisfied(want.id, found))
        return out

    def as_unknown(self, want: Want) -> Unknown:
        """What to hand a caller that wanted an answer now: honest, with what we would try next."""
        best = want.best_satisfier()
        detail = f"{want.question}; next I would {best.kind} ({best.detail})" if best else f"{want.question}; nothing I can do would answer it"
        return Unknown("open_want", detail)


def _modalities(rec: ClaimRecord, frames: Any) -> tuple[str, ...]:
    if frames is not None:
        binding = frames.binding(rec.id)
        if binding is not None:
            return binding.modality
    from .frames import modality_of

    return modality_of(rec)


def _provenance_ok(rec: ClaimRecord, requires: str | None, frames: Any) -> bool:
    if requires is None:
        return True
    mods = _modalities(rec, frames)
    if requires == "observed":
        return any(m in FIRSTHAND for m in mods)
    if requires == "not-hearsay":
        return "hearsay" not in mods
    return requires in mods
