"""Small memory helpers shared by the assistant's parts: state, knowledge, and derivation."""

from __future__ import annotations

from datetime import datetime, timezone

import tensacode as tc
from tensacode.cognition import Thought, integrate
from tensacode.records import Evidence, Patch, Retract, Tell

from ..mind import knowledge


def now() -> datetime:
    return datetime.now(timezone.utc)


def set_state(mind: tc.Store, subject: tc.Ref, predicate: str, value: object, source: str) -> Thought:
    """Replace a subject's single current value for ``predicate`` (lifecycle state, focus, a step's program counter)."""
    old = mind.claims(subject, predicate)
    if len(old) == 1 and old[0].claim.object == value:
        return Thought()
    edits = [Retract(r.id, "superseded") for r in old] + [Tell(tc.Claim(subject, predicate, value), (Evidence(tc.Ref(source), now(), method="assistant"),))]
    commit = mind.apply(Patch(tuple(edits), mind.revision))
    thought = Thought(tuple(mind.claim(i) for i in commit.added), tuple(mind.claim(i) for i in commit.retracted))
    mind.forget(commit.retracted)
    return thought


def unremember(mind: tc.Store, subject: tc.Ref, predicate: str | None, reason: str) -> Thought:
    """Drop what was remembered (a told fact you asked me to forget), keeping the reason."""
    records = mind.claims(subject) if predicate in (None, "*", "") else mind.claims(subject, predicate)
    if not records:
        return Thought()
    commit = mind.apply(Patch(tuple(Retract(r.id, reason) for r in records), mind.revision))
    thought = Thought((), tuple(mind.claim(i) for i in commit.retracted))
    mind.forget(commit.retracted)
    return thought


def note(mind: tc.Store, claims: list[tc.Claim], source: str) -> Thought:
    return integrate(mind, knowledge([(c, None) for c in claims], source, "assistant"))


def derive(mind: tc.Store, claims: list[tc.Claim], source: str, premises: tuple[str, ...], method: str = "procedure") -> Thought:
    """Claims that follow from other claims: ``explain`` can then walk from here back to them."""
    if not claims:
        return Thought()
    evidence = (Evidence(tc.Ref(source), now(), method=method, derived_from=premises),)
    commit = mind.apply(Patch(tuple(Tell(c, evidence) for c in claims), mind.revision))
    return Thought(tuple(mind.claim(i) for i in commit.added), ())
