"""Event time and ordering, so tense stops being a label and starts being a question.

The grammar reads tense and aspect and puts them in a feature; the projection then drops
them. So "the grain arrived" and "the grain will arrive" become the same claim, and
"what did you do before that" has nothing to stand on. Here a parsed tense becomes an
:class:`~tensorcode.records.Interval` on the claim, and events get times that can be
ordered, so the graph answers *when* and *in what order*.

Two clocks are kept apart, because conflating them is how a record of what was said
becomes a record of what happened:

* ``observed_at`` on evidence: when a source said it. Already in the store.
* the event's own time: when the thing happened. That is what this module adds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from .outcomes import Unknown
from .records import Claim, ClaimRecord, Evidence, Interval, Ref, Store

#: how a connective relates the clause it introduces to the main clause
CONNECTIVES: dict[str, str] = {
    "before": "before", "after": "after", "since": "after", "until": "before",
    "while": "during", "when": "during", "whenever": "during", "once": "after",
    "then": "after", "earlier": "before", "later": "after", "meanwhile": "during",
}

AT = "happened_at"
ENDED = "ended_at"


def interval_for(features: Mapping[str, Any], now: datetime) -> Interval:
    """The interval a tense/aspect commits to, relative to ``now``.

    Deliberately coarse: past means "ended before now", future means "starts after now",
    present means "holds now". A coarse interval that is true beats a precise one invented.
    """
    tense, aspect = features.get("tense"), features.get("aspect")
    if tense == "past":
        return Interval(None, now) if aspect != "perfect" else Interval(None, now)
    if tense == "future":
        return Interval(now, None)
    if tense == "present":
        return Interval(now, now) if aspect != "progressive" else Interval(now, None)
    return Interval()


@dataclass(frozen=True)
class Event:
    ref: Ref
    at: datetime
    kind: str | None = None

    def describe(self) -> str:
        return f"{self.ref.id}{f' ({self.kind})' if self.kind else ''} at {self.at.isoformat(timespec='seconds')}"


def tell_event(mind: Store, ref: Ref, *, at: datetime, kind: str | None = None, source: Ref,
               ended: datetime | None = None, method: str = "temporal") -> Ref:
    """Record when an event happened (as distinct from when it was reported)."""
    evidence = Evidence(source=source, observed_at=datetime.now(timezone.utc), method=method)
    mind.tell(Claim(ref, AT, at, valid=Interval(at, ended or at)), evidence)
    if kind:
        mind.tell(Claim(ref, "is_a", kind, valid=Interval(at, ended or at)), evidence)
    if ended:
        mind.tell(Claim(ref, ENDED, ended, valid=Interval(at, ended)), evidence)
    return ref


def event_time(mind: Store, ref: Ref) -> datetime | Unknown:
    """When an event happened, by its own clock; falls back to nothing, never to a guess."""
    for record in mind.claims(ref, AT):
        value = record.claim.object
        if isinstance(value, datetime):
            return value
    return Unknown("no_event_time", f"{ref.id} has no recorded time of happening")


def events(mind: Store, *, kind: str | None = None) -> list[Event]:
    """Every event with a recorded time, earliest first."""
    out: list[Event] = []
    for record in mind.claims(predicate=AT):
        value = record.claim.object
        if not isinstance(value, datetime):
            continue
        ref = record.claim.subject
        of = next((r.claim.object for r in mind.claims(ref, "is_a")), None)
        if kind is not None and of != kind:
            continue
        out.append(Event(ref, value, of if isinstance(of, str) else None))
    return sorted(out, key=lambda e: (e.at, e.ref.id))


def relate(mind: Store, a: Ref, b: Ref, *, tolerance: timedelta = timedelta(0)) -> str | Unknown:
    """'before', 'after' or 'simultaneous' — or a refusal when either time is unrecorded."""
    ta, tb = event_time(mind, a), event_time(mind, b)
    if isinstance(ta, Unknown):
        return ta
    if isinstance(tb, Unknown):
        return tb
    if abs(ta - tb) <= tolerance:
        return "simultaneous"
    return "before" if ta < tb else "after"


def order(mind: Store, refs: Iterable[Ref]) -> tuple[list[Ref], list[Ref]]:
    """Refs sorted by their event time, and the ones that have no time to sort by."""
    timed: list[tuple[datetime, Ref]] = []
    untimed: list[Ref] = []
    for ref in refs:
        at = event_time(mind, ref)
        (untimed if isinstance(at, Unknown) else timed).append(ref if isinstance(at, Unknown) else (at, ref))  # type: ignore[arg-type]
    return [ref for _, ref in sorted(timed, key=lambda pair: (pair[0], pair[1].id))], untimed


def before(mind: Store, ref: Ref, *, kind: str | None = None) -> list[Event]:
    """Events that happened before this one."""
    at = event_time(mind, ref)
    return [] if isinstance(at, Unknown) else [e for e in events(mind, kind=kind) if e.at < at and e.ref != ref]


def after(mind: Store, ref: Ref, *, kind: str | None = None) -> list[Event]:
    at = event_time(mind, ref)
    return [] if isinstance(at, Unknown) else [e for e in events(mind, kind=kind) if e.at > at and e.ref != ref]


def during(mind: Store, start: datetime, end: datetime, *, kind: str | None = None) -> list[Event]:
    return [e for e in events(mind, kind=kind) if start <= e.at <= end]


def first_seen(record: ClaimRecord) -> datetime | None:
    """The earliest moment any source put this claim on record."""
    times = [e.observed_at for e in record.evidence]
    return min(times) if times else None


def changed_since(mind: Store, when: datetime, *, scope: Ref | None = None) -> list[Claim]:
    """Claims that came on record after ``when`` — what is new since a moment.

    This reads the *reporting* clock, which is the right one for "what changed since my
    last message": the question is about the record, not about when the world moved.
    """
    out: list[tuple[datetime, Claim]] = []
    for record in mind.claims(scope=scope) if scope is not None else mind.claims():
        seen = first_seen(record)
        if seen is not None and seen > when:
            out.append((seen, record.claim))
    return [claim for _, claim in sorted(out, key=lambda pair: pair[0])]


def since(mind: Store, ref: Ref, *, kind: str | None = None) -> list[Event] | Unknown:
    """Events after a named event — "what happened since the commit"."""
    at = event_time(mind, ref)
    return at if isinstance(at, Unknown) else after(mind, ref, kind=kind)


def connective_relation(word: str) -> str | Unknown:
    """What ordering a connective asserts between its clause and the main one."""
    got = CONNECTIVES.get(word.strip().lower())
    return got if got else Unknown("not_a_temporal_connective", word)


def tell_order(mind: Store, earlier: Ref, later: Ref, *, source: Ref, relation: str = "before",
               method: str = "temporal:connective") -> Claim:
    """Record an ordering asserted by language, when neither event has a clock time.

    "the grain arrived before the snow came" orders two events without dating either, and
    that is worth keeping: it answers ordering questions that timestamps cannot.
    """
    claim = Claim(earlier, relation, later)
    mind.tell(claim, Evidence(source=source, observed_at=datetime.now(timezone.utc), method=method))
    return claim


def ordered_by_claims(mind: Store, a: Ref, b: Ref) -> str | Unknown:
    """Ordering from asserted ``before``/``after`` claims, for undated events."""
    if mind.claims(a, "before", b):
        return "before"
    if mind.claims(b, "before", a):
        return "after"
    if mind.claims(a, "after", b):
        return "after"
    if mind.claims(b, "after", a):
        return "before"
    return Unknown("no_recorded_order", f"nothing orders {a.id} and {b.id}")
