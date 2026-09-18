"""What an answer rested on, so it can be re-checked later without recomputing it.

A computation declares its dependencies by *reading* them, so recording the reads
recovers the dependency set with no annotation. Revalidating is then one digest
comparison per key read, rather than running the computation again.

Two details carry the value, and both come from ``symbolic-ai-models``'s
``symbolic_ai_core/runtime/certificate.py``:

**Misses are reads.** A key that was absent is recorded with the digest
``MISSING``. "This folder contains nothing" is load-bearing, and a certificate
that records only hits silently fails to notice an *addition* — which in that
repo's one real corpus delta was 411,122 of 744,136 changes.

**Digests, not values.** Revalidation compares one hash per key, so a key whose
value is a large collection costs one comparison instead of touching the whole
thing.

And the honest caveat that module's docstring exists to make: a certificate is
worth ``(cost of recomputing) / (cost of revalidating) × (fraction still valid)``.
For an answer that is itself one pass over its read set — a count, a max — those
two costs are the same and the certificate buys **nothing**. It pays where the
answer was expensive relative to what it depended on: a model call, a fixpoint, a
long agent trace.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..outcomes import Verdict

MISSING = "MISSING"


def value_digest(value: Any) -> str:
    """A short, order-independent digest of a read value."""
    if value is None:
        return MISSING
    try:
        payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    except TypeError:
        payload = repr(value)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _canonical(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {"$map": sorted(([str(k), _canonical(v)] for k, v in value.items()), key=repr)}
    if isinstance(value, (set, frozenset)):
        return {"$set": sorted((_canonical(v) for v in value), key=repr)}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return repr(value)


@dataclass(frozen=True)
class ReadSet:
    """The keys an answer consulted, with a digest each. Absent keys are included."""

    reads: tuple[tuple[str, str], ...] = ()
    at: float = field(default_factory=time.time)
    note: str = ""

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(k for k, _ in self.reads)

    @property
    def misses(self) -> tuple[str, ...]:
        return tuple(k for k, d in self.reads if d == MISSING)

    def digest(self) -> str:
        return value_digest(sorted(self.reads))

    def to_dict(self) -> dict[str, Any]:
        return {"reads": [list(r) for r in self.reads], "at": self.at, "note": self.note}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReadSet":
        return cls(tuple((k, d) for k, d in (tuple(r) for r in data.get("reads", ()))),
                   data.get("at", 0.0), data.get("note", ""))

    def __repr__(self) -> str:
        return f"ReadSet({len(self.reads)} reads, {len(self.misses)} absent)"


class Reader:
    """A fact source that records what was consulted, including what was not there.

        reader = Reader(facts)
        rules.predict(reader)          # or any code that calls .get()/.has()
        certificate = reader.readset()
    """

    def __init__(self, facts: Mapping[str, Any] | frozenset, *, note: str = "") -> None:
        self.facts: Mapping[str, Any] = dict(facts) if isinstance(facts, frozenset) else facts
        self.note = note
        self._reads: dict[str, str] = {}

    # the mapping surface a Literal or a rule uses
    def get(self, key: str, default: Any = None) -> Any:
        present = key in self.facts
        value = self.facts[key] if present else default
        self._reads.setdefault(key, value_digest(self.facts[key]) if present else MISSING)
        return value

    def __contains__(self, key: str) -> bool:
        self._reads.setdefault(key, value_digest(self.facts[key]) if key in self.facts else MISSING)
        return key in self.facts

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def items(self) -> Iterable[tuple[str, Any]]:
        """A full scan is a read of every key, and is recorded as such."""
        for key, value in self.facts.items():
            self._reads.setdefault(key, value_digest(value))
        return self.facts.items()

    def readset(self) -> ReadSet:
        return ReadSet(tuple(sorted(self._reads.items())), note=self.note)


def revalidate(certificate: ReadSet, facts: Mapping[str, Any] | frozenset) -> Verdict:
    """Does the answer still hold? One digest comparison per key that was read."""
    table: Mapping[str, Any] = dict(facts) if isinstance(facts, frozenset) else facts
    changed: list[str] = []
    for key, digest in certificate.reads:
        now = value_digest(table[key]) if key in table else MISSING
        if now != digest:
            changed.append(f"{key}: {digest} -> {now}")
    if changed:
        return Verdict("fails", tuple(changed), tuple(certificate.keys))
    return Verdict("holds", (f"{len(certificate.reads)} reads unchanged",), tuple(certificate.keys))


def certified(answer_fn: Callable[[Reader], Any], facts: Mapping[str, Any] | frozenset,
              *, note: str = "") -> tuple[Any, ReadSet]:
    """Run something over the facts and return its answer with its certificate."""
    reader = Reader(facts, note=note)
    return answer_fn(reader), reader.readset()
