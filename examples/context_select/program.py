"""Context selection. Runs against ``tensacode`` (the prototype).

Retrieve structured facts and unstructured documents, rank by relevance, remove
redundancy, and pack within a token budget. Both sides of any recorded
contradiction are required evidence: a context that shows only one side is
worse than no context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import tensacode as tc


@dataclass(frozen=True)
class Snippet:
    id: str
    text: str
    source: tc.Ref
    claim_id: str | None = None


def select_context(question: str, subject: tc.Ref, world: tc.Store, documents: Sequence[Snippet], *, budget: int) -> tc.Packed[Snippet] | tc.Unknown:
    facts = [as_snippet(rec) for rec in world.neighborhood(subject).claims]
    ranked = tc.rank(question, facts + list(documents))
    if isinstance(ranked, tc.Unknown):
        return ranked

    contested = {rec.id for conflict in world.conflicts(subject) for rec in (conflict.a, conflict.b)}
    required = [f for f in facts if f.claim_id in contested]
    kept, duplicates = tc.dedupe(ranked, similarity=lambda a, b: tc.shingle_similarity(a.text, b.text), threshold=0.5, keep=lambda s: s in required, key=lambda s: s.id)
    return tc.pack(kept, budget=budget, cost=lambda s: tc.approx_tokens(s.text), required=required, key=lambda s: s.id, dropped=duplicates)


def as_snippet(rec: tc.records.ClaimRecord) -> Snippet:
    c, ev = rec.claim, rec.evidence[0]
    obj = getattr(c.object, "value", c.object)
    start, end = (f"{t:%H:%M}" if t else "…" for t in (c.valid.start, c.valid.end))
    when = f"at {start}" if start == end else f"{start}-{end}"
    return Snippet(rec.id, f"{c.subject} {c.predicate} {obj} {when} per {ev.source}", ev.source, rec.id)
