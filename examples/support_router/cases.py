"""Persistent case log on the record store. Supporting code.

A production host would own persistence and audit; the
example keeps a JSON file so the effect semantics are visible end to end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import tensorcode as tc

from .domain import FreezeCard, InboundEmail, Intent, SendArticle


@dataclass(frozen=True)
class Case:
    message_id: str
    sender: str
    status: Literal["in_progress", "resolved", "needs_human"]
    reason: str
    intent: Intent | None = None
    action: FreezeCard | SendArticle | None = None
    idempotency_key: str | None = None
    receipt_status: str | None = None
    verification: str | None = None
    updated_at: datetime | None = None


REGISTRY = tc.TypeRegistry()
for cls in (Case, Intent, FreezeCard, SendArticle):
    REGISTRY.register(cls)


class CaseLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.store = tc.Store(REGISTRY)
        if path and path.exists():
            self.store, report = tc.Store.from_json(json.loads(path.read_text()), REGISTRY)
            assert report.lossless, report

    def _save(self, case: Case) -> Case:
        case = replace(case, updated_at=datetime.now(timezone.utc))
        self.store.put(tc.Ref(f"case:{case.message_id}"), case)
        if self.path:
            self.path.write_text(json.dumps(self.store.to_json(), indent=1))
        return case

    def get(self, message_id: str) -> Case | None:
        return self.store.entities.get(tc.Ref(f"case:{message_id}"))

    def recent_actions(self, sender: str, *, hours: float) -> tuple[object, ...]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return tuple(
            c.action
            for c in self.store.entities.values()
            if c.sender == sender and c.action and c.status == "resolved" and c.updated_at and c.updated_at >= cutoff
        )

    def needs_human(self, email: InboundEmail, reason: str, unknown: tc.Unknown, intent: Intent | None = None) -> Case:
        return self._save(Case(email.id, email.sender, "needs_human", f"{reason}: {unknown.reason}", intent))

    def begin(self, email: InboundEmail, intent: Intent, action: object, key: str) -> Case:
        return self._save(Case(email.id, email.sender, "in_progress", "invoking", intent, action, key))  # type: ignore[arg-type]

    def finish(self, email: InboundEmail, receipt: tc.Receipt, verdict: tc.Verdict) -> Case:
        case = self.get(email.id)
        assert case is not None
        status = "resolved" if verdict.status == "holds" else "needs_human"
        reason = "; ".join(verdict.reasons)
        return self._save(replace(case, status=status, reason=reason, receipt_status=receipt.status, verification=verdict.status))
