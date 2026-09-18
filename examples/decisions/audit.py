"""Auditing: every decision becomes claims with provenance, and can be replayed.

Two questions an auditor asks about an automated decision:

    "why did you refund this?"      -> ``Audit.explain(ticket_id)`` walks the claim chain:
                                       the decision, the clause it cited, the charge it
                                       cited, and the tier that answered the intent question.
    "would it decide that again?"   -> ``replay`` re-runs the stored input through a runtime
                                       and reports whether the answer matches, so a change in
                                       configuration is detectable rather than invisible.

The store is the ordinary ``tc.Store``; nothing here is bespoke. Decisions are claims, their
evidence names the implementation that produced them, and ``tensorcode.cognition.explain``
prints the chain.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorcode as tc
from tensorcode.cognition import explain
from tensorcode.outcomes import Unknown

from .domain import Eligibility, Ticket

SERVICE = tc.Ref("service:decisions")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Audit:
    """Decisions as claims. One store per service instance; cheap to keep in memory."""

    def __init__(self, store: tc.Store | None = None) -> None:
        self.store = store or tc.Store()
        self.inputs: dict[str, dict] = {}  # ticket id -> the state the decision saw

    # -- recording

    def _tell(self, subject: tc.Ref, predicate: str, value: Any, *, source: str, method: str) -> None:
        self.store.tell(tc.Claim(subject, predicate, value), tc.Evidence(tc.Ref(source), _now(), method=method))

    def record_triage(self, ticket: Ticket, triage: Any) -> None:
        ref = tc.Ref(f"ticket:{ticket.id}")
        self.inputs[ticket.id] = _state_of(ticket)
        source = f"impl:{triage.tier}" if triage.tier else "impl:none"
        self._tell(ref, "text", ticket.text, source="channel:inbound", method="received")
        if isinstance(triage.intent, Unknown):
            self._tell(ref, "intent_unknown", triage.intent.reason, source=source, method="classify")
        else:
            self._tell(ref, "intent", triage.intent.name, source=source, method="classify")
            self._tell(ref, "department", triage.department.value, source="policy:department-map", method="derive")
            self._tell(ref, "urgency", triage.urgency.value, source="policy:urgency-map", method="derive")
        if triage.confidence is not None:
            self._tell(ref, "confidence", (round(triage.confidence.value, 4), triage.confidence.kind, triage.confidence.basis), source=source, method="classify")
        self._tell(ref, "gate", (triage.routing.action, triage.routing.why), source="gate:routing", method="threshold")

    def record_eligibility(self, ticket: Ticket, result: Eligibility | Unknown) -> None:
        ref = tc.Ref(f"ticket:{ticket.id}")
        if isinstance(result, Unknown):
            self._tell(ref, "refund_unknown", result.reason, source="policy:refund", method="walk")
            return
        self._tell(ref, "refund_allowed", result.allowed, source="policy:refund", method="walk")
        for clause in result.clauses:
            self._tell(ref, "cites_clause", clause, source="policy:refund", method="walk")
        if result.charge_id:
            self._tell(ref, "cites_charge", result.charge_id, source="records:statement", method="walk")
        for reason in result.reasons:
            self._tell(ref, "reason", reason, source="policy:refund", method="walk")

    # -- reading back

    def decisions(self, ticket_id: str) -> list[tuple[str, Any]]:
        ref = tc.Ref(f"ticket:{ticket_id}")
        return [(r.claim.predicate, r.claim.object) for r in self.store.claims(ref)]

    def explain(self, ticket_id: str, predicate: str = "refund_allowed") -> list[str]:
        """The claim chain behind one decision, as lines an auditor can read."""
        ref = tc.Ref(f"ticket:{ticket_id}")
        found = self.store.claims(ref, predicate)
        if not found:
            return [f"no {predicate} on record for {ticket_id}"]
        return explain(self.store, found[0].id)

    def why(self, ticket_id: str) -> dict:
        """A flat answer to 'why this decision': the citations and the tier, with sources."""
        ref = tc.Ref(f"ticket:{ticket_id}")
        rows = self.store.claims(ref)
        out: dict[str, Any] = {"ticket": ticket_id, "clauses": [], "charges": [], "reasons": [], "sources": {}}
        for r in rows:
            p, v = r.claim.predicate, r.claim.object
            source = r.evidence[-1].source.id if r.evidence else "?"
            if p == "cites_clause":
                out["clauses"].append(v)
            elif p == "cites_charge":
                out["charges"].append(v)
            elif p == "reason":
                out["reasons"].append(v)
            else:
                out[p] = v
            out["sources"][p] = source
        return out

    def snapshot(self, path: Path) -> None:
        path.write_text(json.dumps({"inputs": self.inputs}, indent=1, default=str))


def _state_of(ticket: Ticket) -> dict:
    """The input a decision saw, stored so it can be replayed exactly."""
    return {
        "id": ticket.id,
        "customer": ticket.customer,
        "text": ticket.text,
        "received_at": ticket.received_at.isoformat(),
        "charges": [asdict(c) if is_dataclass(c) else c for c in ticket.charges],
    }


def replay(state: dict, *, runtime: tc.Runtime) -> dict:
    """Re-decide from stored state under a given runtime.

    Determinism is the library's, not ours: the rules and learned tiers declare
    ``deterministic=True``, so the same input under the same bindings must give the same
    answer. Replaying under *different* bindings is the useful case — it shows what a
    configuration change would have done to decisions already made.
    """
    from datetime import datetime as dt

    from .decisions import handle
    from .domain import Charge, Ticket as T

    charges = tuple(
        Charge(
            c["id"],
            c["merchant"],
            float(c["amount_gbp"]),
            dt.fromisoformat(c["at"]) if isinstance(c["at"], str) else c["at"],
            c.get("status", "settled"),
            c.get("duplicate_of"),
        )
        for c in state.get("charges", ())
    )
    ticket = T(state["id"], state["customer"], state["text"], dt.fromisoformat(state["received_at"]), charges)
    with tc.use(runtime):
        return handle(ticket)
