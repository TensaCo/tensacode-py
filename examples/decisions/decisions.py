"""The five decisions, as ordinary functions a request handler calls.

There is no agent loop here. Code owns control flow; each function asks one or more typed
questions, gates the answer on measured confidence, and records what it decided and why.

    triage(ticket)                  -> Triage       (department, urgency, refund asked)
    refund_eligibility(ticket, ...) -> Eligibility | Unknown
    route(ticket)                   -> which tier answered, and what it cost
    rerank(query, passages)         -> [(Passage, Score)] | Unknown
    supports(claim, passage)        -> Verdict       (does this passage back the claim?)

Every answer is either a typed value or ``Unknown``. Nothing returns a default when it does
not know, and nothing here calls a language model: the runtime decides what runs, and the
trace records which tier answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import tensorcode as tc
from tensorcode.outcomes import Score, Unknown, Verdict
from tensorcode.runtime import current

from ..support_router.domain import Intent
from .audit import Audit
from .domain import (
    DEPARTMENT_OF_INTENT,
    REFUND_INTENTS,
    REFUND_POLICY,
    Charge,
    Department,
    Eligibility,
    Passage,
    PolicyClause,
    Ticket,
    Urgency,
    urgency_of,
)
from .gating import Decided, Gate, gates


@dataclass(frozen=True)
class Triage:
    """One pass over a ticket: several bounded questions, each separately gated."""

    ticket_id: str
    intent: Intent | Unknown
    department: Department | Unknown
    urgency: Urgency | Unknown
    refund_asked: Verdict
    routing: Decided
    confidence: Score | None
    tier: str | None
    ms: float


def triage(ticket: Ticket, *, gate: Gate | None = None, audit: Audit | None = None) -> Triage:
    """Classify intent once, then derive the answers that follow from it.

    The intent question is the only one put to a backend; department and urgency follow from
    it by declared mapping, so they inherit its confidence rather than inventing their own.
    An unknown intent leaves all three unknown — it does not fall back to ``general``.
    """
    intent, score, span = classify_with_confidence(ticket.text, Intent)
    tier = span.answered_by if span else None
    gate = gate or gates()["routing"]
    decided = gate.decide(intent, score)

    if isinstance(intent, Unknown):
        department: Department | Unknown = intent
        urgency: Urgency | Unknown = intent
        refund = Verdict("unknown", (f"intent unknown: {intent.reason}",))
    else:
        department = DEPARTMENT_OF_INTENT[intent.name]
        urgency = urgency_of(intent)
        refund = (
            Verdict("holds", (f"intent {intent.name} is a refund request",), (intent,))
            if intent.name in REFUND_INTENTS
            else Verdict("fails", (f"intent {intent.name} does not ask for money back",), (intent,))
        )

    out = Triage(ticket.id, intent, department, urgency, refund, decided, score, tier, span.total_ms if span else 0.0)
    if audit is not None:
        audit.record_triage(ticket, out)
    return out


def refund_eligibility(
    ticket: Ticket,
    *,
    policy: Sequence[PolicyClause] = REFUND_POLICY,
    now: datetime | None = None,
    audit: Audit | None = None,
) -> Eligibility | Unknown:
    """May this refund be paid without a human?

    This one is deliberately *not* a model question. The policy is data, the charges are
    records, and the answer is a walk over both — so the reasons name clauses and charge ids
    and an auditor can check them. ``check`` is used for the one judgement that is genuinely
    about evidence: whether a duplicate exists.
    """
    now = now or datetime.now(timezone.utc)
    clauses = {c.id: c for c in policy}
    duplicates = [c for c in ticket.charges if c.duplicate_of]
    if not duplicates:
        result: Eligibility | Unknown = Unknown("no_duplicate_charge", "no charge on this ticket is marked as a duplicate")
        if audit is not None:
            audit.record_eligibility(ticket, result)
        return result

    charge = max(duplicates, key=lambda c: c.amount_gbp)
    reasons: list[str] = []
    cited: list[str] = []
    allowed = True

    if charge.status == "pending":
        allowed, _ = False, cited.append("R2")
        reasons.append(f"{clauses['R2'].text} ({charge.id} is pending)")
    if charge.status == "reversed":
        allowed, _ = False, cited.append("R3")
        reasons.append(f"{clauses['R3'].text} ({charge.id} was already reversed)")
    age_days = (now - charge.at).days
    if (window := clauses["R4"].window_days) is not None and age_days > window:
        allowed, _ = False, cited.append("R4")
        reasons.append(f"{clauses['R4'].text} ({charge.id} is {age_days} days old)")
    if (cap := clauses["R5"].max_gbp) is not None and charge.amount_gbp > cap:
        allowed, _ = False, cited.append("R5")
        reasons.append(f"{clauses['R5'].text} (£{charge.amount_gbp:.2f} > £{cap:.0f})")
    if allowed:
        cited.append("R1")
        reasons.append(f"{clauses['R1'].text} ({charge.id} duplicates {charge.duplicate_of}, £{charge.amount_gbp:.2f})")

    result = Eligibility(allowed, charge.id, charge.amount_gbp, tuple(cited), tuple(reasons))
    if audit is not None:
        audit.record_eligibility(ticket, result)
    return result


@dataclass(frozen=True)
class Routed:
    """Which tier answered, and what the call cost. Read from the trace, not asserted."""

    answered_by: str | None
    attempts: tuple[tuple[str, str, str], ...]  # (implementation, outcome, reason)
    usd: float | None
    unmetered: bool
    ms: float


def route(ticket: Ticket) -> tuple[Intent | Unknown, Routed]:
    """Ask the intent question and report which tier satisfied it.

    'Routing' here is not a separate model deciding where to send the request — it is the
    cascade the runtime already performs, made visible. The cheap tier answers what it can;
    only what it abstains on reaches the next one.
    """
    answer, _score, span = classify_with_confidence(ticket.text, Intent)
    if span is None:
        return answer, Routed(None, (), None, True, 0.0)
    costs = [a.usd for a in span.attempts if a.usd is not None]
    return answer, Routed(
        span.answered_by,
        tuple((a.implementation, a.outcome, a.reason) for a in span.attempts),
        sum(costs) if costs else None,
        any(a.usd is None and a.outcome not in ("skipped", "cache_hit") for a in span.attempts),
        span.total_ms,
    )


def rerank(query: str, passages: Sequence[Passage], *, limit: int | None = None) -> list[tuple[Passage, Score]] | Unknown:
    """Order candidate passages for a query. Scores are relevance, comparable only here.

    Note the type discipline: ``rank`` returns ``Score(kind="relevance")``, and the money
    gate refuses to threshold that. A reranker's score is not P(correct), and the gate says
    so rather than letting a backend's number become an approval.
    """
    return tc.rank(query, list(passages), limit=limit)


def supports(claim: str, passage: Passage) -> Verdict:
    """Does this passage actually back the claim? Three-valued, so 'unclear' is sayable."""
    return tc.check(("supports", claim), evidence=[passage])


def classify_with_confidence(text: str, labels: type) -> tuple[object, Score | None, object]:
    """``classify``, keeping the confidence the implementation computed.

    GAP IN THE LIBRARY, and the one that matters most for this use case: ``tc.classify``
    returns ``T | Unknown`` and drops the ``Output.score`` beside it. The learned tier
    computes a temperature-scaled, threshold-calibrated probability (see
    ``tensorcode.backends.linear``) and nothing can read it — ``Output.score`` is written by
    implementations and never consumed anywhere in the tree, and the span does not record it
    either. A decision layer cannot gate on a confidence it cannot see, so this example goes
    through ``Runtime.call`` directly to keep it.

    The facade should return the score (or record it on the span); see
    docs/revival/20-decision-layer-examples.md.
    """
    rt = current()
    before = len(rt.trace.spans)
    validate = (lambda v: isinstance(v, labels)) if isinstance(labels, type) else (lambda v: v in labels)
    out = rt.call(tc.Request("classify", text, labels, {}), validate=validate)
    span = next((s for s in reversed(rt.trace.spans[before:]) if s.op == "classify"), None)
    if span is not None and out.score is not None:
        span.labels["confidence"] = f"{out.score.value:.4f} ({out.score.kind})"
    return out.value, out.score, span


def handle(ticket: Ticket, *, audit: Audit | None = None, table: dict[str, Gate] | None = None) -> dict:
    """One request handler, start to finish — the shape a web backend would have.

    Control flow is ordinary Python: triage, then (only if money is at stake) an eligibility
    walk, then a gate decides whether this can be auto-applied. Nothing recurses, nothing
    plans, and the whole thing is one pass.
    """
    table = table or gates()
    t = triage(ticket, gate=table["routing"], audit=audit)
    out: dict = {
        "ticket": ticket.id,
        "intent": None if isinstance(t.intent, Unknown) else t.intent.name,
        "department": None if isinstance(t.department, Unknown) else t.department.value,
        "urgency": None if isinstance(t.urgency, Unknown) else t.urgency.value,
        "refund_asked": t.refund_asked.status,
        "routing": {"action": t.routing.action, "confidence": t.routing.confidence, "why": t.routing.why},
        "tier": t.tier,
        "ms": round(t.ms, 2),
    }
    if t.refund_asked.status != "holds":
        out["next"] = "no refund path" if t.refund_asked.status == "fails" else "needs a human: intent unclear"
        return out

    eligibility = refund_eligibility(ticket, audit=audit)
    if isinstance(eligibility, Unknown):
        out["refund"] = {"decision": "escalate", "why": f"{eligibility.reason}: {eligibility.detail}"}
        return out

    money = table["money"].decide(eligibility.allowed, t.confidence)
    out["refund"] = {
        "decision": "pay" if (eligibility.allowed and money.action == "auto") else ("confirm" if eligibility.allowed else "refuse"),
        "allowed": eligibility.allowed,
        "charge": eligibility.charge_id,
        "amount_gbp": eligibility.amount_gbp,
        "clauses": list(eligibility.clauses),
        "reasons": list(eligibility.reasons),
        "gate": {"action": money.action, "confidence": money.confidence, "why": money.why},
    }
    return out
