"""Typed questions for a back-office decision service.

This is the shape ordinary software wants: code owns control flow, and asks bounded,
typed questions whose answers it can branch on. Nothing here generates text.

Each question declares its answer *type*, so the runtime can validate what came back:

    department   -> Department          (a choice among five)
    urgency      -> Urgency             (an ordered level, so it can be compared)
    refund asked -> Verdict             (holds / fails / unknown, never a bare bool)
    eligibility  -> Eligibility         (a decision plus the clauses it rests on)

The labels come from Banking77 (Casanueva et al., 2020; CC-BY-4.0) via
``examples.support_router.domain``, so intent accuracy is measurable against a public
test split rather than against our own opinion.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import tensacode as tc

from ..support_router.domain import Intent


class Department(enum.Enum):
    """Where a ticket is routed. Five queues a real desk would have."""

    billing = "billing"
    cards = "cards"
    transfers = "transfers"
    identity = "identity"
    general = "general"


class Urgency(enum.Enum):
    """An ordered level. ``compare`` exists because a score you cannot order is not a level."""

    low = "low"
    normal = "normal"
    high = "high"
    critical = "critical"

    @property
    def rank(self) -> int:
        return ["low", "normal", "high", "critical"].index(self.value)


#: Which department owns each Banking77 intent.
#:
#: PROVENANCE: this mapping is ours, not part of the dataset. Department accuracy is
#: therefore "public text + public intent label + our mapping"; intent accuracy alone is
#: measured against the public label with nothing of ours in the path. Both are reported
#: separately in eval/results/decisions_*.json for exactly this reason.
DEPARTMENT_OF_INTENT: dict[str, Department] = {}


def _own(department: Department, *prefixes: str) -> None:
    for intent in Intent:
        if any(intent.name.startswith(p) or p in intent.name for p in prefixes):
            DEPARTMENT_OF_INTENT.setdefault(intent.name, department)


_own(Department.identity, "verify", "why_verify", "unable_to_verify", "edit_personal_details", "age_limit")
_own(Department.transfers, "transfer", "receiving_money", "beneficiary", "balance_not_updated", "cancel_transfer", "failed_transfer")
_own(Department.billing, "refund", "charge", "fee", "exchange", "extra_charge", "wrong_amount", "transaction_charged_twice", "top_up", "topping_up", "automatic_top_up", "pending_top_up")
_own(Department.cards, "card", "pin", "passcode", "contactless", "atm", "cash_withdrawal", "declined", "lost_or_stolen", "compromised", "disposable", "virtual", "supported_cards", "visa_or_mastercard")
for _intent in Intent:  # everything unclaimed
    DEPARTMENT_OF_INTENT.setdefault(_intent.name, Department.general)

#: Intents that mean money is disputed or requested back. Ours, same caveat as above.
REFUND_INTENTS = frozenset(
    {
        "refund_not_showing_up",
        "request_refund",
        "transaction_charged_twice",
        "card_payment_not_recognised",
        "cash_withdrawal_not_recognised",
        "direct_debit_payment_not_recognised",
        "extra_charge_on_statement",
        "wrong_amount_of_cash_received",
        "reverted_card_payment",
    }
)

#: Intents where a customer is locked out or exposed, so waiting costs them something.
URGENT_INTENTS = frozenset({"compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone", "card_swallowed", "pin_blocked", "unable_to_verify_identity"})
HIGH_INTENTS = frozenset({"declined_card_payment", "declined_cash_withdrawal", "declined_transfer", "card_not_working", "top_up_failed", "failed_transfer", "transfer_not_received_by_recipient"})


def urgency_of(intent: Intent) -> Urgency:
    if intent.name in URGENT_INTENTS:
        return Urgency.critical
    if intent.name in HIGH_INTENTS:
        return Urgency.high
    if intent.name in REFUND_INTENTS:
        return Urgency.normal
    return Urgency.low


# ------------------------------------------------------------------ records


@dataclass(frozen=True)
class Charge:
    """A line on the customer's statement."""

    id: str
    merchant: str
    amount_gbp: float
    at: datetime
    status: Literal["settled", "pending", "reversed"] = "settled"
    duplicate_of: str | None = None


@dataclass(frozen=True)
class Ticket:
    """What arrives from the customer. ``text`` is real Banking77 text in the eval."""

    id: str
    customer: str
    text: str
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    charges: tuple[Charge, ...] = ()


@dataclass(frozen=True)
class PolicyClause:
    """One rule an auditor can point at. ``id`` is what a decision cites."""

    id: str
    text: str
    max_gbp: float | None = None
    window_days: int | None = None


#: A refund policy as data. Clause ids appear in every eligibility decision's reasons,
#: so "why did you refund this?" is answerable by pointing at the clause and the charge.
REFUND_POLICY: tuple[PolicyClause, ...] = (
    PolicyClause("R1", "A charge billed twice may be refunded automatically up to £50.", max_gbp=50.0, window_days=90),
    PolicyClause("R2", "A pending charge is not refundable until it settles.", None, None),
    PolicyClause("R3", "A charge already reversed is not refundable again.", None, None),
    PolicyClause("R4", "A charge older than 90 days needs a specialist.", None, window_days=90),
    PolicyClause("R5", "A refund above £50 needs a human approval.", max_gbp=50.0),
)


@dataclass(frozen=True)
class Eligibility:
    """The answer to 'may we refund this?'. Carries the clauses it rests on."""

    allowed: bool
    charge_id: str | None
    amount_gbp: float | None
    clauses: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Passage:
    """A candidate document for reranking; ``id`` is what a citation refers to."""

    id: str
    title: str
    text: str
