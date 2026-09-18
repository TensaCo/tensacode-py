"""Domain types and policy for a card-support router. Supporting code, not the program."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import tensacode as tc

# The 77 intent labels of Banking77 (Casanueva et al., 2020; CC-BY-4.0), verbatim.
LABELS = """Refund_not_showing_up activate_my_card age_limit apple_pay_or_google_pay atm_support automatic_top_up
balance_not_updated_after_bank_transfer balance_not_updated_after_cheque_or_cash_deposit beneficiary_not_allowed
cancel_transfer card_about_to_expire card_acceptance card_arrival card_delivery_estimate card_linking card_not_working
card_payment_fee_charged card_payment_not_recognised card_payment_wrong_exchange_rate card_swallowed
cash_withdrawal_charge cash_withdrawal_not_recognised change_pin compromised_card contactless_not_working
country_support declined_card_payment declined_cash_withdrawal declined_transfer direct_debit_payment_not_recognised
disposable_card_limits edit_personal_details exchange_charge exchange_rate exchange_via_app extra_charge_on_statement
failed_transfer fiat_currency_support get_disposable_virtual_card get_physical_card getting_spare_card
getting_virtual_card lost_or_stolen_card lost_or_stolen_phone order_physical_card passcode_forgotten
pending_card_payment pending_cash_withdrawal pending_top_up pending_transfer pin_blocked receiving_money
request_refund reverted_card_payment? supported_cards_and_currencies terminate_account top_up_by_bank_transfer_charge
top_up_by_card_charge top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card
transaction_charged_twice transfer_fee_charged transfer_into_account transfer_not_received_by_recipient
transfer_timing unable_to_verify_identity verify_my_identity verify_source_of_funds verify_top_up
virtual_card_not_working visa_or_mastercard why_verify_identity wrong_amount_of_cash_received
wrong_exchange_rate_for_cash_withdrawal""".split()

Intent = enum.Enum("Intent", [(label.rstrip("?").lower(), label) for label in LABELS])

# -- observations and state


@dataclass(frozen=True)
class InboundEmail:
    id: str
    received_at: datetime
    sender: str
    subject: str
    body: str


@dataclass(frozen=True)
class SupportRequest:
    message_id: str
    sender: str
    text: str
    card_last4: str | None


@dataclass(frozen=True)
class Card:
    id: str
    last4: str
    status: Literal["active", "frozen", "cancelled"]


@dataclass(frozen=True)
class Account:
    customer_id: str
    email: str
    cards: tuple[Card, ...]


@dataclass(frozen=True)
class Situation:
    intent: enum.Enum
    request: SupportRequest
    account: Account | None
    recent_actions: tuple[object, ...]


# -- actions


@tc.action(effect="external", idempotent=True, reversible=True)
@dataclass(frozen=True)
class FreezeCard:
    card_id: str

    def achieved(self, card: Card) -> bool:
        return card.status == "frozen"


@tc.action(effect="external", idempotent=True)
@dataclass(frozen=True)
class SendArticle:
    to: str
    article: str

    def achieved(self, outbox: tuple[tuple[str, str], ...]) -> bool:
        return (self.to, self.article) in outbox


ARTICLES = {
    Intent.card_arrival: "kb/where-is-my-card",
    Intent.card_delivery_estimate: "kb/where-is-my-card",
    Intent.pin_blocked: "kb/unblock-pin",
    Intent.lost_or_stolen_card: "kb/lost-card",
    Intent.compromised_card: "kb/suspicious-activity",
    Intent.card_payment_not_recognised: "kb/suspicious-activity",
    Intent.terminate_account: "kb/close-account",
    Intent.apple_pay_or_google_pay: "kb/mobile-wallets",
    Intent.card_acceptance: "kb/where-cards-work",
    Intent.visa_or_mastercard: "kb/card-network",
    Intent.declined_card_payment: "kb/declined-payments",
}
FRAUD_RISK = {Intent.lost_or_stolen_card, Intent.compromised_card, Intent.card_payment_not_recognised}


def options(intent: enum.Enum, request: SupportRequest, account: Account | None) -> list[object]:
    """Enumerate candidate actions. Feasibility is decided by constraints, not here."""
    found: list[object] = []
    if intent in FRAUD_RISK and account:
        cards = [c for c in account.cards if request.card_last4 in (None, c.last4)]
        found += [FreezeCard(c.id) for c in cards]
    if intent in ARTICLES:
        found.append(SendArticle(request.sender, ARTICLES[intent]))
    return found


# -- policy: hard constraints and objective


def _card(situation: Situation, card_id: str) -> Card | tc.Unknown:
    if situation.account is None:
        return tc.Unknown("account_not_found")
    return next((c for c in situation.account.cards if c.id == card_id), tc.Unknown("card_not_on_account"))


def card_is_active(option: object, situation: Situation) -> bool | tc.Unknown:
    if not isinstance(option, FreezeCard):
        return True
    card = _card(situation, option.card_id)
    return card if isinstance(card, tc.Unknown) else card.status == "active"


def not_repeated(option: object, situation: Situation) -> bool:
    return option not in situation.recent_actions


SUPPORT_POLICY = (tc.Constraint("card_is_active", card_is_active), tc.Constraint("not_repeated_in_24h", not_repeated))



def _utility(option: object, situation: Situation) -> float:
    if isinstance(option, FreezeCard):
        return 10.0 if situation.intent in FRAUD_RISK else -10.0
    return 1.0


RESOLVE_SAFELY = tc.Objective(
    "resolve_safely",
    "Stop possible fraud first; otherwise send the most specific help article.",
    utility=_utility,
)
