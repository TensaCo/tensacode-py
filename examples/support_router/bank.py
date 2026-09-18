"""A simulated card service with an idempotency ledger and injectable faults. Supporting code."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import tensacode as tc

from .domain import Account, Card, FreezeCard, SendArticle


@dataclass
class Bank:
    accounts: dict[str, Account]
    faults: dict[str, str] = field(default_factory=dict)  # idempotency key -> fault
    honors_idempotency: bool = True
    outbox: list[tuple[str, str]] = field(default_factory=list)
    effects: list[tuple[str, object]] = field(default_factory=list)  # every effect that really happened
    _ledger: dict[str, tc.Receipt] = field(default_factory=dict)
    reachable: bool = True

    def find_account(self, email: str) -> Account | None:
        return next((a for a in self.accounts.values() if a.email == email), None)

    def execute(self, action: object, *, key: str | None) -> tc.Receipt:
        if self.honors_idempotency and key in self._ledger:
            return self._ledger[key]
        fault = self.faults.pop(key, None) if key else None
        if fault == "unavailable":  # rejected at the front door: nothing happened
            return tc.Receipt(action, "failed", retryable=True, idempotency_key=key, error="503 service unavailable", retry_after_s=2.0)
        if fault == "forbidden":
            return tc.Receipt(action, "rejected", retryable=False, idempotency_key=key, error="403 card belongs to another customer")
        self._apply(action)
        if fault == "timeout_after_commit":  # the effect happened; the reply was lost
            raise TimeoutError("no response within 5s")
        receipt = tc.Receipt(action, "applied", idempotency_key=key, effect_id=f"eff-{len(self.effects)}")
        if key:
            self._ledger[key] = receipt
        return receipt

    def _apply(self, action: object) -> None:
        self.effects.append((type(action).__name__, action))
        if isinstance(action, FreezeCard):
            for account in self.accounts.values():
                cards = tuple(dataclasses.replace(c, status="frozen") if c.id == action.card_id else c for c in account.cards)
                self.accounts[account.customer_id] = dataclasses.replace(account, cards=cards)
        elif isinstance(action, SendArticle):
            self.outbox.append((action.to, action.article))
        else:
            raise ValueError(f"unsupported action {action!r}")

    def observe(self, action: object) -> Card | tuple[tuple[str, str], ...] | tc.Unknown:
        if not self.reachable:
            return tc.Unknown("observation_endpoint_unreachable")
        if isinstance(action, FreezeCard):
            cards = [c for a in self.accounts.values() for c in a.cards if c.id == action.card_id]
            return cards[0] if cards else tc.Unknown("card_not_found")
        return tuple(self.outbox)
