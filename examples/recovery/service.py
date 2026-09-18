"""A scripted, flaky payments ledger that records ground truth. Supporting code for demos and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import tensacode as tc

from .domain import CreditAccount

# Per-invocation behaviors:
#   ok                    applied, reply delivered
#   unavailable           503 before any effect (retryable)
#   forbidden             403, no effect (terminal)
#   timeout_after_commit  effect applied, reply lost
#   timeout_before_commit no effect, reply lost (indistinguishable from the above to the caller)


@dataclass
class Ledger:
    script: list[str]  # behavior for invocation 1, 2, ...; the last entry repeats
    honors_keys: bool
    observe_script: list[bool] = field(default_factory=lambda: [True])  # observation endpoint reachable, per call
    lag_reads: int = 0  # reads after a commit that do not yet reflect it (replica lag)
    applied: list[CreditAccount] = field(default_factory=list)  # ground truth
    _keys: dict[str, tc.Receipt] = field(default_factory=dict)
    _stale_reads_left: int = 0
    invocations: int = 0
    observations: int = 0

    def execute(self, act: CreditAccount, *, key: str | None) -> tc.Receipt:
        behavior = self.script[min(self.invocations, len(self.script) - 1)]
        self.invocations += 1
        if self.honors_keys and key in self._keys:
            return self._keys[key]
        if behavior == "unavailable":
            return tc.Receipt(act, "failed", retryable=True, idempotency_key=key, error="503", retry_after_s=1.0)
        if behavior == "forbidden":
            return tc.Receipt(act, "rejected", idempotency_key=key, error="403 account closed")
        if behavior == "timeout_before_commit":
            raise TimeoutError("no reply")
        self.applied.append(act)
        self._stale_reads_left = self.lag_reads
        receipt = tc.Receipt(act, "applied", idempotency_key=key, effect_id=f"txn-{len(self.applied)}")
        if self.honors_keys and key:
            self._keys[key] = receipt
        if behavior == "timeout_after_commit":
            raise TimeoutError("no reply")
        return receipt

    def observe(self) -> tuple[CreditAccount, ...] | tc.Unknown:
        reachable = self.observe_script[min(self.observations, len(self.observe_script) - 1)]
        self.observations += 1
        if not reachable:
            return tc.Unknown("ledger_query_unavailable")
        if self._stale_reads_left > 0:
            self._stale_reads_left -= 1
            return tuple(self.applied[:-1])
        return tuple(self.applied)
