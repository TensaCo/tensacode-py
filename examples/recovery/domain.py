"""Types, rules, and policy for bounded recovery. Supporting code."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import tensacode as tc
from tensacode.actions import spec_of
from tensacode.backends.builtin import IN_PROCESS


@tc.action(effect="external", idempotent=False)
@dataclass(frozen=True)
class CreditAccount:
    account: str
    cents: int
    reason: str

    def achieved(self, observed: tuple[CreditAccount, ...]) -> bool:
        return self in observed


class Outcome(enum.Enum):
    transient = "transient"  # known not applied; the executor says trying again is fine
    terminal = "terminal"  # refused or failed in a way repetition will not fix
    effect_unknown = "effect_unknown"  # may have happened; no observation settles it
    effect_missing = "effect_missing"  # reported or possibly applied, but not observed (lag or silent failure)


@dataclass(frozen=True)
class AttemptReport:
    receipt: tc.Receipt
    verdict: tc.Verdict


@dataclass(frozen=True)
class Limits:
    max_invocations: int = 3
    max_observations: int = 2
    deadline_s: float = 30.0
    executor_honors_keys: bool = False  # facts about the target system, not hopes:
    observation_is_authoritative: bool = False  # a read that cannot lag behind a committed effect


@dataclass(frozen=True)
class Retry:
    after_s: float


@dataclass(frozen=True)
class Reobserve:
    after_s: float


@dataclass(frozen=True)
class Escalate:
    reason: str


@dataclass
class Episode:
    action: CreditAccount
    limits: Limits
    outcome: Outcome | tc.Unknown | None = None
    invocations: int = 0
    observations: int = 0
    elapsed_s: float = 0.0
    receipts: list[tc.Receipt] = field(default_factory=list)


@dataclass(frozen=True)
class Resolution:
    status: str  # "done" | "escalated"
    reason: str
    invocations: int
    observations: int
    elapsed_s: float


# -- classify: what is true about the failure (rules over receipt + verification)


@tc.implementation(
    "classify",
    name="receipt-rules",
    version="1",
    accepts=lambda r: isinstance(r.subject, AttemptReport) and r.target is Outcome,
    profile=IN_PROCESS,
)
def classify_attempt(request: tc.Request) -> Outcome | tc.Unknown:
    receipt, verdict = request.subject.receipt, request.subject.verdict
    if receipt.status == "rejected":
        return Outcome.terminal
    if receipt.status == "failed":
        return Outcome.transient if receipt.retryable else Outcome.terminal
    if verdict.status == "unknown":
        return Outcome.effect_unknown
    if verdict.status == "fails":
        return Outcome.effect_missing
    return tc.Unknown("verified_success_is_not_a_failure")


# -- choose: which bounded step, under hard constraints


def next_steps(ep: Episode) -> list[object]:
    backoff = min(2.0 ** ep.invocations, 8.0)
    return [Retry(backoff), Reobserve(1.0), Escalate(f"{ep.outcome.value if isinstance(ep.outcome, Outcome) else 'unclassified'} after {ep.invocations} invocation(s)")]


def retry_is_safe(step: object, ep: Episode) -> bool:
    if not isinstance(step, Retry):
        return True
    if ep.outcome is Outcome.transient:
        return True
    if ep.outcome is Outcome.effect_missing and ep.limits.observation_is_authoritative:
        return True  # an authoritative read shows no effect: repeating cannot duplicate it
    if ep.outcome in (Outcome.effect_unknown, Outcome.effect_missing):
        idempotent = spec_of(ep.action).idempotent  # type: ignore[union-attr]
        return idempotent or ep.limits.executor_honors_keys
    return False


def reobserve_is_useful(step: object, ep: Episode) -> bool:
    if not isinstance(step, Reobserve):
        return True
    if ep.outcome is Outcome.effect_missing and ep.limits.observation_is_authoritative:
        return False  # waiting cannot change an authoritative answer
    return ep.outcome in (Outcome.effect_unknown, Outcome.effect_missing)


def within_limits(step: object, ep: Episode) -> bool:
    if isinstance(step, Retry):
        return ep.invocations < ep.limits.max_invocations and ep.elapsed_s + step.after_s <= ep.limits.deadline_s
    if isinstance(step, Reobserve):
        return ep.observations < ep.limits.max_observations and ep.elapsed_s + step.after_s <= ep.limits.deadline_s
    return True


RECOVERY_POLICY = (
    tc.Constraint("retry_is_safe", retry_is_safe),
    tc.Constraint("reobserve_is_useful", reobserve_is_useful),
    tc.Constraint("within_limits", within_limits),
)


def _utility(step: object, ep: Episode) -> float:
    if isinstance(step, Reobserve):
        return 3.0  # when the effect is uncertain, look before acting again
    if isinstance(step, Retry):
        return 2.0
    return 1.0


RECOVER_SAFELY = tc.Objective("recover_safely", "Observe before repeating an uncertain effect; retry only when safe; otherwise escalate.", _utility)

