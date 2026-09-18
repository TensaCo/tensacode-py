"""The recovery agent. Runs against ``tensacode`` (the prototype).

A single effect with bounded recovery. Nothing here makes the effect exactly-once:
retries are chosen only when the receipt, a fresh observation, or the executor's
idempotency guarantee makes them safe; otherwise the agent escalates.
"""

from __future__ import annotations

from typing import Callable

import tensacode as tc

from .domain import RECOVER_SAFELY, RECOVERY_POLICY, AttemptReport, Episode, Escalate, Outcome, Resolution, Retry, next_steps


def credit_with_recovery(ep: Episode, *, executor: tc.actions.Executor, observe: Callable[[], object], key: str, sleep: Callable[[float], None]) -> Resolution:
    receipt = tc.invoke(ep.action, executor=executor, key=key)
    ep.invocations += 1
    for _ in range(ep.limits.max_invocations + ep.limits.max_observations):  # hard stop even if a constraint is wrong
        verdict = tc.verify(receipt, observe=observe, expect=ep.action.achieved)
        if verdict.holds:
            return Resolution("done", "; ".join(verdict.reasons), ep.invocations, ep.observations, ep.elapsed_s)
        ep.outcome = tc.classify(AttemptReport(receipt, verdict), Outcome)
        step = tc.choose(next_steps(ep), objective=RECOVER_SAFELY, given=ep, constraints=RECOVERY_POLICY)
        if isinstance(step, (Escalate, tc.Unknown)):
            return Resolution("escalated", step.reason, ep.invocations, ep.observations, ep.elapsed_s)
        sleep(step.after_s)
        ep.elapsed_s += step.after_s
        if isinstance(step, Retry):
            receipt = tc.invoke(ep.action, executor=executor, key=key)
            ep.invocations += 1
        else:
            ep.observations += 1
    return Resolution("escalated", "step limit reached", ep.invocations, ep.observations, ep.elapsed_s)
