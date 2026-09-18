"""Scripted recovery scenarios, including the ones where the right answer is to stop.

    python -m examples.recovery.demo
"""

from __future__ import annotations

from dataclasses import replace

import tensacode as tc
from tensacode.backends.builtin import UtilityChooser

from .agent import credit_with_recovery
from .domain import CreditAccount, Episode, Limits, classify_attempt
from .service import Ledger

LIMITS = Limits(max_invocations=3, max_observations=2, deadline_s=30.0)
SCENARIOS = [
    ("transient 503, then success", Ledger(["unavailable", "ok"], honors_keys=False), LIMITS),
    ("terminal 403", Ledger(["forbidden"], honors_keys=False), LIMITS),
    ("reply lost after commit; ledger query works", Ledger(["timeout_after_commit"], honors_keys=False), LIMITS),
    ("reply lost before commit; replica read (may lag)", Ledger(["timeout_before_commit", "ok"], honors_keys=False), LIMITS),
    ("reply lost before commit; authoritative read", Ledger(["timeout_before_commit", "ok"], honors_keys=False), replace(LIMITS, observation_is_authoritative=True)),
    ("reply lost; query down; keys NOT honored", Ledger(["timeout_after_commit", "ok"], honors_keys=False, observe_script=[False]), LIMITS),
    ("reply lost; query down, then up; keys honored", Ledger(["timeout_after_commit", "ok"], honors_keys=True, observe_script=[False, False, True]), replace(LIMITS, executor_honors_keys=True)),
    ("persistent 503", Ledger(["unavailable"], honors_keys=False), LIMITS),
]


def main() -> None:
    runtime = tc.Runtime([classify_attempt, UtilityChooser()])
    with tc.use(runtime):
        for title, ledger, limits in SCENARIOS:
            mark = len(runtime.trace.spans)
            ep = Episode(CreditAccount("acct-42", 1250, "refund order 7781"), limits)
            with runtime.trace.section("episode", scenario=title):
                result = credit_with_recovery(ep, executor=ledger, observe=ledger.observe, key="refund-7781", sleep=lambda s: None)
            print(f"### {title}")
            print(f"result: {result.status} ({result.reason}); invocations={result.invocations} observations={result.observations} simulated_s={result.elapsed_s}")
            print(f"ground truth: effects applied = {len(ledger.applied)}")
            print(runtime.trace.render(since=mark))
            print()


if __name__ == "__main__":
    main()
