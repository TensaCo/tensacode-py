"""Complete recovery trajectories against a simulated flaky ledger, with ground truth.

    python eval/recovery_trajectories.py [--episodes 5000]

This is a simulation. Fault rates are chosen to exercise every branch, not to
model any real payments system; conclusions are about decision logic under the
stated fault model, not about production reliability.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import tensacode as tc  # noqa: E402
from examples.recovery.agent import credit_with_recovery  # noqa: E402
from examples.recovery.domain import AUTHORITY, CreditAccount, Episode, Limits, classify_attempt  # noqa: E402
from examples.recovery.service import Ledger  # noqa: E402
from tensacode.backends.builtin import UtilityChooser  # noqa: E402

BEHAVIORS = {"ok": 0.60, "unavailable": 0.15, "timeout_after_commit": 0.08, "timeout_before_commit": 0.07, "forbidden": 0.05, "ok_after_lag": 0.05}


@dataclass(frozen=True)
class World:
    script: list[str]
    observe_script: list[bool]
    honors_keys: bool
    authoritative: bool
    lag_reads: int


def sample_world(rng: random.Random) -> World:
    names, weights = zip(*BEHAVIORS.items())
    script = [rng.choices(names, weights)[0] for _ in range(4)]
    authoritative = rng.random() < 0.5
    lag = 0 if authoritative else (1 if "ok_after_lag" in script or rng.random() < 0.3 else 0)
    script = [("ok" if b == "ok_after_lag" else b) for b in script]
    return World(script, [rng.random() < 0.85 for _ in range(6)], rng.random() < 0.5, authoritative, lag)


def ledger(w: World) -> Ledger:
    return Ledger(list(w.script), honors_keys=w.honors_keys, observe_script=list(w.observe_script), lag_reads=w.lag_reads)


ACTION = CreditAccount("acct-42", 1250, "refund order 7781")
KEY = "refund-7781"


def policy_tensacode(w: World, rt: tc.Runtime) -> tuple[str, int, float]:
    led = ledger(w)
    ep = Episode(ACTION, Limits(max_invocations=3, max_observations=2, deadline_s=30.0, executor_honors_keys=w.honors_keys, observation_is_authoritative=w.authoritative))
    with tc.use(rt):
        r = credit_with_recovery(ep, executor=led, observe=led.observe, key=KEY, sleep=lambda s: None)
    return r.status, len(led.applied), r.elapsed_s


def policy_tensacode_misinformed(w: World, rt: tc.Runtime) -> tuple[str, int, float]:
    """Sensitivity: the agent is wrongly told every executor honors idempotency keys."""
    led = ledger(w)
    ep = Episode(ACTION, Limits(max_invocations=3, max_observations=2, deadline_s=30.0, executor_honors_keys=True, observation_is_authoritative=w.authoritative))
    with tc.use(rt):
        r = credit_with_recovery(ep, executor=led, observe=led.observe, key=KEY, sleep=lambda s: None)
    return r.status, len(led.applied), r.elapsed_s


def policy_naive_retry(w: World, rt: tc.Runtime) -> tuple[str, int, float]:
    """Retry any non-applied outcome (including timeouts) up to 3 times; trust the receipt."""
    led, elapsed = ledger(w), 0.0
    with tc.use(rt):
        for attempt in range(3):
            receipt = tc.invoke(ACTION, executor=led, key=KEY, authorization=AUTHORITY)
            if receipt.status == "applied":
                return "done", len(led.applied), elapsed
            elapsed += 2.0**attempt
    return "escalated", len(led.applied), elapsed


def policy_verify_then_retry(w: World, rt: tc.Runtime) -> tuple[str, int, float]:
    """Verify by observation; if not verified, retry (no safety analysis), up to 3 invocations."""
    led, elapsed = ledger(w), 0.0
    with tc.use(rt):
        for attempt in range(3):
            receipt = tc.invoke(ACTION, executor=led, key=KEY, authorization=AUTHORITY)
            if receipt.status == "rejected":
                return "escalated", len(led.applied), elapsed
            if tc.verify(receipt, observe=led.observe, expect=ACTION.achieved).holds:
                return "done", len(led.applied), elapsed
            elapsed += 2.0**attempt
    return "escalated", len(led.applied), elapsed


def policy_single_attempt(w: World, rt: tc.Runtime) -> tuple[str, int, float]:
    led = ledger(w)
    with tc.use(rt):
        receipt = tc.invoke(ACTION, executor=led, key=KEY, authorization=AUTHORITY)
    return ("done" if receipt.status == "applied" else "escalated"), len(led.applied), 0.0


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "eval/results/recovery.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    worlds = [sample_world(rng) for _ in range(args.episodes)]
    results = {"episodes": args.episodes, "seed": args.seed, "behavior_weights": BEHAVIORS, "policies": {}}
    for name, policy in [("tensacode_recovery", policy_tensacode), ("tensacode_recovery_misinformed_keys", policy_tensacode_misinformed), ("naive_retry", policy_naive_retry), ("verify_then_retry", policy_verify_then_retry), ("single_attempt", policy_single_attempt)]:
        outcomes: Counter[str] = Counter()
        cpu, sim = 0.0, 0.0
        for w in worlds:
            rt = tc.Runtime([classify_attempt, UtilityChooser()])
            t0 = time.perf_counter()
            status, effects, elapsed = policy(w, rt)
            cpu += time.perf_counter() - t0
            sim += elapsed
            if effects >= 2:
                outcomes["duplicate_effect"] += 1
            elif status == "done" and effects == 1:
                outcomes["done_exactly_once"] += 1
            elif status == "done" and effects == 0:
                outcomes["false_done"] += 1
            elif status == "escalated" and effects == 1:
                outcomes["escalated_effect_happened"] += 1
            else:
                outcomes["escalated_no_effect"] += 1
        n = len(worlds)
        results["policies"][name] = {
            "rates": {k: round(v / n, 4) for k, v in sorted(outcomes.items())},
            "ci95": {k: wilson(v, n) for k, v in sorted(outcomes.items())},
            "mean_simulated_seconds": round(sim / n, 3),
            "mean_cpu_ms_per_episode": round(1e3 * cpu / n, 4),
        }
        print(name, json.dumps(results["policies"][name]["rates"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
