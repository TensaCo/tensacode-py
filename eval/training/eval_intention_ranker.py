"""Put the learned ranker in charge and see whether the tasks still finish.

    PYTHONPATH=src:. python eval/training/eval_intention_ranker.py --model DIR --episodes 20 --first-seed 9001

Imitation accuracy on held-out decisions says whether the ranker agrees with the
hand-written objective. It does not say whether an agent driven by it works: one wrong
choice early can lose an episode that the rest of the run would have scored. So this runs
whole episodes with the learned chooser in the runtime instead of ``UtilityChooser``, on
seeds the ranker was not trained on, and scores them the way the bench does — by asking
the app.

Both arms run the same seeds, the same agents and the same perception. Only the ``choose``
implementation differs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

import tensorcode as tc  # noqa: E402
from examples.browser_agents import harness  # noqa: E402
from examples.browser_agents.browser import Browser, LabelMatcher, classify_feedback  # noqa: E402
from examples.browser_agents.learned.chooser import LearnedChooser  # noqa: E402
from examples.browser_agents.learned.text_embed import LabelEncoder  # noqa: E402
from examples.browser_agents.mind import run_mind  # noqa: E402
from tensorcode.backends.builtin import UtilityChooser  # noqa: E402


def run_arm(browser, base: str, task, seeds: range, chooser) -> dict:
    context = browser.new_context(viewport={"width": task.viewport[0], "height": task.viewport[1]})
    harness.block_outside_network(context, base)
    page = context.new_page()
    rows = []
    for seed in seeds:
        runtime = tc.Runtime(
            [LabelMatcher(), chooser, classify_feedback, *task.bindings()],
            policy=tc.Policy(localities=frozenset({"in_process"}), allow_egress=False, available=frozenset({"sklearn"}), cache=False),
        )
        t0 = time.perf_counter()
        with tc.use(runtime):
            if task.setup:
                task.setup(seed)
            page.goto(task.url(base, seed))
            ui = Browser(page, episode=f"{task.name}-{seed}")
            try:
                outcome = run_mind(ui, task.spec)
                error = None
            except Exception as exc:  # noqa: BLE001
                outcome, error = None, f"{type(exc).__name__}: {exc}"
        score = task.score(page, seed) if task.score else (page.evaluate("() => window.__score ? window.__score() : null") or {})
        rows.append({
            "seed": seed, "items": score.get("items", 0), "correct": score.get("correct", 0),
            "duplicates": score.get("duplicates", 0), "actions": ui.stats.actions,
            "status": getattr(outcome, "status", "error"), "reason": getattr(outcome, "reason", error),
            "seconds": round(time.perf_counter() - t0, 3),
        })
    context.close()
    items = sum(r["items"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    return {
        "episodes": len(rows), "items": items, "correct": correct,
        "item_accuracy": round(correct / items, 4) if items else None,
        "episodes_fully_correct": sum(r["items"] and r["correct"] == r["items"] for r in rows),
        "escalated": sum(r["status"] != "done" for r in rows),
        "duplicates": sum(r["duplicates"] for r in rows),
        "mean_actions": round(statistics.mean(r["actions"] for r in rows), 1),
        "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 3),
        "failures": [{k: r[k] for k in ("seed", "correct", "items", "status", "reason")} for r in rows if r["items"] and r["correct"] != r["items"]][:6],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tasks", default="access,shop,recon,chart")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=9001)
    ap.add_argument("--out", type=Path, default=Path("eval/results/learned_intention_ranker.json"))
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    encoder_path = args.model / "label_encoder.npz"
    encoder = LabelEncoder.load(encoder_path) if encoder_path.exists() else None
    report = {"model": str(args.model), "seeds": [args.first_seed, args.first_seed + args.episodes - 1], "tasks": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        base, _ = harness.serve()
        for name in args.tasks.split(","):
            task = harness.tasks([name])[name]
            task.bindings()
            seeds = range(args.first_seed, args.first_seed + args.episodes)
            report["tasks"][name] = {
                "hand_written_objective": run_arm(browser, base, task, seeds, UtilityChooser()),
                "learned_ranker": run_arm(browser, base, task, seeds, LearnedChooser.load(args.model / "intention_ranker.npz", encoder=encoder, source=str(args.model))),
            }
        browser.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"{'task':<9} {'arm':<24} {'items':<12} {'acc':<7} {'eps ok':<8} {'esc':<5} {'dup':<5} actions")
    for name, arms in report["tasks"].items():
        for arm, s in arms.items():
            acc = f"{s['item_accuracy']:.4f}" if s["item_accuracy"] is not None else "n/a"
            print(f"{name:<9} {arm:<24} {s['correct']}/{s['items']:<9} {acc:<7} {s['episodes_fully_correct']}/{s['episodes']:<6} {s['escalated']:<5} {s['duplicates']:<5} {s['mean_actions']}")


if __name__ == "__main__":
    main()
