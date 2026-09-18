"""The chart agent in the environment we did NOT change, and in the one we did.

After the chart agent mis-read two pixel-identical bars, the generator was changed to redraw
until the tallest bar is visibly tallest, and the score went to 100%. That is an environment
change following an agent failure, so the honest headline has to come from the original
generator, with the agent's near-tie abstention doing the work instead.

    python eval/chart_environment.py --episodes 40

`ties=allow` is the original generator (pixel ties possible; the agent should abstain on them).
`ties=avoid` is the changed generator, kept for comparison only.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from examples.browser_agents import harness

OUT = Path(__file__).resolve().parents[1] / "eval" / "results" / "chart_environment.json"


KNOWN_TIE_SEEDS = [199, 4014, 4077, 4109, 4142]  # found when the generator was changed: 5 of 400 seeds had a pixel tie


def run(config: str, episodes: int, first_seed: int, seeds: list[int] | None = None) -> dict:
    from playwright.sync_api import sync_playwright

    task = harness.tasks(["chart"])["chart"]
    suffix = "" if config == "allow" else "&ties=avoid"
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": task.viewport[0], "height": task.viewport[1]})
        base, _ = harness.serve()
        harness.block_outside_network(context, base)
        page = context.new_page()
        chart = harness.Task(task.name, lambda b, s: f"{b}/chart.html?seed={s}{suffix}", task.spec, task.bindings,
                             task.setup, task.score, task.viewport)
        for seed in (seeds if seeds is not None else range(first_seed, first_seed + episodes)):
            r = harness.run_episode(context, base, chart, seed, page=page)
            o = r.outcome
            rows.append({"seed": seed, "items": r.score.get("items", 0), "correct": r.score.get("correct", 0),
                         "status": getattr(o, "status", "error"), "reason": getattr(o, "reason", r.error),
                         "seconds": round(r.seconds, 3), "truth": r.score.get("truth", {}).get("values"),
                         "answers": r.score.get("answers")})
        browser.close()
    items = sum(r["items"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    abstained = [r for r in rows if r["status"] == "escalated" and "too close to call" in (r["reason"] or "")]
    other_escalations = [r for r in rows if r["status"] == "escalated" and r not in abstained]
    wrong = [r for r in rows if r["status"] == "done" and r["correct"] < r["items"]]
    return {
        "config": config,
        "generator": ("ORIGINAL: pixel ties possible" if config == "allow"
                      else "CHANGED after an agent failure: redraws until the tallest bar is visibly tallest"),
        "episodes": len(rows),
        "items_correct": correct, "items": items,
        "item_accuracy": round(correct / max(1, items), 4),
        "episodes_fully_correct": sum(r["correct"] == r["items"] for r in rows),
        "abstained_near_tie": len(abstained),
        "escalated_other": len(other_escalations),
        "answered_and_wrong": len(wrong),
        "seconds_per_episode": round(sum(r["seconds"] for r in rows) / max(1, len(rows)), 3),
        "abstention_reasons": [r["reason"] for r in abstained][:5],
        "wrong_examples": [{"seed": r["seed"], "reason": r["reason"], "answers": r["answers"]} for r in wrong][:5],
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--first-seed", type=int, default=5000)
    ap.add_argument("--seeds", default="", help="explicit comma-separated seeds instead of a range")
    ap.add_argument("--label", default="", help="label for this run inside the report")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    report = {"date": time.strftime("%Y-%m-%d %H:%M"),
              "environment": {"python": platform.python_version(), "platform": platform.platform()},
              "note": ("Fresh seeds, never used for tuning. The `allow` row is the headline: it is the generator "
                       "as originally written, which we did not change after failing in it. Items are scored by "
                       "the app's own window.__score(), so the grader is still ours (docs/revival/11)."),
              "configs": {}}
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or None
    if seeds:
        report["note"] += f" This run uses explicit seeds {seeds}."
    for config in ("allow", "avoid"):
        print(f"=== ties={config}", flush=True)
        r = run(config, args.episodes, args.first_seed, seeds)
        report["configs"][config] = r
        print(f"  items {r['items_correct']}/{r['items']} ({r['item_accuracy']:.3f}); "
              f"near-tie abstentions {r['abstained_near_tie']}; answered-and-wrong {r['answered_and_wrong']}", flush=True)
        args.out.write_text(json.dumps(report, indent=1, default=str))
    a, b = report["configs"]["allow"], report["configs"]["avoid"]
    report["comparison"] = {
        "item_accuracy_original_generator": a["item_accuracy"],
        "item_accuracy_changed_generator": b["item_accuracy"],
        "difficulty_removed_by_the_change": round(b["item_accuracy"] - a["item_accuracy"], 4),
        "near_tie_abstentions_original": a["abstained_near_tie"],
        "verdict": ("The change removed real difficulty; the original generator is the honest headline."
                    if b["item_accuracy"] > a["item_accuracy"] + 0.001 else
                    "The change made no measurable difference on these seeds; the tie case is rare."),
    }
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print("\nwrote", args.out, flush=True)


if __name__ == "__main__":
    main()
