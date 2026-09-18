"""Swap the learned encoder in for the access agent's label scorer and run real episodes.

    PYTHONPATH=src:. python eval/training/eval_access_labels.py --model DIR --episodes 30

``tasks/access.py`` scores a form field against hand-written synonym lists with
``label_similarity``. This replaces that one function with the learned encoder's
similarity — same agent, same rules, same threshold — and runs the same seeds both ways.

The app picks one of four label variants per field at random, and the synonym lists know
three of them, so seeds differ in how much they ask of the matcher. Reported separately:
all seeds, and the subset whose form shows at least one variant the lists do not contain.
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
from examples.browser_agents.learned.text_embed import LabelEncoder  # noqa: E402
from eval.training.train_label_encoder import APP_LABELS, HELD_OUT  # noqa: E402

KNOWN = {v for labels in APP_LABELS.values() for v in labels[:-1]}


def run(episodes: int, first_seed: int, scorer) -> list[dict]:
    """Run access episodes with ``scorer`` installed as the label similarity function."""
    from playwright.sync_api import sync_playwright

    from examples.browser_agents.tasks import access, access_script

    original = access.label_similarity
    access.label_similarity = scorer
    access_script.label_similarity = scorer
    rows = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            base, _ = harness.serve()
            task = harness.tasks(["access"])["access"]
            task.bindings()
            context = browser.new_context(viewport={"width": task.viewport[0], "height": task.viewport[1]})
            harness.block_outside_network(context, base)
            page = context.new_page()
            for seed in range(first_seed, first_seed + episodes):
                t0 = time.perf_counter()
                result = harness.run_episode(context, base, task, seed, page=page)
                variants = (result.score or {}).get("variants") or {}
                rows.append({
                    "seed": seed, "items": result.score.get("items", 0), "correct": result.score.get("correct", 0),
                    "duplicates": result.score.get("duplicates", 0), "status": getattr(result.outcome, "status", "error"),
                    "reason": getattr(result.outcome, "reason", result.error), "actions": result.actions,
                    "seconds": round(time.perf_counter() - t0, 3),
                    "unseen_variants": sorted(v for k, v in variants.items() if k in HELD_OUT and v not in KNOWN),
                })
            browser.close()
    finally:
        access.label_similarity = original
        access_script.label_similarity = original
    return rows


def summarize(rows: list[dict]) -> dict:
    def over(sel: list[dict]) -> dict:
        items = sum(r["items"] for r in sel)
        correct = sum(r["correct"] for r in sel)
        return {
            "episodes": len(sel), "items": items, "correct": correct,
            "item_accuracy": round(correct / items, 4) if items else None,
            "episodes_fully_correct": sum(r["items"] and r["correct"] == r["items"] for r in sel),
            "escalated": sum(r["status"] != "done" for r in sel),
            "duplicates": sum(r["duplicates"] for r in sel),
            "mean_actions": round(statistics.mean(r["actions"] for r in sel), 1) if sel else None,
        }

    unseen = [r for r in rows if r["unseen_variants"]]
    return {"all": over(rows), "seeds_showing_an_unseen_variant": over(unseen), "n_unseen_seeds": len(unseen)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=9101)
    ap.add_argument("--out", type=Path, default=Path("eval/results/learned_access_labels.json"))
    args = ap.parse_args()

    from examples.browser_agents.browser import label_similarity as rules_similarity

    encoder = LabelEncoder.load(args.model / "label_encoder.npz")
    report = {
        "model": str(args.model), "seeds": [args.first_seed, args.first_seed + args.episodes - 1],
        "rules": summarize(run(args.episodes, args.first_seed, rules_similarity)),
        "learned": summarize(run(args.episodes, args.first_seed, encoder.similarity)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    for arm in ("rules", "learned"):
        for split, s in report[arm].items():
            if split == "n_unseen_seeds":
                continue
            acc = f"{s['item_accuracy']:.4f}" if s["item_accuracy"] is not None else "n/a"
            print(f"{arm:<8} {split:<34} items={s['correct']}/{s['items']:<5} acc={acc:<7} eps_ok={s['episodes_fully_correct']}/{s['episodes']:<4} esc={s['escalated']} dup={s['duplicates']} actions={s['mean_actions']}")


if __name__ == "__main__":
    main()
