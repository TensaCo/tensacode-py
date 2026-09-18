"""Benchmark the cognitive browser agents (headless, no screencast).

    python -m examples.browser_agents.bench [--episodes 40] [--tasks access,shop,recon,chart,inbox,desktop]

One process per task. Seeds are 1..N for every task. The desktop task runs inside the
computerworld engine (no browser, no server); the inbox task needs TENSORCODE_BANKING77_DIR.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import statistics
import sys
import time
from pathlib import Path

from . import harness

OUT = Path(__file__).parents[2] / "eval" / "results" / "browser_agents.json"


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4) if xs else float("nan")


def run_task(name: str, episodes: int, out: mp.Queue) -> None:
    from playwright.sync_api import sync_playwright

    task = harness.tasks([name])[name]
    task.bindings()  # train/load any learned tiers before timing
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": task.viewport[0], "height": task.viewport[1]})
        base, _ = harness.serve()
        harness.block_outside_network(context, base)
        page = context.new_page()
        for seed in range(1, episodes + 1):
            r = harness.run_episode(context, base, task, seed, page=page)
            o = r.outcome
            rows.append({
                "seed": seed, "items": r.score.get("items", 0), "correct": r.score.get("correct", 0), "duplicates": r.score.get("duplicates", 0),
                "seconds": r.seconds, "actions": r.actions, "observations": r.observations, "browser_s": r.browser_s,
                "tensacode_ops_ms": r.tensacode_ms, "think_ms": getattr(o, "think_ms", 0.0), "cycles": getattr(o, "cycles", 0),
                "beliefs": getattr(o, "beliefs", 0), "status": getattr(o, "status", "error"), "reason": getattr(o, "reason", r.error),
                "model_calls": r.model_calls, "error": r.error,
            })
        browser.close()
    out.put((name, rows, harness.model_libraries_loaded()))


def summarize(name: str, rows: list[dict], libs: list[str]) -> dict:
    secs = [r["seconds"] for r in rows]
    items, correct = sum(r["items"] for r in rows), sum(r["correct"] for r in rows)
    actions = sum(r["actions"] for r in rows)
    total_s = sum(secs)
    browser = sum(r["browser_s"] for r in rows)
    think = sum(r["think_ms"] for r in rows) / 1e3
    ops = sum(r["tensacode_ops_ms"] for r in rows) / 1e3
    return {
        "task": name,
        "episodes": len(rows),
        "items_correct": correct,
        "items": items,
        "item_accuracy": round(correct / items, 4) if items else None,
        "episodes_fully_correct": sum(r["correct"] == r["items"] for r in rows),
        "escalated_episodes": sum(r["status"] != "done" for r in rows),
        "duplicate_effects": sum(r["duplicates"] for r in rows),
        "seconds_per_episode": {"mean": round(statistics.mean(secs), 3), "p50": pct(secs, 0.5), "p95": pct(secs, 0.95)},
        "ui_actions_per_episode": round(actions / len(rows), 1),
        "ui_actions_per_second": round(actions / total_s, 1),
        "time_split_share": {
            "browser (input dispatch, DOM reads, waiting for the app)": round(browser / total_s, 3),
            "tensorcode ops (parse/classify/choose/rank/check spans)": round(ops / total_s, 3),
            "think (rule firing incl. nested ops)": round(think / total_s, 3),
        },
        "mean_beliefs_at_end": round(statistics.mean(r["beliefs"] for r in rows), 1),
        "model_calls": sum(r["model_calls"] for r in rows),
        "model_libraries_loaded": libs,
        "failures": [{k: r[k] for k in ("seed", "correct", "items", "status", "reason", "error")} for r in rows if r["correct"] != r["items"]][:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--tasks", default="access,shop,recon,chart,inbox,desktop")
    args = ap.parse_args()
    names = args.tasks.split(",")
    q: mp.Queue = mp.get_context("spawn").Queue()
    t0 = time.time()
    procs = [mp.get_context("spawn").Process(target=run_task, args=(n, args.episodes, q)) for n in names]
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()
    import playwright
    import sklearn

    report = {
        "date": time.strftime("%Y-%m-%d"),
        "wall_seconds": round(time.time() - t0, 1),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "playwright": getattr(playwright, "__version__", "?"), "scikit_learn": sklearn.__version__, "browser": "Chromium headless shell (Playwright)", "parallel_processes": len(procs)},
        "tasks": [summarize(*r) for r in sorted(results)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=str))
    for t in report["tasks"]:
        print(f"{t['task']:<8} items {t['items_correct']}/{t['items']} ({t['item_accuracy']:.1%})  episodes ok {t['episodes_fully_correct']}/{t['episodes']}  dup {t['duplicate_effects']}  {t['seconds_per_episode']['mean']}s/ep  {t['ui_actions_per_second']} actions/s  model calls {t['model_calls']}  libs {t['model_libraries_loaded']}")
    _ = sys


if __name__ == "__main__":
    main()
