"""The desktop chore on the computerworld engine: accuracy, speed, and exact reproducibility.

    python -m eval.computerworld_bench [--episodes 40] [--out eval/results/computerworld.json]

Each episode builds its own world from a definition plus a seed, so nothing carries over
between runs. Besides accuracy and speed this checks two things the previous simulator
could not offer: replaying a seed reaches the same ``state_hash``, and two different seeds
do not.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import tensacode as tc

from examples.browser_agents import harness
from examples.browser_agents.mind import run_mind

OUT = Path(__file__).parents[1] / "eval" / "results" / "computerworld.json"


def episode(task, seed: int) -> dict:
    ui, world = harness.body_for(task, seed)
    runtime = harness.runtime_for(task)
    t0 = time.perf_counter()
    error, outcome = None, None
    with tc.use(runtime):
        try:
            outcome = run_mind(ui, task.spec)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - t0
    score = task.score(world, seed)
    return {
        "seed": seed, "correct": score.get("correct", 0), "items": score.get("items", 0),
        "checks": score.get("checks", {}), "duplicates": score.get("duplicates", 0),
        "status": getattr(outcome, "status", "error"), "reason": getattr(outcome, "reason", error),
        "seconds": round(seconds, 4), "cycles": getattr(outcome, "cycles", 0), "actions": ui.stats.actions,
        "observations": ui.stats.observations, "perception_ms_p50": round(statistics.median(ui.perception_ms), 3) if ui.perception_ms else None,
        "state_hash": world.state_hash(), "error": error,
    }


def determinism(task, seed: int) -> dict:
    """Same seed twice, then a different seed: hashes must repeat, then differ."""
    first, second = episode(task, seed), episode(task, seed)
    other = episode(task, seed + 1)
    return {
        "seed": seed, "repeatable": first["state_hash"] == second["state_hash"],
        "seed_changes_world": first["state_hash"] != other["state_hash"],
        "hash": first["state_hash"], "other_hash": other["state_hash"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    task = harness.tasks(["desktop"])["desktop"]
    assert task.simulated, "the desktop task should be running inside the engine"

    t0 = time.perf_counter()
    rows = [episode(task, seed) for seed in range(1, args.episodes + 1)]
    wall = time.perf_counter() - t0
    secs = [r["seconds"] for r in rows]
    perception = [r["perception_ms_p50"] for r in rows if r["perception_ms_p50"] is not None]
    import computerworld

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "engine": "computerworld", "engine_python_package": getattr(computerworld, "__version__", "0.1.0a1"),
            "browser": "none", "simulator_server": "none",
        },
        "episodes": len(rows),
        "items_correct": sum(r["correct"] for r in rows),
        "items": sum(r["items"] for r in rows),
        "episodes_fully_correct": sum(r["correct"] == r["items"] and r["items"] > 0 for r in rows),
        "duplicate_effects": sum(r["duplicates"] for r in rows),
        "seconds_per_episode": {"mean": round(statistics.mean(secs), 4), "p50": round(statistics.median(secs), 4), "max": round(max(secs), 4)},
        "episodes_per_second": round(len(rows) / wall, 2),
        "actions_per_episode": round(statistics.mean(r["actions"] for r in rows), 1),
        "actions_per_second": round(sum(r["actions"] for r in rows) / sum(secs), 1),
        "observations_per_episode": round(statistics.mean(r["observations"] for r in rows), 1),
        "perception_ms_p50": round(statistics.median(perception), 3) if perception else None,
        "model_calls": 0,
        "determinism": [determinism(task, seed) for seed in (1, 7)],
        "failures": [{k: r[k] for k in ("seed", "correct", "items", "status", "reason", "checks")} for r in rows if r["correct"] != r["items"]][:10],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"desktop on computerworld: items {report['items_correct']}/{report['items']} "
          f"episodes ok {report['episodes_fully_correct']}/{report['episodes']} "
          f"{report['seconds_per_episode']['mean']}s/ep ({report['episodes_per_second']}/s) "
          f"{report['actions_per_second']} actions/s perception p50 {report['perception_ms_p50']}ms")
    for check in report["determinism"]:
        print(f"  seed {check['seed']}: repeatable={check['repeatable']} seed_changes_world={check['seed_changes_world']}")
    for failure in report["failures"]:
        print("  FAIL", failure)


if __name__ == "__main__":
    main()
