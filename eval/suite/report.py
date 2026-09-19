"""The scorecard: every registered task, its latest result, and what is missing.

    python -m eval.suite.report [--subject agent:learned:desktop+vision] [--split dev]

Built from the registry, not from a list written here, so a task cannot be left out of
the picture by forgetting to mention it: if it has never been run, it says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

from eval.suite import store  # noqa: E402
from eval.suite.core import registry  # noqa: E402

FLAGS = {"self_authored": "self-authored data or world: regression evidence, never a headline"}


def value_of(task, row) -> str:
    if row is None:
        return "not run"
    m = row["metrics"]
    if task.headline == "score" and m.get("score") is not None:
        return f"{m['score']:.3f}"
    if m.get("accuracy") is not None:
        return f"{m['accuracy']:.3f}"
    if m.get("n"):
        return f"{m.get('answered', 0)}/{m['n']} answered"
    return "–"


def _status(task, split: str) -> str:
    """Whether this task can produce a row, and if not, what is missing.

    A task whose data is present but whose items were never wired up reported as *ready* and
    then quietly produced nothing — the scorecard showed a measurement that could not happen.
    """
    if not task.dataset.available():
        return task.dataset.status()
    try:
        if not list(task.items(split)):
            return "not wired"
    except Exception:  # noqa: BLE001 - a loader that raises is not ready either
        return "loader fails"
    return task.dataset.status()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="agent:learned:desktop+vision")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    tasks = registry()
    rows = []
    for tid, task in sorted(tasks.items(), key=lambda kv: (kv[1].area, kv[0])):
        row = store.latest(tid, args.subject, args.split)
        control = store.latest(tid, "control:abstain", args.split)
        rows.append({
            "task": tid, "area": task.area, "what": task.what, "status": _status(task, args.split),
            "value": value_of(task, row), "metric": task.headline,
            "control(abstain)": value_of(task, control) if control else "–",
            "n": (row or {}).get("metrics", {}).get("n"),
            "ci": (row or {}).get("metrics", {}).get("accuracy_ci95"),
            "underpowered": (row or {}).get("metrics", {}).get("underpowered"),
            "wrong": (row or {}).get("metrics", {}).get("wrong"),
            "dataset": task.dataset.name, "license": task.dataset.license,
            "self_authored": task.self_authored, "notes": task.notes,
        })
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    task_width = max(len(r["task"]) for r in rows)
    print(f"subject {args.subject}   split {args.split}\n")
    area = None
    for r in rows:
        if r["area"] != area:
            area = r["area"]
            print(f"\n{area.upper()}")
        wrong = f"  wrong={r['wrong']}" if r["wrong"] else ""
        power = "  (n<30: underpowered)" if r.get("underpowered") else ""
        ci = f"  ci95={r['ci']}" if r.get("ci") else ""
        print(f"  {r['task']:{task_width}}  {r['value']:>14}  {r['status']:12} {r['dataset'][:28]:28}{wrong}{ci}{power}")
        if r["self_authored"]:
            print(f"  {'':{task_width}}  {FLAGS['self_authored']}")
    ready = [r for r in rows if r["status"] == "ready"]
    run = [r for r in ready if r["value"] != "not run"]
    print(f"\n{len(rows)} tasks: {len(ready)} have data, {len(run)} have a result for this subject, "
          f"{len(rows) - len(ready)} need data.")
    print("needs data:", ", ".join(sorted(r["task"] for r in rows if r["status"] != "ready")))
    print("has data, never run:", ", ".join(sorted(r["task"] for r in ready if r["value"] == "not run")) or "—")


if __name__ == "__main__":
    main()
