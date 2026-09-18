"""Turn longhorizon.json into the tables for docs/revival/08-long-horizon.md.

    PYTHONPATH=src:. python -m eval.longhorizon.report [--labels asis,improved,react]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parents[2] / "eval" / "results" / "longhorizon.json"


def load(labels: list[str]) -> dict[str, dict]:
    data = json.loads(RESULTS.read_text())
    runs = data.get("runs", {})
    return {label: runs[label] for label in labels if label in runs}


def row_map(run: dict) -> dict[str, dict]:
    return {r["task"]: r for r in run["tasks"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="asis,improved,react")
    args = ap.parse_args()
    labels = args.labels.split(",")
    runs = load(labels)
    if not runs:
        print("no runs yet")
        return
    by_label = {label: row_map(run) for label, run in runs.items()}
    tasks = sorted({t for rows in by_label.values() for t in rows})

    head = "| Task | Split | " + " | ".join(f"{label} score" for label in by_label) + " |"
    print(head)
    print("| --- | --- | " + " | ".join("---" for _ in by_label) + " |")
    for task in tasks:
        cells = []
        split = next((rows[task]["split"] for rows in by_label.values() if task in rows), "?")
        for label, rows in by_label.items():
            r = rows.get(task)
            if r is None:
                cells.append("–")
            else:
                cells.append("**PASS**" if r["passed"] else f"{r['score']:.2f}")
        print(f"| `{task}` | {split} | " + " | ".join(cells) + " |")

    print()
    for label, rows in by_label.items():
        rs = list(rows.values())
        if not rs:
            continue
        passed = sum(r["passed"] for r in rs)
        print(f"{label}: {passed}/{len(rs)} passed | mean score {sum(r['score'] for r in rs) / len(rs):.2f} "
              f"| model calls {sum(r['model_calls'] for r in rs)} | commands {sum(r['commands'] for r in rs)} "
              f"| wall {sum(r['wall_seconds'] for r in rs) / 3600:.1f} h")
        for split in ("tune", "heldout"):
            sub = [r for r in rs if r["split"] == split]
            if sub:
                print(f"    {split}: {sum(r['passed'] for r in sub)}/{len(sub)} passed, mean score {sum(r['score'] for r in sub) / len(sub):.2f}")
        statuses: dict[str, int] = {}
        for r in rs:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        print(f"    end states: {statuses}")

    print("\nper-task detail")
    for label, rows in by_label.items():
        for task, r in rows.items():
            checks = r.get("checks") or {}
            failed = [k for k, v in checks.items() if not v]
            print(f"  {label:<9} {task:<30} score={r['score']:<5} status={r['status']:<14} calls={r['model_calls']:<4} cmds={r['commands']:<4} {r['wall_seconds']:>6.0f}s  failed_checks={failed or '-'}")


if __name__ == "__main__":
    main()


def teacher_usage(path: Path) -> dict:
    """Totals from a teacher log: calls, seconds, and tokens where the log has them."""
    calls = seconds = prompt_tokens = new_tokens = 0
    approx = 0
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        calls += 1
        seconds += row.get("seconds") or 0
        if row.get("prompt_tokens"):
            prompt_tokens += row["prompt_tokens"]
            new_tokens += row.get("new_tokens") or 0
        else:  # older logs kept only the text
            approx += 1
            prompt_tokens += (len(row.get("system", "")) + len(row.get("user", ""))) // 4
            new_tokens += len(row.get("reply") or "") // 4
    return {"calls": calls, "seconds": round(seconds, 1), "prompt_tokens": prompt_tokens, "new_tokens": new_tokens, "token_counts_approximate_for": approx}
