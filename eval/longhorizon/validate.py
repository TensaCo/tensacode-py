"""Pre-registration check: every hidden checker must pass its reference solution and fail an empty workspace.

    LH_SCRATCH=/path PYTHONPATH=src:. python -m eval.longhorizon.validate [--tasks id,id]

Writes eval/results/longhorizon_validation.json, including a digest of each checker and task
instruction, so the record shows the checkers were fixed before any agent ran.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import shutil
import time
from pathlib import Path

from . import tasks as T

REF = Path(__file__).parent / "reference"
OUT = Path(__file__).parents[2] / "eval" / "results" / "longhorizon_validation.json"


def digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
    return h.hexdigest()[:16]


def solve(task: T.Task, w: Path) -> str:
    """Apply the reference solution for a task inside the sandbox; returns a note about how."""
    if task.id == "own/coding_ledger":
        shutil.copy(REF / "ledger.py", w / "ledger.py")
        return "copied reference ledger.py"
    if task.id == "own/data_cleaning":
        shutil.copy(REF / "clean.py", w / "clean.py")
        r = T.sandbox_run(w, [T.PY, "/app/clean.py"], timeout=300)
        return f"ran reference clean.py (rc={r.returncode})"
    if task.id == "own/cad_bracket":
        shutil.copy(REF / "bracket.py", w / "bracket.py")
        r = T.sandbox_run(w, [T.PY, "/app/bracket.py", "/app/out/bracket.stl", "/app/out/report.json"], timeout=900)
        return f"ran reference bracket.py (rc={r.returncode}) {r.stderr[-200:] if r.returncode else ''}"
    if task.id == "own/eeg_motor_imagery":
        shutil.copy(REF / "analyse_eeg.py", w / "analyse.py")
        r = T.sandbox_run(w, [T.PY, "/app/analyse.py", "/app/data", "/app/out"], timeout=1800)
        return f"ran reference analyse.py (rc={r.returncode}) {r.stderr[-300:] if r.returncode else ''}"
    if task.id == "own/service_incident":
        shutil.copy(REF / "fix_incident.sh", w / "_fix.sh")
        r = T.sandbox_run(w, ["/bin/bash", "/app/_fix.sh"], timeout=300)
        (w / "_fix.sh").unlink(missing_ok=True)
        return f"ran reference fix (rc={r.returncode}) {r.stderr[-200:] if r.returncode else ''}"
    if task.id.startswith("tb2/"):
        name = task.id.split("/", 1)[1]
        script = (T.TB2 / name / "solution" / "solve.sh").read_text()
        # the oracle installs packages; this sandbox has no network and no root, so drop those lines
        cleaned = "\n".join(ln for ln in script.splitlines() if not re.match(r"\s*(apt-get|apt |pip install|curl -LsSf|source \$HOME/\.local)", ln))
        (w / "_solve.sh").write_text(cleaned)
        r = T.sandbox_run(w, ["/bin/bash", "/app/_solve.sh"], timeout=1200)
        (w / "_solve.sh").unlink(missing_ok=True)
        return f"ran oracle solve.sh without package installs (rc={r.returncode}) {(r.stderr or '')[-200:]}"
    return "no reference solution"


def make_truth(task: T.Task) -> None:
    """Some checkers compare against a reference measurement; produce it once, from the reference solution."""
    T.TRUTH.mkdir(parents=True, exist_ok=True)
    if task.id == "own/cad_bracket" and not (T.TRUTH / "cad_bracket.json").exists():
        w = T.SP / "lh" / "truthwork" / "cad"
        shutil.rmtree(w, ignore_errors=True)
        w.mkdir(parents=True)
        task.setup(w)
        solve(task, w)
        probe = json.dumps({"stl": "/app/out/bracket.stl", "solid": [], "void": []})
        (w / ".p.py").write_text("import json,sys,trimesh\nm=trimesh.load(json.loads(sys.argv[1])['stl'])\nprint(json.dumps({'volume':float(m.volume),'bbox':[float(x) for x in m.extents],'watertight':bool(m.is_watertight)}))\n")
        r = T.sandbox_run(w, [T.PY, "/app/.p.py", probe], timeout=600)
        (T.TRUTH / "cad_bracket.json").write_text(r.stdout.strip().splitlines()[-1])
    if task.id == "own/eeg_motor_imagery" and not (T.TRUTH / "eeg_motor_imagery.json").exists():
        w = T.SP / "lh" / "truthwork" / "eeg"
        shutil.rmtree(w, ignore_errors=True)
        w.mkdir(parents=True)
        task.setup(w)
        note = solve(task, w)
        got = json.loads((w / "out" / "results.json").read_text())
        per = {s: v["cv_accuracy"] for s, v in got["per_subject"].items()}
        counts = [v["n_epochs"] for v in got["per_subject"].values()]
        (T.TRUTH / "eeg_motor_imagery.json").write_text(json.dumps({
            "per_subject": per, "mean": got["mean_cv_accuracy"], "mean_floor": 0.60,
            "n_epochs_min": min(counts) - 2, "n_epochs_max": max(counts) + 2, "reference_note": note}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()
    rows = []
    for task in T.by_id(args.tasks.split(",") if args.tasks else None):
        make_truth(task)
        base = T.SP / "lh" / "validate" / task.id.replace("/", "__")
        empty, solved = base / "empty", base / "solved"
        for d in (empty, solved):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            task.setup(d)
        t0 = time.time()
        empty_result = task.check(empty)
        note = solve(task, solved)
        solved_result = task.check(solved)
        row = {"task": task.id, "split": task.split, "source": task.source, "difficulty": task.difficulty,
               "instruction_digest": digest(task.instruction), "checker_digest": digest(inspect.getsource(task.check) if not task.id.startswith("tb2/") else (T.TB2 / task.id.split("/", 1)[1] / "tests" / "test_outputs.py").read_text()),
               "empty_passes": empty_result["passed"], "empty_score": empty_result["score"], "reference_passes": solved_result["passed"], "reference_score": solved_result["score"],
               "reference_note": note, "reference_detail": solved_result.get("detail", "")[:400], "empty_detail": empty_result.get("detail", "")[:200], "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        ok = (not row["empty_passes"]) and row["reference_passes"]
        print(f"{'OK  ' if ok else 'BAD '} {task.id:<34} empty={row['empty_score']:<5} reference={row['reference_score']:<5} {row['reference_detail'][:120]}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "tasks": rows}, indent=1))
    good = sum((not r["empty_passes"]) and r["reference_passes"] for r in rows)
    print(f"\n{good}/{len(rows)} checkers validated (reference passes, empty fails)")


if __name__ == "__main__":
    main()
