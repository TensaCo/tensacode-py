"""Long-horizon tasks with pre-registered, hidden checkers.

Every task prepares a working directory that becomes /app inside the sandbox. The agent
sees only that directory and the terminal. Checks run afterwards, outside the agent's
reach, and score the *state* of the directory (files, databases, repositories, a service
it must make work) or run hidden tests the agent never sees.

Sources
    tb2/<name>   adapted from Terminal-Bench 2 (Apache-2.0), reproduced without Docker:
                 the environment's setup steps are replayed into the working directory and
                 the task's own pytest suite is the checker. Never used for tuning.
    own/<name>   written for this evaluation: CAD, EEG analysis, a coding project with a
                 hidden test suite, a data-cleaning pipeline with a hidden answer key, and
                 a broken-service incident.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SP = Path(os.environ.get("LH_SCRATCH", "/tmp/lh"))
TB2 = Path(os.environ.get("TB2_DIR", SP / "bench" / "tb2"))
TOOLS = SP / "lh" / "tools"
DATA = SP / "lh" / "data"
TRUTH = SP / "lh" / "truth"
PY = "/opt/tools/bin/python"


@dataclass
class Task:
    id: str
    source: str
    difficulty: str
    split: str  # "tune" (transcripts may be inspected while improving) | "heldout"
    instruction: str
    setup: Callable[[Path], None]
    check: Callable[[Path], dict]
    minutes: float = 45.0
    max_model_calls: int = 150
    notes: str = ""


# ------------------------------------------------------------------ helpers


def sandbox_run(workdir: Path, command: list[str], *, binds: list[tuple[Path, str]] = (), timeout: float = 600, network: bool = False) -> subprocess.CompletedProcess:
    """Run a command in the same sandbox shape the agent had, for setup and checking."""
    from .webterm import bwrap_command

    cmd = bwrap_command(workdir, TOOLS, network=network)
    for src, dst in binds:
        cmd += ["--ro-bind", str(src), dst]
    return subprocess.run(cmd + command, capture_output=True, text=True, timeout=timeout)


def pytest_check(workdir: Path, tests: Path, *, copy_into_app: list[str] = (), timeout: float = 900) -> dict:
    """Run a task's hidden pytest suite against the agent's /app."""
    staged = workdir / ".verify"
    staged.mkdir(exist_ok=True)
    for name in copy_into_app:
        shutil.copy(tests / name, workdir / name)
    out = sandbox_run(workdir, [PY, "-m", "pytest", "-q", "-rA", "--no-header", "/tests/test_outputs.py"], binds=[(tests, "/tests")], timeout=timeout)
    shutil.rmtree(staged, ignore_errors=True)
    for name in copy_into_app:
        (workdir / name).unlink(missing_ok=True)
    text = (out.stdout + out.stderr)[-4000:]
    passed = out.returncode == 0
    return {"passed": passed, "score": 1.0 if passed else 0.0, "detail": text.strip().splitlines()[-1] if text.strip() else f"pytest rc={out.returncode}", "log": text}


def tb2_task(name: str, *, setup: Callable[[Path], None], split: str = "heldout", minutes: float = 45.0, copy_into_app: list[str] = ()) -> Task:
    d = TB2 / name
    meta = (d / "task.toml").read_text()
    difficulty = next((ln.split('"')[1] for ln in meta.splitlines() if ln.startswith("difficulty")), "?")
    return Task(
        id=f"tb2/{name}", source="Terminal-Bench 2 (Apache-2.0)", difficulty=difficulty, split=split,
        instruction=(d / "instruction.md").read_text().strip(),
        setup=setup,
        check=lambda w, d=d: pytest_check(w, d / "tests", copy_into_app=copy_into_app),
        minutes=minutes,
        notes="reproduced without Docker; the task's own tests are the checker",
    )


def _write(workdir: Path, rel: str, text: str) -> None:
    p = workdir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text))


# ------------------------------------------------------ tb2 task setups


def _setup_log_summary(w: Path) -> None:
    gen = TB2 / "log-summary-date-ranges" / "environment" / "log_generator_deterministic.py"
    shutil.copy(gen, w / "_gen.py")
    sandbox_run(w, [PY, "/app/_gen.py"], timeout=300)
    (w / "_gen.py").unlink(missing_ok=True)


def _setup_sqlite_trunc(w: Path) -> None:
    shutil.copy(TB2 / "sqlite-db-truncate" / "environment" / "trunc.db", w / "trunc.db")


def _setup_cancel_async(w: Path) -> None:
    pass  # empty /app; the hidden suite brings its own helper


def _setup_git_leak(w: Path) -> None:
    script = TB2 / "git-leak-recovery" / "environment" / "challenge-setup.sh"
    shutil.copy(script, w / "_setup.sh")
    sandbox_run(w, ["/bin/bash", "/app/_setup.sh"], timeout=300)
    (w / "_setup.sh").unlink(missing_ok=True)


# ------------------------------------------------------------------ own: CAD

CAD_SPEC = """
Design a parametric mounting bracket with build123d (already installed: use /opt/tools/bin/python) and export it.

Dimensions, in millimetres, with the plate centred on the origin:
  - Base plate: 120 long (x), 80 wide (y), 10 thick (z), sitting on z=0 so its top face is z=10.
  - Four through-holes, diameter 8, with centres 15 mm in from each side of the plate
    (that is, at x = +/-45, y = +/-25), going all the way through the plate.
  - A cylindrical boss on the middle of the top face: diameter 40, rising from z=10 to z=30.
  - A central bore, diameter 16, through the boss and the plate: from z=30 down to z=0.
  - A counterbore at the top of the boss: diameter 26, 5 deep (so from z=30 down to z=25).
  - A chamfer of 3 mm on the four vertical corner edges of the plate.

Deliverables:
  1. /app/bracket.py  - the script that builds the part, runnable with /opt/tools/bin/python.
  2. /app/out/bracket.stl - a watertight STL export of the part.
  3. /app/out/report.json - {"volume_mm3": <number>, "bbox_mm": [x, y, z], "holes": 4}
     with the volume and bounding box measured from the model you built (not hand-typed).

The part must be a single solid. Check your own work before you finish.
"""


def _setup_cad(w: Path) -> None:
    (w / "out").mkdir(exist_ok=True)


def _check_cad(w: Path) -> dict:
    checks: dict[str, bool] = {}
    stl = w / "out" / "bracket.stl"
    checks["stl_exists"] = stl.exists() and stl.stat().st_size > 1000
    checks["script_exists"] = (w / "bracket.py").exists()
    detail = []
    probe = json.dumps({
        "stl": "/app/out/bracket.stl",
        "solid": [[30, 0, 5], [0, 30, 5], [-30, -20, 2], [15, 0, 27], [0, 18, 20]],
        "void": [[45, 25, 5], [-45, 25, 5], [45, -25, 5], [-45, -25, 5], [0, 0, 5], [0, 0, 28], [5, 0, 27], [59, 39, 5], [0, 0, 35], [50, 0, 20], [70, 0, 5]],
    })
    script = """
import json, sys, numpy as np, trimesh
spec = json.loads(sys.argv[1])
m = trimesh.load(spec["stl"])
out = {"watertight": bool(m.is_watertight), "volume": float(m.volume), "bbox": [float(x) for x in m.extents]}
for key in ("solid", "void"):
    pts = np.array(spec[key], dtype=float)
    out[key] = [bool(v) for v in m.contains(pts)]
print(json.dumps(out))
"""
    _write(w, ".probe.py", script)
    res = sandbox_run(w, [PY, "/app/.probe.py", probe], timeout=300)
    (w / ".probe.py").unlink(missing_ok=True)
    measured = {}
    try:
        measured = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        detail.append(f"probe failed: {(res.stderr or res.stdout)[-200:]}")
    if measured:
        ref = json.loads((TRUTH / "cad_bracket.json").read_text())
        checks["watertight"] = measured["watertight"]
        checks["bbox"] = all(abs(a - b) < 0.8 for a, b in zip(sorted(measured["bbox"]), sorted(ref["bbox"])))
        checks["volume"] = abs(measured["volume"] - ref["volume"]) / ref["volume"] < 0.04
        checks["material_where_required"] = all(measured["solid"])
        checks["voids_where_required"] = not any(measured["void"])
        detail.append(f"volume {measured['volume']:.0f} vs ref {ref['volume']:.0f} mm3, bbox {[round(x, 1) for x in measured['bbox']]}")
    report = w / "out" / "report.json"
    try:
        r = json.loads(report.read_text())
        checks["report_volume_matches_model"] = measured and abs(float(r["volume_mm3"]) - measured["volume"]) / measured["volume"] < 0.01
        checks["report_holes"] = int(r.get("holes", 0)) == 4
    except Exception as exc:
        checks["report_volume_matches_model"] = False
        checks["report_holes"] = False
        detail.append(f"report.json unusable ({type(exc).__name__})")
    score = sum(bool(v) for v in checks.values()) / len(checks)
    return {"passed": all(checks.values()), "score": round(score, 3), "detail": "; ".join(detail), "checks": checks}


# ------------------------------------------------------------------ own: EEG

EEG_SPEC = """
Analyse the motor-imagery EEG in /app/data (PhysioNet EEGBCI, EDF files for subjects 1, 2 and 3,
runs 6, 10 and 14). MNE-Python, scikit-learn, numpy and matplotlib are installed for
/opt/tools/bin/python. There is no network access.

For each subject, run this pipeline:
  - load the three runs and concatenate them, standardise the channel names and montage;
  - band-pass filter 7-30 Hz;
  - build epochs for the two imagery classes (hands vs feet), from -1 s to +4 s around the cue;
  - classify with CSP (4 components) followed by LDA, scored with 5-fold cross-validation
    on the epochs of that subject.

Deliverables:
  1. /app/analyse.py - the script that does the whole analysis, runnable end to end.
  2. /app/out/results.json -
     {"per_subject": {"1": {"cv_accuracy": <0-1>, "n_epochs": <int>}, "2": {...}, "3": {...}},
      "mean_cv_accuracy": <0-1>}
     with numbers your script actually computed.
  3. /app/out/psd.png - a plot of the power spectral density of one subject's filtered data.
  4. /app/out/report.md - a short write-up: what you did, the per-subject accuracies, and
     one sentence on whether the result is above chance.

Do not invent numbers: every number in results.json and report.md must come from the data.
"""


def _setup_eeg(w: Path) -> None:
    (w / "out").mkdir(exist_ok=True)
    src = DATA / "eeg"
    dst = w / "data"
    dst.mkdir(exist_ok=True)
    for edf in sorted(src.rglob("*.edf")):
        shutil.copy(edf, dst / edf.name)


def _check_eeg(w: Path) -> dict:
    checks, detail = {}, []
    ref = json.loads((TRUTH / "eeg_motor_imagery.json").read_text())
    checks["script_exists"] = (w / "analyse.py").exists()
    png = w / "out" / "psd.png"
    checks["figure_exists"] = png.exists() and png.stat().st_size > 8000
    checks["report_exists"] = (w / "out" / "report.md").exists() and len((w / "out" / "report.md").read_text()) > 200
    try:
        got = json.loads((w / "out" / "results.json").read_text())
        subs = got["per_subject"]
        checks["three_subjects"] = sorted(subs) == ["1", "2", "3"]
        checks["epoch_counts_plausible"] = all(ref["n_epochs_min"] <= int(subs[s]["n_epochs"]) <= ref["n_epochs_max"] for s in subs)
        accs = {s: float(subs[s]["cv_accuracy"]) for s in subs}
        checks["accuracies_in_range"] = all(0.3 <= a <= 1.0 for a in accs.values())
        checks["mean_above_chance"] = float(got["mean_cv_accuracy"]) >= ref["mean_floor"]
        # the reported numbers must be close to an independent reference run of the same pipeline
        checks["matches_reference"] = all(abs(accs[s] - ref["per_subject"][s]) <= 0.15 for s in accs)
        mean_ok = abs(float(got["mean_cv_accuracy"]) - sum(accs.values()) / len(accs)) < 0.02
        checks["mean_is_consistent"] = mean_ok
        detail.append(f"reported {accs}, reference {ref['per_subject']}")
        if (w / "out" / "report.md").exists():
            text = (w / "out" / "report.md").read_text()
            checks["report_quotes_its_numbers"] = sum(f"{a:.2f}"[:4] in text or f"{a * 100:.0f}" in text for a in accs.values()) >= 2
    except Exception as exc:
        for k in ("three_subjects", "epoch_counts_plausible", "accuracies_in_range", "mean_above_chance", "matches_reference", "mean_is_consistent", "report_quotes_its_numbers"):
            checks.setdefault(k, False)
        detail.append(f"results.json unusable ({type(exc).__name__}: {exc})")
    return {"passed": all(checks.values()), "score": round(sum(bool(v) for v in checks.values()) / len(checks), 3), "detail": "; ".join(detail), "checks": checks}


# ------------------------------------------------- own: coding with feedback

LEDGER_SPEC = """
Build a small command-line expense ledger in /app/ledger.py, runnable as
`/opt/tools/bin/python /app/ledger.py <command> ...`, storing data in /app/ledger.json.

Commands and exact output formats:
  add <amount> <category> <description...> [--date YYYY-MM-DD]
      Records an expense. Amount is a decimal with at most 2 places, in EUR, and may be
      negative (a refund). Default date is today. Prints: `added #<id> <amount> <category>`
      where <amount> has exactly two decimals, and <id> starts at 1 and never repeats.
  list [--category C] [--month YYYY-MM]
      Prints one line per matching expense, oldest first, in the form
      `#<id> <YYYY-MM-DD> <amount> <category> <description>`, then a final line
      `total <amount>`. With no matches it prints only `total 0.00`.
  balance [--month YYYY-MM]
      Prints one line per category with a non-zero total, sorted by category name,
      `<category> <amount>`, then `total <amount>`.
  export <path>
      Writes a CSV with header `id,date,amount,category,description` and one row per
      expense, oldest first, quoting fields that contain a comma. Prints `exported <n> rows`.
  delete <id>
      Removes that expense. Prints `deleted #<id>`. Unknown id: print
      `error: no expense #<id>` to stderr and exit with status 2.

Rules: money is exact to the cent (no floating-point drift); invalid amounts or dates print
`error: <reason>` to stderr and exit 2; the data file survives between runs and must remain
valid JSON. Five example tests are in /app/tests_public/test_public.py - run them with
`/opt/tools/bin/python -m pytest /app/tests_public -q`. A larger hidden suite will be run
against the same interface, so follow the formats exactly.
"""

PUBLIC_LEDGER_TESTS = '''
import json, subprocess, sys
from pathlib import Path
PY, LEDGER = "/opt/tools/bin/python", "/app/ledger.py"

def run(*args, cwd="/app"):
    return subprocess.run([PY, LEDGER, *args], capture_output=True, text=True, cwd=cwd)

def setup_function(_):
    Path("/app/ledger.json").unlink(missing_ok=True)

def test_add_prints_id_and_amount():
    out = run("add", "12.5", "food", "lunch", "--date", "2026-01-05")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "added #1 12.50 food"

def test_list_and_total():
    run("add", "12.50", "food", "lunch", "--date", "2026-01-05")
    run("add", "7.25", "travel", "bus ticket", "--date", "2026-01-06")
    out = run("list")
    assert out.stdout.splitlines() == ["#1 2026-01-05 12.50 food lunch", "#2 2026-01-06 7.25 travel bus ticket", "total 19.75"]

def test_balance_sorted_by_category():
    run("add", "5", "travel", "tram", "--date", "2026-02-01")
    run("add", "3", "food", "apple", "--date", "2026-02-02")
    out = run("balance")
    assert out.stdout.splitlines() == ["food 3.00", "travel 5.00", "total 8.00"]

def test_delete_unknown_id_exits_2():
    out = run("delete", "99")
    assert out.returncode == 2 and "no expense #99" in out.stderr

def test_bad_amount_exits_2():
    out = run("add", "abc", "food", "x")
    assert out.returncode == 2 and out.stderr.startswith("error:")
'''


def _setup_ledger(w: Path) -> None:
    _write(w, "tests_public/test_public.py", PUBLIC_LEDGER_TESTS)


def _check_ledger(w: Path) -> dict:
    return pytest_check(w, Path(__file__).parent / "hidden" / "ledger")


# --------------------------------------------- own: data cleaning + analysis

DATA_SPEC = """
Three messy sales exports are in /app/raw (sales_eu.csv, sales_us.csv, sales_apac.csv).
They disagree about formats and contain duplicates and bad rows. Clean them and answer the
questions below, writing /app/out/answers.json.

What you need to handle: different date formats, amounts written with currency symbols,
thousands separators and some in parentheses for negatives, inconsistent region and
category capitalisation and whitespace, exact duplicate rows (same order id) that must be
counted once, rows with a missing or unparseable amount or date (drop them and count them),
and refunds (negative amounts) which stay in the totals.

Write /app/out/answers.json with exactly these keys:
  {"rows_in": <int>,               // data rows read across the three files, before cleaning
   "rows_dropped": <int>,          // rows dropped as unusable (missing/unparseable)
   "duplicates_removed": <int>,    // duplicate order ids removed
   "net_revenue": <number>,        // sum of all valid amounts after cleaning, 2 decimals
   "revenue_by_region": {"<region>": <number>, ...},   // lower-case region names
   "best_month": "<YYYY-MM>",      // month with the highest net revenue
   "top_category": "<name>"}       // category with the highest net revenue, lower-case
Also write /app/clean.py (the script that produced it) and /app/out/clean.csv, the cleaned
rows with header `order_id,date,region,category,amount` sorted by date then order_id.
"""


def _setup_data_cleaning(w: Path) -> None:
    import csv
    import random

    rng = random.Random(20260917)
    (w / "raw").mkdir(parents=True, exist_ok=True)
    (w / "out").mkdir(parents=True, exist_ok=True)
    regions, cats = ["EU", "us", " APAC "], ["Hardware", "software ", "SERVICES", "hardware"]
    base: list[dict] = []
    oid = 1000
    for _ in range(420):
        oid += 1
        base.append({"order_id": f"A{oid}", "region": rng.choice(regions), "category": rng.choice(cats),
                     "month": rng.randint(1, 12), "day": rng.randint(1, 28), "amount": round(rng.uniform(-400, 4000), 2), "bad": ""})
    dups = [dict(base[i]) for i in rng.sample(range(len(base)), 24)]
    bads = []
    for i in range(18):
        oid += 1
        bads.append({"order_id": f"A{oid}", "region": rng.choice(regions), "category": rng.choice(cats), "month": rng.randint(1, 12), "day": rng.randint(1, 28),
                     "amount": round(rng.uniform(10, 900), 2), "bad": "amount" if i % 2 else "date"})
    every = base + dups + bads
    rng.shuffle(every)
    names = ["sales_eu.csv", "sales_us.csv", "sales_apac.csv"]
    buckets: dict[str, list[dict]] = {n: [] for n in names}
    for k, row in enumerate(every):
        buckets[names[k % 3]].append(row)

    def amount_text(value: float, style: int, bad: str) -> str:
        if bad == "amount":
            return "" if style % 2 else "n/a"
        if value < 0:
            return [f"({abs(value):,.2f})", f"-€{abs(value):,.2f}", f"-{abs(value):.2f}"][style]
        return [f"€{value:,.2f}", f"${value:,.2f}", f"{value:.2f}"][style]

    def date_text(row: dict, style: int) -> str:
        if row["bad"] == "date":
            return "31/02/2025" if style else "not a date"
        y, m, d = 2025, row["month"], row["day"]
        return [f"{y}-{m:02d}-{d:02d}", f"{d:02d}/{m:02d}/{y}", f"{m}/{d}/{y}"][style]

    for style, name in enumerate(names):
        with (w / "raw" / name).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["order_id", "date", "region", "category", "amount"])
            for row in buckets[name]:
                writer.writerow([row["order_id"], date_text(row, style), row["region"], row["category"], amount_text(row["amount"], style, row["bad"])])
    # hidden answer key, computed from the same source of truth
    clean = {}
    for row in base:
        clean[row["order_id"]] = {"date": f"2025-{row['month']:02d}-{row['day']:02d}", "region": row["region"].strip().lower(),
                                  "category": row["category"].strip().lower(), "amount": row["amount"]}
    by_region: dict[str, float] = {}
    by_month: dict[str, float] = {}
    by_cat: dict[str, float] = {}
    for r in clean.values():
        by_region[r["region"]] = round(by_region.get(r["region"], 0) + r["amount"], 2)
        by_month[r["date"][:7]] = round(by_month.get(r["date"][:7], 0) + r["amount"], 2)
        by_cat[r["category"]] = round(by_cat.get(r["category"], 0) + r["amount"], 2)
    truth = {"rows_in": len(every), "rows_dropped": len(bads), "duplicates_removed": len(dups),
             "net_revenue": round(sum(r["amount"] for r in clean.values()), 2), "revenue_by_region": by_region,
             "best_month": max(by_month, key=by_month.get), "top_category": max(by_cat, key=by_cat.get), "rows_clean": len(clean)}
    TRUTH.mkdir(parents=True, exist_ok=True)
    (TRUTH / "data_cleaning.json").write_text(json.dumps(truth, indent=1))


def _check_data_cleaning(w: Path) -> dict:
    truth = json.loads((TRUTH / "data_cleaning.json").read_text())
    checks, detail = {}, []
    try:
        got = json.loads((w / "out" / "answers.json").read_text())
    except Exception as exc:
        return {"passed": False, "score": 0.0, "detail": f"answers.json unusable ({type(exc).__name__})", "checks": {"answers_json": False}}
    checks["rows_in"] = int(got.get("rows_in", -1)) == truth["rows_in"]
    checks["rows_dropped"] = int(got.get("rows_dropped", -1)) == truth["rows_dropped"]
    checks["duplicates_removed"] = int(got.get("duplicates_removed", -1)) == truth["duplicates_removed"]
    checks["net_revenue"] = abs(float(got.get("net_revenue", 0)) - truth["net_revenue"]) < 1.0
    got_regions = {str(k).strip().lower(): float(v) for k, v in (got.get("revenue_by_region") or {}).items()}
    checks["revenue_by_region"] = set(got_regions) == set(truth["revenue_by_region"]) and all(abs(got_regions[k] - v) < 1.0 for k, v in truth["revenue_by_region"].items())
    checks["best_month"] = str(got.get("best_month", "")) == truth["best_month"]
    checks["top_category"] = str(got.get("top_category", "")).strip().lower() == truth["top_category"]
    clean_csv = w / "out" / "clean.csv"
    checks["clean_csv"] = False
    if clean_csv.exists():
        lines = clean_csv.read_text().strip().splitlines()
        header_ok = lines[0].replace(" ", "") == "order_id,date,region,category,amount"
        checks["clean_csv"] = header_ok and abs(len(lines) - 1 - truth["rows_clean"]) <= 1
    checks["script_exists"] = (w / "clean.py").exists()
    detail.append(f"truth rows_in={truth['rows_in']} dropped={truth['rows_dropped']} dups={truth['duplicates_removed']} net={truth['net_revenue']}")
    detail.append(f"got rows_in={got.get('rows_in')} dropped={got.get('rows_dropped')} dups={got.get('duplicates_removed')} net={got.get('net_revenue')}")
    return {"passed": all(checks.values()), "score": round(sum(bool(v) for v in checks.values()) / len(checks), 3), "detail": " | ".join(detail), "checks": checks}


# ------------------------------------------------- own: service incident

INCIDENT_SPEC = """
The orders API on this machine stopped serving after a deploy. Everything is in /app:
  - app/server.py      the service (do not rewrite it from scratch; fix what is wrong)
  - app/config.json     its configuration
  - run.sh              how it is started in production
  - logs/orders.log     the log from the failed deploy
  - data/orders.csv     the data it serves

Get the service working again so that, with the service started by `bash /app/run.sh`:
  - GET http://127.0.0.1:8080/health returns HTTP 200 and the JSON {"status": "ok"}
  - GET http://127.0.0.1:8080/orders returns HTTP 200 and a JSON list of all 50 orders,
    each with the keys id, customer, amount and status
  - GET http://127.0.0.1:8080/orders?status=open returns only the open orders

Then write /app/postmortem.md explaining what was actually wrong (there is more than one
fault) and what you changed. Name the configuration key and the port involved.
"""

INCIDENT_SERVER = '''
import csv, json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

CONFIG = os.environ.get("ORDERS_CONFIG", "/app/app/config.json")


def load_config():
    with open(CONFIG) as fh:
        cfg = json.load(fh)
    return cfg["listen_port"], cfg["data_file"]


def load_orders(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append({"id": int(r["id"]), "customer": r["customer"], "amount": float(r["amount"]), "status": r["status"]})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/health":
            self._send(200, {"status": "ok"})
            return
        if url.path == "/orders":
            orders = load_orders(DATA_FILE)
            want = parse_qs(url.query).get("status", [None])[0]
            if want:
                orders = [o for o in orders if o["status"] == want]
            self._send(200, orders)
            return
        self._send(404, {"error": "not found"})


if __name__ == "__main__":
    PORT, DATA_FILE = load_config()
    print(f"serving orders on {PORT} from {DATA_FILE}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''

INCIDENT_LOG = """
2026-09-14 09:12:01 INFO  deploy 41 starting
2026-09-14 09:12:01 INFO  loading config from /app/app/config.json
2026-09-14 09:12:01 ERROR Traceback (most recent call last):
2026-09-14 09:12:01 ERROR   File "/app/app/server.py", line 41, in <module>
2026-09-14 09:12:01 ERROR     PORT, DATA_FILE = load_config()
2026-09-14 09:12:01 ERROR   File "/app/app/server.py", line 10, in load_config
2026-09-14 09:12:01 ERROR     return cfg["listen_port"], cfg["data_file"]
2026-09-14 09:12:01 ERROR KeyError: 'listen_port'
2026-09-14 09:12:02 INFO  deploy 41 exited with status 1
2026-09-14 09:13:10 INFO  operator restarted service manually
2026-09-14 09:13:10 ERROR FileNotFoundError: [Errno 2] No such file or directory: '/app/data/orders_2026.csv'
2026-09-14 09:14:44 INFO  health check on 127.0.0.1:8080 failed: connection refused
"""


def _setup_incident(w: Path) -> None:
    import csv
    import random

    rng = random.Random(4242)
    _write(w, "app/server.py", INCIDENT_SERVER)
    # fault 1: the config was renamed to "port"; fault 2: it points at a data file that does not exist
    _write(w, "app/config.json", json.dumps({"port": 8080, "data_file": "/app/data/orders_2026.csv", "log_level": "info"}, indent=1))
    _write(w, "logs/orders.log", INCIDENT_LOG)
    _write(w, "run.sh", "#!/bin/bash\n# production launcher\nexec /opt/tools/bin/python /app/app/server.py\n")
    (w / "data").mkdir(parents=True, exist_ok=True)
    with (w / "data" / "orders.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "customer", "amount", "status"])
        for i in range(1, 51):
            writer.writerow([i, f"customer-{i:02d}", f"{rng.uniform(10, 900):.2f}", rng.choice(["open", "shipped", "closed"])])


def _check_incident(w: Path) -> dict:
    """Start the service the way production does, then probe it from inside the sandbox."""
    checks, detail = {}, []
    probe = """
import json, subprocess, time, urllib.request
p = subprocess.Popen(["/bin/bash", "/app/run.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
out = {}
try:
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as r:
                out["health_code"], out["health_body"] = r.status, r.read().decode()[:200]
            break
        except Exception:
            if p.poll() is not None:
                out["exited"] = p.stderr.read()[-400:]
                break
            time.sleep(0.5)
    for name, url in (("orders", "http://127.0.0.1:8080/orders"), ("open", "http://127.0.0.1:8080/orders?status=open")):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                body = json.loads(r.read().decode())
            out[name + "_code"] = r.status
            out[name + "_n"] = len(body) if isinstance(body, list) else None
            out[name + "_keys_ok"] = bool(body) and isinstance(body, list) and all({"id", "customer", "amount", "status"} <= set(o) for o in body)
            out[name + "_statuses"] = sorted({o.get("status") for o in body}) if isinstance(body, list) else None
        except Exception as exc:
            out[name + "_error"] = f"{type(exc).__name__}: {exc}"
finally:
    p.kill()
print(json.dumps(out))
"""
    _write(w, ".probe_service.py", probe)
    res = sandbox_run(w, [PY, "/app/.probe_service.py"], timeout=180)
    (w / ".probe_service.py").unlink(missing_ok=True)
    got = {}
    try:
        got = json.loads(res.stdout.strip().splitlines()[-1])
    except Exception:
        detail.append(f"probe failed: {(res.stderr or res.stdout)[-300:]}")
    checks["health_200"] = got.get("health_code") == 200 and json.loads(got.get("health_body", "{}") or "{}").get("status") == "ok"
    checks["orders_200"] = got.get("orders_code") == 200 and got.get("orders_n") == 50
    checks["orders_shape"] = bool(got.get("orders_keys_ok"))
    expected_open = sum(1 for ln in (w / "data" / "orders.csv").read_text().splitlines()[1:] if ln.strip().endswith("open"))
    checks["status_filter"] = got.get("open_code") == 200 and got.get("open_n") == expected_open and got.get("open_statuses") == ["open"]
    pm = w / "postmortem.md"
    text = pm.read_text() if pm.exists() else ""
    checks["postmortem"] = len(text) > 150 and ("listen_port" in text or "port" in text) and "8080" in text
    if got.get("exited"):
        detail.append("service exited: " + got["exited"][-200:])
    detail.append(f"health={got.get('health_code')} orders={got.get('orders_code')} n={got.get('orders_n')} open={got.get('open_n')} (expected {expected_open})")
    return {"passed": all(checks.values()), "score": round(sum(bool(v) for v in checks.values()) / len(checks), 3), "detail": " | ".join(detail), "checks": checks}


# ------------------------------------------------------------------ registry


def all_tasks() -> list[Task]:
    return [
        Task("own/cad_bracket", "own", "hard", "heldout", CAD_SPEC.strip(), _setup_cad, _check_cad, minutes=45),
        Task("own/eeg_motor_imagery", "own", "hard", "heldout", EEG_SPEC.strip(), _setup_eeg, _check_eeg, minutes=45),
        Task("own/coding_ledger", "own", "medium", "tune", LEDGER_SPEC.strip(), _setup_ledger, _check_ledger, minutes=45),
        Task("own/data_cleaning", "own", "medium", "tune", DATA_SPEC.strip(), _setup_data_cleaning, _check_data_cleaning, minutes=45),
        Task("own/service_incident", "own", "medium", "heldout", INCIDENT_SPEC.strip(), _setup_incident, _check_incident, minutes=45),
        tb2_task("log-summary-date-ranges", setup=_setup_log_summary),
        tb2_task("sqlite-db-truncate", setup=_setup_sqlite_trunc),
        tb2_task("cancel-async-tasks", setup=_setup_cancel_async, copy_into_app=["test.py"]),
        tb2_task("git-leak-recovery", setup=_setup_git_leak),
    ]


def by_id(ids: list[str] | None = None) -> list[Task]:
    tasks = all_tasks()
    return [t for t in tasks if not ids or t.id in ids or t.id.split("/")[-1] in ids]
