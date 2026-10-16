"""Bundle every declared trial, including failures, with offline sensor audits."""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "examples/robotics/transfer"))
from audit import audit_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--prefix", default="")
parser.add_argument("--summary-name", default="results.json")
parser.add_argument("--demonstration", default="sensor-expert-v3")
parser.add_argument(
    "--cases", nargs="+", default=["nominal", "shift", "height", "frame"]
)
args = parser.parse_args()
base = args.runs
output = args.output
output.mkdir(parents=True, exist_ok=True)
names = [args.demonstration] + [
    f"{args.prefix}{case}-{mode}"
    for case in args.cases
    for mode in ["fixed", "context"]
]
rows = []
for name in names:
    src = base / name
    if not (src / "probe.json").exists():
        continue
    dst = output / "runs" / name
    dst.mkdir(parents=True, exist_ok=True)
    for fname in [
        "probe.json",
        "probe.npz",
        "scene.xml",
        "context.json",
        "context-policy.json",
        "sensor-contract.json",
        "sensor-first-frame.png",
        "sensor-evidence.npz",
    ]:
        if (src / fname).exists():
            shutil.copy2(src / fname, dst / fname)
    for file in src.iterdir():
        if file.name.startswith("sensor") and file.is_file():
            shutil.copy2(file, dst / file.name)
    for fname in ["records.json"]:
        with gzip.GzipFile(
            filename=str(dst / (fname + ".gz")), mode="wb", mtime=0
        ) as stream:
            stream.write((src / fname).read_bytes())
    audit = audit_run(
        src, base / args.demonstration / "context.json", base / args.demonstration
    )
    (dst / "sensor-audit.json").write_text(json.dumps(audit, indent=2))
    budget = base / (name + "-budget")
    if (budget / "resources.json").exists():
        shutil.copy2(budget / "resources.json", dst / "resources.json")
    if (budget / "stdout.log").exists():
        shutil.copy2(budget / "stdout.log", dst / "stdout.log")
    result = json.loads((src / "probe.json").read_text())
    result["run"] = name
    result["sensor_audit"] = audit
    rows.append(result)
summary = {
    "protocol": "Paired deterministic simulation conditions; all declared outcomes retained. Same frozen RL walker and one frozen commanded trajectory.",
    "declared_conditions": args.cases,
    "variant": "Controller settings recorded in each run probe",
    "baseline": "Same Unitree RL walker plus fixed world-coordinate command replay",
    "treatment": "Same Unitree RL walker plus authored ALG.2 attention retargeting from sensor-estimated pickup anchor",
    "limitations": [
        "Commanded trajectory context, not video-only learning or learned ICL",
        "One episode per condition, no statistical generalization claim",
        "Common camera-frame offset is not relative hand-eye calibration",
        "Decorative warehouse scene, not demonstrated warehouse logistics",
    ],
    "runs": rows,
}
(output / args.summary_name).write_text(json.dumps(summary, indent=2))
print([(r["run"], r["success"], r["max_lift"]) for r in rows])
