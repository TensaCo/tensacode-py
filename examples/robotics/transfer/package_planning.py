"""Bundle current G1 controller sessions for the diagnosis interface."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from audit import audit_run
from run_planning_suite import CASES

DEVELOPMENT = [
    "expert-height",
    "planned-height-v1",
    "planned-nominal-v1",
    "height02",
    "height06",
    "shift",
    "frame",
    "height04-alg0",
]


def package(source, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    final = source / "final" if (source / "final").is_dir() else source
    for role, root, names in [
        ("development", source, DEVELOPMENT if final != source else []),
        ("regression", final, list(CASES)),
    ]:
        for name in names:
            run = root / name
            if not (run / "probe.json").exists():
                continue
            identifier = (
                (
                    "development-height-v1"
                    if name == "planned-height-v1"
                    else "development-" + name
                )
                if role == "development"
                else name
            )
            target = output / "runs" / identifier
            target.mkdir(parents=True, exist_ok=True)
            probe = json.loads((run / "probe.json").read_text())
            audit = audit_run(run)
            (target / "sensor-audit.json").write_text(
                json.dumps(audit, indent=2) + "\n"
            )
            payload = (run / "records.json").read_bytes()
            (target / "records.json.gz").write_bytes(gzip.compress(payload, mtime=0))
            for filename in (
                "probe.json",
                "probe.npz",
                "scene.xml",
                "events.json",
                "source.zip",
                "sensor-rgb.png",
                "sensor-depth.npz",
                "sensor-contract.json",
                "controller-algorithm.json",
            ):
                if (run / filename).exists():
                    shutil.copy2(run / filename, target / filename)
            resources = root / (name + "-budget") / "resources.json"
            if resources.exists():
                shutil.copy2(resources, target / "resources.json")
            records = json.loads(payload)
            state = [
                v["controller_state"] for v in records if v.get("controller_state")
            ]
            rows.append(
                {
                    "id": identifier,
                    "role": "first_combined_check" if name == "combined" else role,
                    "success": probe["success"],
                    "max_lift": probe["max_lift"],
                    "held_above_10cm_seconds": probe["held_above_10cm_seconds"],
                    "seconds": probe["seconds"],
                    "height_shift": probe.get("height_shift", 0),
                    "walking_distance_m": probe["walking_distance_m"],
                    "bilateral_contact_seconds": probe["bilateral_contact_seconds"],
                    "mode": probe["mode"],
                    "ik_backend": probe.get(
                        "ik_backend", "scipy" if probe.get("planned") else None
                    ),
                    "solver_fallback_count": state[-1].get("solver_fallback_count", 0)
                    if state
                    else 0,
                    "trace": "runs/" + identifier + "/records.json.gz",
                    "source_snapshot_captured": (run / "source.zip").is_file(),
                    "events": json.loads((run / "events.json").read_text())
                    if (run / "events.json").exists()
                    else [],
                }
            )
    protocol = final / "protocol.json"
    if protocol.exists():
        shutil.copy2(protocol, output / "protocol.json")
    interrupted = source / "height04-alg2-budget"
    if interrupted.exists():
        target = output / "runtime-limit"
        target.mkdir(exist_ok=True)
        for filename in ("stdout.log", "resources.json"):
            if (interrupted / filename).exists():
                shutil.copy2(interrupted / filename, target / filename)
    report = {
        "complete": all(any(r["id"] == name for r in rows) for name in CASES),
        "runs": rows,
        "scope": "Deterministic G1 upright-cube pickup. Six final conditions are development/regression after tuning; combined is a single fresh combination. No trained visual ICL, warehouse-task execution, or matched GR00T comparison.",
        "architecture": "RGB-D belief and proprioception -> guarded geometric waypoints -> continuous bounded IK with sampled tool clearance -> arm/gripper servos, alongside a frozen Unitree RL walker.",
        "priors": [
            "Calibrated camera and initial root pose",
            "Known robot kinematics and pelvis fiducial",
            "Upright orange 5cm cube",
            "Square support footprint 20cm across; top inferred from observed cube height",
        ],
        "limitations": [
            "Pickup anchor freezes at reach onset; freshness gates descent but is not visual servoing during lift",
            "Clearance checks four intermediate command poses for the tool only; not whole-body or continuous collision certification",
            "Unexpected contact aborts and attempts an open-gripper retraction; no automatic retry",
            "ALG0 compiled projected-QP run uses explicit convergence-gated SciPy fallbacks",
            "Full128-iteration ALG2 physical run was interrupted after125s due dense interpreter reads; no completed ALG2 QP rollout claimed",
        ],
        "runtime_limit": {
            "backend": "alg2",
            "status": "interrupted_for_latency",
            "evidence": "runtime-limit/stdout.log",
        },
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "bundle-sha256.json"
    }
    (output / "bundle-sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = package(args.runs, args.output)
    print(json.dumps({"complete": report["complete"], "runs": len(report["runs"])}))


if __name__ == "__main__":
    main()
