"""Bundle current G1 controller sessions for the diagnosis interface."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import statistics
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


def package_handling(source: Path, output: Path):
    """Retain every completed handling attempt, including development failures."""
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    declared_cases = {
        name
        for protocol in source.rglob("protocol.json")
        for name in json.loads(protocol.read_text()).get("cases", {})
    }
    for probe_path in sorted(source.rglob("probe.json")):
        run = probe_path.parent
        identifier = "-".join(run.relative_to(source).parts)
        probe = json.loads((run / "probe.json").read_text())
        if probe.get("mode") != "handling":
            continue
        target = output / "runs" / identifier
        target.mkdir(parents=True, exist_ok=True)
        payload = (run / "records.json").read_bytes()
        records = json.loads(payload)
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
            if (run / filename).is_file():
                shutil.copy2(run / filename, target / filename)
        budget = run.parent / (run.name + "-budget")
        for filename in ("resources.json", "stdout.log"):
            if (budget / filename).is_file():
                shutil.copy2(budget / filename, target / filename)
        (target / "sensor-audit.json").write_text(
            json.dumps(audit_run(run), indent=2) + "\n"
        )
        carried = [
            r["controller_state"]
            for r in records
            if (r.get("controller_state") or {}).get("phase") == "transport"
        ]
        tracking = [r["tracking_error_m"] for r in carried if "tracking_error_m" in r]
        masses = [
            r["payload_mass_estimate_kg"]
            for r in carried[len(carried) // 2 :]
            if r.get("payload_mass_estimate_kg") is not None
        ]
        row = {
            "id": identifier,
            "role": "declared_check" if run.name in declared_cases else "development",
            "trace": f"runs/{identifier}/records.json.gz",
            "source_snapshot_captured": (target / "source.zip").is_file(),
            "transport_tracking_error_median_m": statistics.median(tracking)
            if tracking
            else None,
            "late_transport_tracking_error_median_m": statistics.median(
                r["tracking_error_m"] for r in carried[len(carried) // 2 :]
            )
            if carried
            else None,
            "late_transport_orientation_error_median_rad": statistics.median(
                r["orientation_error_rad"] for r in carried[len(carried) // 2 :]
            )
            if carried
            else None,
            "transport_payload_mass_estimate_kg": statistics.median(masses)
            if masses
            else None,
        }
        for key in (
            "success",
            "seconds",
            "max_lift",
            "held_above_10cm_seconds",
            "walking_distance_m",
            "bilateral_contact_seconds",
            "mode",
            "placement_settled_seconds",
            "placement_xy_error_m",
            "hand_clearance_m",
            "controller_phase",
            "controller_failed",
            "object_mass_kg",
            "load_compensation",
            "receiving_surface_xyz",
            "source_sha256",
        ):
            if key in probe:
                row[key] = probe[key]
        rows.append(row)
    for filename in ("protocol.json", "execution.json"):
        for protocol in source.rglob(filename):
            name = "-".join(protocol.relative_to(source).parts)
            shutil.copy2(protocol, output / name)
    report = {
        "complete": bool(rows),
        "runs": rows,
        "scope": "Deterministic G1 walk, camera-localized pickup and stationary receiver, friction grasp, carry, tactile support transfer, release and withdrawal. All completed development attempts retained; no statistical generalization claim.",
        "architecture": "RGB-D object/destination beliefs + proprioception + local fingertip forces -> guarded task stages -> bounded IK and tactile load compensation -> arm/gripper servos alongside a frozen Unitree RL walker.",
        "priors": [
            "Calibrated camera and initial root pose; known robot kinematics and pelvis fiducial",
            "Upright orange 5cm cube and stationary axis-aligned magenta 20cm receiving surface",
        ],
        "limitations": [
            "Development conditions chosen within the fixed-wrist arm workspace",
            "Color/geometry perception; no learned visual ICL or general object recognition",
            "Frozen pickup and destination anchors; no moving-target tracking",
            "No matched VLA comparison, hardware result, or whole-body collision certification",
        ],
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
    parser.add_argument("--task", choices=("planning", "handling"), default="planning")
    args = parser.parse_args()
    pack = package_handling if args.task == "handling" else package
    report = pack(args.runs, args.output)
    print(json.dumps({"complete": report["complete"], "runs": len(report["runs"])}))


if __name__ == "__main__":
    main()
