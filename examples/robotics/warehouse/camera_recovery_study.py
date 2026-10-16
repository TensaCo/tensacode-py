"""Audit the declared development-only camera/recovery factorial without running it."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from examples.robotics.warehouse.validate_v4 import validate


def summarize(protocol_path: Path):
    protocol = json.loads(protocol_path.read_text())
    code = Path(protocol["source_directory"])
    archive = code.parent / "source.zip"
    if (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        != protocol["source_zip_sha256"]
    ):
        raise ValueError("Frozen source archive differs from declared protocol")
    rows = []
    source_hashes = None
    environment = None
    initial = {}
    for case in protocol["cases"]:
        path = Path(case["run_path"])
        row = dict(case)
        if not (path / "probe.json").is_file():
            row.update(status="pending", successful_robots=None)
            rows.append(row)
            continue
        probe_bytes = (path / "probe.json").read_bytes()
        probe = json.loads(probe_bytes)
        scenario_path = code / case["scenario_path"]
        if (
            hashlib.sha256(scenario_path.read_bytes()).hexdigest()
            != case["scenario_sha256"]
        ):
            raise ValueError(f"Scenario source changed: {case['id']}")
        expected = {
            "robots": protocol["robots"],
            "policy": "authored",
            "grasp_recovery": case["grasp_recovery"],
            "offscreen_samples": case["offscreen_samples"],
            "full_horizon": True,
            "requested_duration_s": protocol["requested_duration_s"],
            "arm_slew_rad": None,
            "scenario": json.loads(scenario_path.read_text()),
        }
        for key, value in expected.items():
            if probe.get(key) != value:
                raise ValueError(f"Condition mismatch {case['id']}: {key}")
        if (
            probe["render_backend"] != "egl"
            or protocol["driver"] not in probe["opengl"]["version"]
        ):
            raise ValueError("Matched peer EGL driver is required")
        if probe["opengl"]["vendor"] != protocol["opengl_vendor"]:
            raise ValueError("Matched GL vendor is required")
        current_environment = {
            "opengl": probe["opengl"],
            "mujoco_version": probe["mujoco_version"],
        }
        if environment is None:
            environment = current_environment
        elif environment != current_environment:
            raise ValueError("MuJoCo/OpenGL runtime changed between factorial cells")
        if source_hashes is None:
            source_hashes = probe["source_sha256"]
            with zipfile.ZipFile(archive) as frozen:
                for name, digest in source_hashes.items():
                    source_bytes = frozen.read("examples/robotics/" + name)
                    if hashlib.sha256(source_bytes).hexdigest() != digest:
                        raise ValueError(f"Runtime source mismatch: {name}")
        elif probe["source_sha256"] != source_hashes:
            raise ValueError("Factorial runtime sources differ")
        previous = initial.setdefault(case["condition"], probe["initial_qpos_sha256"])
        if previous != probe["initial_qpos_sha256"]:
            raise ValueError("Factorial initial physics state differs within condition")
        if (
            "existing_probe_sha256" in case
            and hashlib.sha256(probe_bytes).hexdigest() != case["existing_probe_sha256"]
        ):
            raise ValueError("Reused evidence changed")
        audit = validate(path)
        if (
            "existing_records_sha256" in case
            and audit["records_sha256"] != case["existing_records_sha256"]
        ):
            raise ValueError("Reused trajectory changed")
        row.update(
            status="complete",
            successful_robots=sum(
                s["strict_current_settle_success"] for s in audit["robots"].values()
            ),
            robot_count=len(audit["robots"]),
            validation=audit,
        )
        rows.append(row)
    contrasts = {}
    for condition in sorted({r["condition"] for r in rows}):
        cells = {
            (r["offscreen_samples"], r["grasp_recovery"]): r
            for r in rows
            if r["condition"] == condition
        }
        if len(cells) != 4 or any(r["status"] != "complete" for r in cells.values()):
            contrasts[condition] = None
            continue
        counts = {key: row["successful_robots"] for key, row in cells.items()}
        contrasts[condition] = {
            "unit": "difference in successful robots, two robots per scene; not independent replicate statistics",
            "camera0_minus4_recovery_off": counts[0, False] - counts[4, False],
            "camera0_minus4_recovery_on": counts[0, True] - counts[4, True],
            "recovery_on_minusoff_camera4": counts[4, True] - counts[4, False],
            "recovery_on_minusoff_camera0": counts[0, True] - counts[0, False],
        }
    return {
        "schema": "tensorcode.warehouse.camera_recovery_factorial.results.v1",
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "source_zip_sha256": protocol["source_zip_sha256"],
        "environment": environment,
        "scope": protocol["scope"],
        "replication": protocol["replication"],
        "complete": all(r["status"] == "complete" for r in rows),
        "completed_cells": sum(r["status"] == "complete" for r in rows),
        "within_condition_contrasts": contrasts,
        "cells": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "completed_cells": result["completed_cells"],
            }
        )
    )


if __name__ == "__main__":
    main()
