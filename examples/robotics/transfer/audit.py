"""Offline scoring of sensor estimates against simulator labels, never a controller.

The simulator pose columns are intentionally consumed here only after a rollout
has completed. This tool cannot select actions, update context, or rerun physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(errors: list[np.ndarray]) -> dict:
    if not errors:
        return {"samples": 0}
    xyz = np.asarray(errors, dtype=float)
    norm = np.linalg.norm(xyz, axis=1)
    return {
        "samples": len(norm),
        "mean_m": float(norm.mean()),
        "p95_m": float(np.quantile(norm, 0.95)),
        "max_m": float(norm.max()),
        "mean_signed_xyz_m": xyz.mean(axis=0).tolist(),
    }


def audit_records(records: list[dict]) -> dict:
    root, object_all, pickup, hand, destination = [], [], [], [], []
    first_reach = None
    root_updates = camera_updates = 0
    for row in records:
        sensor = row.get("sensor_state", {})
        root_estimate = sensor.get("root_position")
        object_estimate = sensor.get("object_position")
        destination_estimate = sensor.get("destination_position")
        if (
            destination_estimate is not None
            and len(row.get("receiving_surface_xyz", [])) == 3
        ):
            destination.append(
                np.asarray(destination_estimate) - row["receiving_surface_xyz"]
            )
        if root_estimate is not None and len(row.get("base", [])) == 3:
            root.append(np.asarray(root_estimate) - row["base"])
        if object_estimate is not None and len(row.get("object", [])) == 3:
            error = np.asarray(object_estimate) - row["object"]
            object_all.append(error)
            if row.get("stage") in ("walk", "settle"):
                pickup.append(error)
        if row.get("desired_hand") and len(row.get("hand", [])) == 3:
            hand.append(np.asarray(row["hand"]) - row["desired_hand"])
        if first_reach is None and row.get("stage") == "reach":
            first_reach = row
        root_updates += bool(sensor.get("root_camera_valid", False))
        camera_updates += bool(sensor.get("camera_valid", False))
    anchor_error = None
    if first_reach is not None and first_reach.get("target_anchor"):
        # The controller deliberately aims 9 mm above the estimated cube center.
        anchor_error = float(
            np.linalg.norm(
                np.asarray(first_reach["target_anchor"])
                - np.asarray(first_reach["object"])
                - [0.0, 0.0, 0.009]
            )
        )
    count = len(records)
    return {
        "role": "Offline scoring only; simulator labels never feed control or context",
        "samples": count,
        "root_localization": _summary(root),
        "object_localization_before_grasp": _summary(pickup),
        "object_localization_all_phases": _summary(object_all),
        "hand_command_tracking": _summary(hand),
        "destination_localization": _summary(destination),
        "frozen_pickup_anchor_error_m": anchor_error,
        "root_optical_valid_fraction": root_updates / count if count else None,
        "object_optical_valid_fraction": camera_updates / count if count else None,
        "interpretation": [
            "Object goal is frozen at reach onset; fast lifting can make the object posterior stale",
            "Hand tracking error includes transient commanded motion, not just final placement accuracy",
            "Valid fractions count control samples, not independent camera frames",
            "These measurements do not establish learned visual ICL or superiority over a VLA",
        ],
    }


def audit_run(
    directory: Path, context: Path | None = None, demonstration: Path | None = None
) -> dict:
    records_path = directory / "records.json"
    report = audit_records(json.loads(records_path.read_text()))
    report["records_sha256"] = _sha(records_path)
    report["auditor_sha256"] = _sha(Path(__file__))
    probe_path = directory / "probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    declared = probe.get("source_context_sha256")
    report["declared_source_context_sha256"] = declared
    report["context_hash_verified"] = False
    report["demonstration_hash_verified"] = False
    if context is not None:
        digest = _sha(context)
        if declared is not None and digest != declared:
            raise ValueError(
                "Context file does not match the rollout's recorded source hash"
            )
        report["source_context_sha256"] = digest
        report["context_hash_verified"] = declared is not None
        payload = json.loads(context.read_text())
        if demonstration is not None:
            source_records = demonstration / "records.json"
            if _sha(source_records) != payload["source_sha256"]:
                raise ValueError(
                    "Demonstration records do not match the frozen context source hash"
                )
            report["demonstration_records_sha256"] = payload["source_sha256"]
            report["demonstration_hash_verified"] = True
    elif demonstration is not None:
        raise ValueError("Demonstration verification requires --context")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--demonstration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_run(args.directory, args.context, args.demonstration)
    output = args.output or args.directory / "sensor-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
