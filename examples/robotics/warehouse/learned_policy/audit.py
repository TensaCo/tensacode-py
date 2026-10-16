"""Offline recorder integrity checks; saved simulator state never enters policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import load_episode, sha256


def audit_run(directory):
    import mujoco

    records = json.loads((directory / "records.json").read_text())
    poses = np.load(directory / "probe.npz")["qpos"]
    model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    reports = []
    for path in sorted(directory.glob("r*/bc-episode.json.gz")):
        x, y, document = load_episode(path)
        robot = document["robot_id"]
        joints = [
            i
            for i in range(model.njnt)
            if model.joint(i).name.startswith(robot + "/")
            and int(model.jnt_type[i])
            in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        ]
        addresses = sorted(int(model.jnt_qposadr[i]) for i in joints)
        rows = document["steps"]
        if len(rows) != len(records) or len(poses) != len(records):
            raise ValueError("Recorder, trace and saved poses have different lengths")
        encoders = np.asarray([row["observation"]["joint_position"] for row in rows])
        encoder_error = float(np.abs(encoders - poses[:, addresses]).max())
        grip_error = max(
            abs(row["action"]["grip"] - record["robots"][robot]["grip_command"])
            for row, record in zip(rows, records)
        )
        sensor_mismatches = sum(
            row["observation"]["sensor_state"]
            != record["robots"][robot]["sensor_state"]
            for row, record in zip(rows, records)
        )
        time_error = max(
            abs(row["t"] - record["t"]) for row, record in zip(rows, records)
        )
        targets_match = all(
            row["action"]["joint_target"][22:29] == row["action"]["arm_target"]
            for row in rows
        )
        valid = (
            encoder_error == grip_error == sensor_mismatches == time_error == 0
            and targets_match
        )
        reports.append(
            {
                "case": directory.name,
                "robot": robot,
                "episode_sha256": sha256(path),
                "steps": len(rows),
                "feature_count": x.shape[1],
                "action_count": y.shape[1],
                "max_encoder_vs_saved_qpos_error": encoder_error,
                "max_grip_record_error": grip_error,
                "sensor_record_mismatches": sensor_mismatches,
                "max_timestamp_error": time_error,
                "joint_target_arm_consistent": targets_match,
                "passed": valid,
            }
        )
    if not reports:
        raise ValueError("No collected policy episodes")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "offline recorder consistency, not policy performance or perception truth accuracy",
        "runs": [row for directory in args.runs for row in audit_run(directory)],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(row["passed"] for row in report["runs"]):
        raise SystemExit(1)
