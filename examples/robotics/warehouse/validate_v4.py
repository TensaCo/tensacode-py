"""Independent, phase-free physical scoring of v4 warehouse evidence.

This reads evaluation records; nothing here supplies inputs to a controller.
Completion declarations are reported separately from physical placement.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, tolerance=1e-8):
    require(np.isfinite(actual) and abs(actual - expected) <= tolerance, name)


def physical_scores(records, robot_id):
    streak = longest = 0
    ever = False
    first_success = None
    minimum_height = float("inf")
    final_success = False
    for record in records:
        robot = record["robots"][robot_id]
        e = robot["evaluation"]
        position = np.asarray(e["object_position"], dtype=float)
        surface = np.asarray(e["receiving_surface_xyz"], dtype=float)
        require(position.shape == surface.shape == (3,), "Invalid position shape")
        require(np.isfinite(np.r_[position, surface]).all(), "Nonfinite geometry")
        speed, clearance, grip = (
            float(e["object_speed_m_s"]),
            float(e["hand_clearance_m"]),
            float(robot["grip_command"]),
        )
        require(
            np.isfinite([speed, clearance, grip, e["minimum_root_height"]]).all()
            and speed >= 0
            and clearance >= 0,
            "Invalid physical measurement",
        )
        close(
            e["placement_xy_error_m"],
            float(np.linalg.norm(position[:2] - surface[:2])),
            "Placement error differs from geometry",
        )
        require(
            e["minimum_root_height"] <= minimum_height + 1e-9,
            "Minimum root height increases",
        )
        minimum_height = e["minimum_root_height"]
        contacts = set(e["contacts"])
        valid = (
            robot_id + "/receiving_marker" in contacts
            and np.max(np.abs(position[:2] - surface[:2])) <= 0.065
            and abs(position[2] - surface[2] - 0.025) <= 0.01
            and speed < 0.03
            and not contacts.intersection({robot_id + "/pad_a", robot_id + "/pad_b"})
            and grip <= 1e-5
            and clearance > 0.1
        )
        streak = streak + 1 if valid else 0
        longest = max(longest, streak)
        close(e["streak"], streak * 0.02, "Invalid contiguous settling time")
        close(e["settled_seconds"], longest * 0.02, "Invalid longest settling time")
        final_success = bool(valid and longest * 0.02 >= 1 and minimum_height > 0.65)
        require(e["physical_success"] == final_success, "Incorrect physical success")
        require(e["success"] == final_success, "Incorrect physical success alias")
        if final_success and not ever:
            first_success = record["t"]
        ever |= final_success
    final = records[-1]["robots"][robot_id]
    return {
        "physical_success": final_success,
        "strict_current_settle_success": bool(
            streak * 0.02 >= 1 and minimum_height > 0.65
        ),
        "current_settled_seconds": streak * 0.02,
        "ever_physical_success": ever,
        "first_physical_success_s": first_success,
        "settled_seconds": longest * 0.02,
        "placement_xy_error_m": final["evaluation"]["placement_xy_error_m"],
        "hand_clearance_m": final["evaluation"]["hand_clearance_m"],
        "final_phase": final["phase"],
        "authored_phase_completed": final["phase"] == "complete"
        and not final["controller_state"].get("failed", False),
        "reported_controller_completion": final["evaluation"][
            "controller_declared_completion"
        ],
    }


def validate(directory: Path):
    probe = json.loads((directory / "probe.json").read_text())
    require(probe.get("schema") == "tensorcode.warehouse.physics.v4", "Not v4 evidence")
    compressed = directory / "records.json.gz"
    raw = (
        gzip.decompress(compressed.read_bytes())
        if compressed.exists()
        else (directory / "records.json").read_bytes()
    )
    records = json.loads(raw)
    poses = np.load(directory / "probe.npz", allow_pickle=False)["qpos"]
    require(len(records) == len(poses) > 0, "Record/pose count mismatch")
    require(np.isfinite(poses).all(), "Nonfinite replay pose")
    times = np.asarray([r["t"] for r in records])
    require(
        np.isfinite(times).all()
        and times[0] >= 0
        and np.allclose(np.diff(times), 0.02, atol=1e-8),
        "Unexpected control sampling times",
    )
    require(
        -1e-8 <= probe["seconds"] - times[-1] <= 0.020001,
        "Recorded duration differs from samples",
    )
    require(
        hashlib.sha256(poses[0].tobytes()).hexdigest() == probe["initial_qpos_sha256"],
        "Initial physical state hash mismatch",
    )
    ids = set(records[0]["robots"])
    require(len(ids) == probe["robots"], "Robot count mismatch")
    require(all(set(row["robots"]) == ids for row in records), "Robot IDs changed")
    with zipfile.ZipFile(directory / "source.zip") as archive:
        for name, expected in probe["source_sha256"].items():
            require(
                hashlib.sha256(archive.read(name)).hexdigest() == expected,
                f"Source hash mismatch: {name}",
            )
    scores = {name: physical_scores(records, name) for name in sorted(ids)}
    for name, score in scores.items():
        require(
            probe["evaluation"][name] == records[-1]["robots"][name]["evaluation"],
            "Probe and final evaluator record differ",
        )
        if "ever_physical_success" in probe:
            require(
                probe["ever_physical_success"][name] == score["ever_physical_success"],
                "Incorrect ever-qualified placement claim",
            )
    tasks = {task["id"]: task for task in probe["tasks"]}
    require(len(tasks) == len(probe["tasks"]), "Duplicate task IDs")
    physical = bool(tasks and all(s["physical_success"] for s in scores.values()))
    require(probe["physical_success"] == physical, "Incorrect aggregate physical score")
    events = json.loads((directory / "events.json").read_text())
    assigned = {e["task_id"]: e for e in events if e["type"] == "assigned"}
    finished = {e["task_id"]: e for e in events if e["type"] == "task_finished"}
    outcomes = {o["task_id"]: o for o in probe["outcomes"]}
    intervals = []
    for task_id, event in assigned.items():
        require(task_id in tasks, "Unknown dispatched task")
        for field in ("object_id", "destination_id"):
            require(
                event[field] == tasks[task_id][field], "Dispatched reference changed"
            )
        for dependency in tasks[task_id]["dependencies"]:
            require(
                dependency in finished
                and finished[dependency]["t"] <= event["t"] + 1e-9
                and outcomes[dependency]["success"],
                "Dependency started without controller completion",
            )
        intervals.append(
            (event["t"], finished.get(task_id, {}).get("t", float(times[-1])), event)
        )
    for index, (start, end, event) in enumerate(intervals):
        for start2, end2, event2 in intervals[index + 1 :]:
            if min(end, end2) > max(start, start2) + 1e-9:
                require(event["robot_id"] != event2["robot_id"], "Robot double-booked")
                require(
                    not set(event["resources"]).intersection(event2["resources"]),
                    "Concurrent resource reservation conflict",
                )
    controller_success = bool(
        tasks
        and len(outcomes) == len(tasks)
        and all(o["success"] for o in outcomes.values())
    )
    expected = physical and (
        probe["policy"] in {"learned_bc", "learned_diffusion"} or controller_success
    )
    require(probe["success"] == expected, "Incorrect legacy aggregate success")
    if probe.get("termination_reason") == "horizon":
        close(
            probe["seconds"], probe["requested_duration_s"], "Horizon mismatch", 0.0021
        )
    return {
        "validated": True,
        "schema": probe["schema"],
        "physical_success": physical,
        "strict_current_settle_success": bool(
            tasks and all(s["strict_current_settle_success"] for s in scores.values())
        ),
        "controller_outcomes_success": controller_success,
        "policy": probe["policy"],
        "robots": scores,
        "seconds": probe["seconds"],
        "full_horizon": probe.get("full_horizon"),
        "source_snapshot_verified": True,
        "initial_qpos_sha256": probe["initial_qpos_sha256"],
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "Phase-independent sampled contact/release/settling scoring and dispatch ordering. The extra strict_current_settle_success requires one continuous second ending at the final sample; legacy physical_success uses longest prior settling plus current valid contact. Evaluator geometry/speed/clearance are retained measurements, not independently reconstructed dynamics. Early v4 completion flags included aborts; use authored_phase_completed separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation-v4.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
