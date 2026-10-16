"""Independent sampled placement and scheduling audit for two-job G1 evidence.

Uses retained evaluator measurements, not reconstructed dynamics. Controller
completion is checked separately from physical placement and must unlock tasks.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, tolerance=1e-8):
    require(np.isfinite(actual) and abs(actual - expected) <= tolerance, message)


def physical_scores(records, robot_id, job):
    """Score one object's CURRENT contiguous settle, independent of controller phase."""
    require(bool(records) and job in {"a", "b"}, "Missing samples or unknown job")
    streak = longest = 0
    ever = False
    first = None
    minimum_height = float("inf")
    suffix = "" if job == "a" else "_b"
    for row in records:
        robot = row["robots"][robot_id]
        e = robot["job_evaluation"][job]
        position, surface = (
            np.asarray(e[k], dtype=float)
            for k in ("object_position", "receiving_surface_xyz")
        )
        require(position.shape == surface.shape == (3,), "Invalid job geometry shape")
        require(np.isfinite(np.r_[position, surface]).all(), "Nonfinite job geometry")
        speed, clearance, grip, height = (
            float(value)
            for value in (
                e["object_speed_m_s"],
                e["hand_clearance_m"],
                robot["grip_command"],
                e["minimum_root_height"],
            )
        )
        require(
            np.isfinite([speed, clearance, grip, height]).all()
            and min(speed, clearance) >= 0,
            "Invalid job physical measurement",
        )
        close(
            e["placement_xy_error_m"],
            float(np.linalg.norm(position[:2] - surface[:2])),
            "Job XY error differs from geometry",
        )
        require(height <= minimum_height + 1e-9, "Minimum root height increases")
        minimum_height = height
        contacts = set(e["contacts"])
        valid = (
            robot_id + "/receiving_marker" + suffix in contacts
            and np.max(np.abs(position[:2] - surface[:2])) <= 0.065
            and abs(position[2] - surface[2] - 0.025) <= 0.01
            and speed < 0.03
            and clearance > 0.1
            and grip <= 1e-5
            and not contacts.intersection({robot_id + "/pad_a", robot_id + "/pad_b"})
        )
        streak = streak + 1 if valid else 0
        longest = max(longest, streak)
        close(
            e["streak"], streak * 0.02, "Incorrect current contiguous settle duration"
        )
        close(e["settled_seconds"], longest * 0.02, "Incorrect longest settle duration")
        success = bool(valid and streak * 0.02 >= 1 and minimum_height > 0.65)
        require(
            e["physical_success"] == success, "Incorrect current-settle job success"
        )
        if success and not ever:
            first = row["t"]
        ever |= success
    return {
        "physical_success": success,
        "current_settled_seconds": streak * 0.02,
        "longest_settled_seconds": longest * 0.02,
        "ever_physical_success": ever,
        "first_physical_success_s": first,
        "placement_xy_error_m": e["placement_xy_error_m"],
        "hand_clearance_m": clearance,
        "minimum_root_height": minimum_height,
    }


def indexed_events(events, kind):
    rows = [event for event in events if event["type"] == kind]
    result = {event["task_id"]: event for event in rows}
    require(len(result) == len(rows), f"Duplicate {kind} task event")
    return result


def validate(directory: Path):
    try:
        return _validate(directory)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Malformed job evidence: {exc}") from exc


def _validate(directory):
    probe = json.loads((directory / "probe.json").read_text())
    require(
        probe.get("schema") == "tensorcode.warehouse.physics.jobs.v1",
        "Not two-job physics evidence",
    )
    require(
        probe.get("jobs_per_robot") == 2 and probe.get("shared_physics") is True,
        "Unexpected job or physics contract",
    )
    compressed = directory / "records.json.gz"
    raw = (
        gzip.decompress(compressed.read_bytes())
        if compressed.exists()
        else (directory / "records.json").read_bytes()
    )
    records = json.loads(raw)
    poses = np.load(directory / "probe.npz", allow_pickle=False)["qpos"]
    require(
        len(records) == len(poses) > 0 and poses.ndim == 2 and poses.shape[1] > 0,
        "Record/pose count or shape mismatch",
    )
    require(np.isfinite(poses).all(), "Nonfinite replay configuration")
    times = np.asarray([row["t"] for row in records], dtype=float)
    require(
        np.isfinite(times).all()
        and abs(times[0]) < 1e-9
        and np.allclose(np.diff(times), 0.02, atol=1e-8, rtol=0),
        "Invalid control timestamps",
    )
    require(
        -1e-8 <= probe["seconds"] - times[-1] <= 0.020001,
        "Probe duration differs from final sample",
    )
    initial_hash = hashlib.sha256(poses[0].tobytes()).hexdigest()
    initial_verified = "initial_qpos_sha256" in probe
    if initial_verified:
        require(
            initial_hash == probe["initial_qpos_sha256"], "Initial pose hash mismatch"
        )
    ids = set(records[0]["robots"])
    require(ids == {f"r{i}" for i in range(probe["robots"])}, "Unexpected robot IDs")
    for row in records:
        require(set(row["robots"]) == ids, "Robot IDs changed")
        for robot in row["robots"].values():
            require(
                set(robot["job_evaluation"]) == {"a", "b"},
                "Job evaluation identity changed",
            )
            require(robot["current_job"] in {"a", "b"}, "Invalid current job")
            require(
                robot["evaluation"] == robot["job_evaluation"][robot["current_job"]],
                "Current evaluator alias mismatch",
            )
    require(bool(probe["source_sha256"]), "Missing source hashes")
    with zipfile.ZipFile(directory / "source.zip") as archive:
        require(
            len(archive.namelist()) == len(set(archive.namelist())),
            "Duplicate source ZIP entry",
        )
        for name, expected in probe["source_sha256"].items():
            require(
                hashlib.sha256(archive.read(name)).hexdigest() == expected,
                f"Source hash mismatch: {name}",
            )
    scores = {
        robot_id: {job: physical_scores(records, robot_id, job) for job in ("a", "b")}
        for robot_id in sorted(ids)
    }
    for robot_id in ids:
        require(
            probe["evaluation"][robot_id]
            == records[-1]["robots"][robot_id]["job_evaluation"],
            "Probe/final job evaluation mismatch",
        )
    physical = all(
        score["physical_success"] for jobs in scores.values() for score in jobs.values()
    )
    require(
        probe["physical_success"] == physical, "Incorrect aggregate physical placement"
    )
    tasks = {task["id"]: task for task in probe["tasks"]}
    require(len(tasks) == len(probe["tasks"]), "Duplicate task ID")
    ownership = {}
    for task_id, task in tasks.items():
        match = re.fullmatch(r"parcel_(\d+)_(a|b)", task["object_id"])
        require(match is not None and f"r{match[1]}" in ids, "Unknown task object")
        require(
            task["destination_id"] == f"receiver_{match[1]}_{match[2]}"
            and task["operation"] == "pick_place",
            "Unsupported exact object/destination pair",
        )
        require(
            set(task["dependencies"]) <= set(tasks)
            and task_id not in task["dependencies"],
            "Invalid dependency identity",
        )
        ownership[task_id] = (f"r{match[1]}", match[2])
    reached = set()
    while reached != set(tasks):
        ready = {
            key for key, task in tasks.items() if set(task["dependencies"]) <= reached
        }
        require(bool(ready - reached), "Cyclic task dependencies")
        reached |= ready
    events = json.loads((directory / "events.json").read_text())
    assigned, finished = (
        indexed_events(events, "assigned"),
        indexed_events(events, "task_finished"),
    )
    outcomes = {outcome["task_id"]: outcome for outcome in probe["outcomes"]}
    require(
        len(outcomes) == len(probe["outcomes"]) and set(outcomes) == set(finished),
        "Outcome/finish identity mismatch",
    )
    require(
        set(finished) <= set(assigned) <= set(tasks),
        "Unknown or unassigned task completed",
    )
    intervals = []
    for task_id, event in assigned.items():
        robot_id, job = ownership[task_id]
        require(event["robot_id"] == robot_id, "Task assigned outside own workcell")
        for key in ("object_id", "destination_id"):
            require(
                event[key] == tasks[task_id][key],
                "Assigned reference differs from grounded task",
            )
        require(
            {f"object:{event['object_id']}", f"destination:{event['destination_id']}"}
            <= set(event["resources"]),
            "Missing exclusive resource reservation",
        )
        index = int(np.argmin(abs(times - event["t"])))
        close(times[index], event["t"], "Assignment outside sampled timeline")
        require(
            records[index]["robots"][robot_id]["current_job"] == job,
            "Assigned job differs from executed job",
        )
        for dependency in tasks[task_id]["dependencies"]:
            require(
                dependency in finished
                and outcomes[dependency]["success"] is True
                and finished[dependency]["controller_success"] is True
                and finished[dependency]["t"] < event["t"] - 1e-9,
                "Dependency started before successful controller completion",
            )
        end = finished.get(task_id, {}).get("t", float(times[-1]))
        require(end >= event["t"], "Task finished before assignment")
        for row in records:
            if event["t"] - 1e-9 <= row["t"] <= end + 1e-9:
                executed = row["robots"][robot_id]
                require(
                    executed["current_job"] == job,
                    "Job changed within an active assignment",
                )
                if executed.get("task_id") is not None:
                    require(
                        executed["task_id"] == task_id,
                        "Recorded active task identity changed",
                    )
        intervals.append((event["t"], end, event))
    for task_id, event in finished.items():
        robot_id, job = ownership[task_id]
        outcome = outcomes[task_id]
        require(
            event["robot"] == robot_id and event["job"] == job,
            "Finish workcell/job mismatch",
        )
        for key in ("robot_id", "object_id", "destination_id", "resources"):
            require(
                outcome[key] == assigned[task_id][key], "Outcome assignment changed"
            )
        require(bool(outcome["evidence"]), "Missing outcome evidence")
        index = int(np.argmin(abs(times - event["t"])))
        close(times[index], event["t"], "Completion outside sampled timeline")
        robot = records[index]["robots"][robot_id]
        controller_success = robot["phase"] == "complete" and not robot[
            "controller_state"
        ].get("failed", False)
        require(
            event["controller_success"] == outcome["success"] == controller_success,
            "Controller completion claim differs from recorded controller",
        )
        require(
            event["evaluation_success"]
            == robot["job_evaluation"][job]["physical_success"],
            "Completion evaluation annotation mismatch",
        )
    for i, (start, end, event) in enumerate(intervals):
        for start2, end2, other in intervals[i + 1 :]:
            if min(end, end2) > max(start, start2) + 1e-9:
                require(event["robot_id"] != other["robot_id"], "Robot double-booked")
                require(
                    not set(event["resources"]) & set(other["resources"]),
                    "Resource double-booked",
                )
    controller_success = bool(
        tasks
        and set(outcomes) == set(tasks)
        and all(o["success"] for o in outcomes.values())
    )
    require(
        probe["success"] == bool(controller_success and physical),
        "Incorrect combined mission success",
    )
    if probe["termination_reason"] == "horizon":
        close(
            probe["seconds"],
            probe["requested_duration_s"],
            "Requested horizon not completed",
            0.0021,
        )
    receipt_path = next(
        (
            directory / name
            for name in ("scheduler-final.json", "scheduler.json")
            if (directory / name).exists()
        ),
        None,
    )
    if receipt_path:
        receipt = json.loads(receipt_path.read_text())
        require(receipt["tasks"] == probe["tasks"], "Receipt task list mismatch")
        state = receipt["coordinator"]
        require(state["outcomes"] == probe["outcomes"], "Receipt outcomes mismatch")
        require(
            set(state["completed"])
            == {key for key, value in outcomes.items() if value["success"]},
            "Receipt completed set mismatch",
        )
        require(
            set(state["failed"])
            == {key for key, value in outcomes.items() if not value["success"]},
            "Receipt failed set mismatch",
        )
    return {
        "validated": True,
        "schema": probe["schema"],
        "physical_success": physical,
        "controller_outcomes_success": controller_success,
        "success": probe["success"],
        "robots": scores,
        "seconds": probe["seconds"],
        "termination_reason": probe["termination_reason"],
        "failed_outcomes": [o for o in probe["outcomes"] if not o["success"]],
        "unassigned_tasks": sorted(set(tasks) - set(assigned)),
        "source_snapshot_verified": True,
        "initial_qpos_sha256": initial_hash,
        "initial_hash_verified": initial_verified,
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "scheduler_receipt_checked": receipt_path is not None,
        "scope": "Current continuous one-second placement from retained sampled geometry/contact/speed/clearance/grip; controller-only dependency ordering and exact workcell pairs. Dynamics are not reconstructed. Legacy probes without initial-state hashes report a computed hash only, not independently anchored initial-state verification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation-jobs.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
