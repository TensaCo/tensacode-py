"""Recompute fleet placement and dispatch ordering from archived observations.

Evaluation fields are used only here; this validator never selects robot actions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def placement(records, robot_id):
    streak = longest = 0
    final = None
    for record in records:
        robot = record["robots"][robot_id]
        measured = robot["evaluation"]
        obj, surface = (
            np.asarray(measured[k], dtype=float)
            for k in ("object_position", "receiving_surface_xyz")
        )
        assert obj.shape == surface.shape == (3,)
        assert np.isfinite(np.r_[obj, surface]).all()
        contacts = set(measured["contacts"])
        xy = float(np.linalg.norm(obj[:2] - surface[:2]))
        assert abs(xy - measured["placement_xy_error_m"]) < 1e-9
        clear, speed = measured["hand_clearance_m"], measured["object_speed_m_s"]
        assert np.isfinite([clear, speed]).all() and min(clear, speed) >= 0
        valid = (
            robot["phase"] in {"withdraw", "complete"}
            and robot_id + "/receiving_marker" in contacts
            and np.max(np.abs(obj[:2] - surface[:2])) <= 0.065
            and abs(obj[2] - surface[2] - 0.025) <= 0.01
            and speed < 0.03
            and not contacts.intersection({robot_id + "/pad_a", robot_id + "/pad_b"})
            and robot["grip_command"] <= 1e-5
            and clear > 0.10
        )
        streak = streak + 1 if valid else 0
        longest = max(longest, streak)
        assert abs(streak * 0.02 - measured["streak"]) < 1e-8
        assert abs(longest * 0.02 - measured["settled_seconds"]) < 1e-8
        final = robot
    assert final is not None
    e = final["evaluation"]
    return {
        "success": bool(
            final["phase"] == "complete"
            and longest * 0.02 >= 1
            and e["minimum_root_height"] > 0.65
            and not final["controller_state"].get("failed", False)
        ),
        "settled_seconds": longest * 0.02,
        "placement_xy_error_m": e["placement_xy_error_m"],
        "hand_clearance_m": e["hand_clearance_m"],
    }


def validate(directory: Path):
    probe = json.loads((directory / "probe.json").read_text())
    compressed = directory / "records.json.gz"
    raw = (
        gzip.decompress(compressed.read_bytes())
        if compressed.exists()
        else (directory / "records.json").read_bytes()
    )
    records = json.loads(raw)
    poses = np.load(directory / "probe.npz", allow_pickle=False)["qpos"]
    assert len(records) == len(poses) > 0 and np.isfinite(poses).all()
    times = np.array([r["t"] for r in records])
    assert times[0] >= 0 and np.allclose(np.diff(times), 0.02, atol=1e-8)
    ids = set(records[0]["robots"])
    assert len(ids) == probe["robots"]
    assert all(set(r["robots"]) == ids for r in records)
    with zipfile.ZipFile(directory / "source.zip") as archive:
        for path, expected in probe["source_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == expected, path
    scores = {i: placement(records, i) for i in sorted(ids)}
    for robot_id, score in scores.items():
        assert score["success"] == probe["evaluation"][robot_id]["success"]
    events = json.loads((directory / "events.json").read_text())
    assigned = {e["task_id"]: e for e in events if e["type"] == "assigned"}
    finished = {e["task_id"]: e for e in events if e["type"] == "task_finished"}
    tasks = {t["id"]: t for t in probe["tasks"]}
    for task_id, event in assigned.items():
        assert event["object_id"] == tasks[task_id]["object_id"]
        assert event["destination_id"] == tasks[task_id]["destination_id"]
        for dependency in tasks[task_id]["dependencies"]:
            assert (
                dependency in finished
                and finished[dependency]["t"] <= event["t"] + 1e-9
            )
            outcome = next(x for x in probe["outcomes"] if x["task_id"] == dependency)
            assert outcome["success"], (
                "Dependent task started without controller completion"
            )
    for outcome in probe["outcomes"]:
        if outcome["success"]:
            event = finished[outcome["task_id"]]
            row = min(records, key=lambda r: abs(r["t"] - event["t"]))["robots"][
                outcome["robot_id"]
            ]
            assert row["phase"] == "complete" and not row["controller_state"].get(
                "failed", False
            )
    success = bool(
        tasks
        and len(probe["outcomes"]) == len(tasks)
        and all(o["success"] for o in probe["outcomes"])
        and all(s["success"] for s in scores.values())
    )
    assert success == probe["success"]
    # Resource overlaps are computed from actual dispatch intervals, not claimed labels.
    intervals = []
    for task_id, event in assigned.items():
        end = finished.get(task_id, {}).get("t", float(times[-1]))
        intervals.append((event["t"], end, event))
    for i, (start, end, event) in enumerate(intervals):
        for start2, end2, event2 in intervals[i + 1 :]:
            if min(end, end2) > max(start, start2) + 1e-9:
                assert event["robot_id"] != event2["robot_id"]
                assert not set(event["resources"]).intersection(event2["resources"])
    return {
        "validated": True,
        "success": success,
        "robots": scores,
        "seconds": float(times[-1]),
        "source_snapshot_verified": True,
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "dispatch_intervals": [
            {
                "task_id": e["task_id"],
                "robot_id": e["robot_id"],
                "start_s": a,
                "end_s": b,
            }
            for a, b, e in intervals
        ],
        "scope": "Recomputed sampled physical placement criteria and dependency/resource ordering. Ideal sensor simulation and correctness of perception are separate claims; no geometric collision-avoidance certification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
