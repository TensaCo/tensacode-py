"""Counterexample tests for multi-job evidence, independent of simulator assets."""

import hashlib
import json
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.robotics.warehouse.validate_jobs import physical_scores, validate


def rows(valid):
    records = []
    streak = longest = 0
    for index, qualified in enumerate(valid):
        streak = streak + 1 if qualified else 0
        longest = max(longest, streak)
        evaluation = {}
        for job in ("a", "b"):
            evaluation[job] = {
                "object_position": [0.0, 0.0, 0.825],
                "receiving_surface_xyz": [0.0, 0.0, 0.8],
                "contacts": ["r0/receiving_marker" + ("_b" if job == "b" else "")]
                if qualified
                else [],
                "object_speed_m_s": 0.0,
                "hand_clearance_m": 0.2,
                "placement_xy_error_m": 0.0,
                "streak": streak * 0.02,
                "settled_seconds": longest * 0.02,
                "minimum_root_height": 0.8,
                "physical_success": qualified and streak >= 50,
            }
        job = "a" if index <= 1 else "b"
        records.append(
            {
                "t": index * 0.02,
                "robots": {
                    "r0": {
                        "phase": "complete"
                        if index in {1, len(valid) - 1}
                        else "carry",
                        "current_job": job,
                        "completed_jobs": [],
                        "grip_command": 0.0,
                        "controller_state": {},
                        "evaluation": deepcopy(evaluation[job]),
                        "job_evaluation": evaluation,
                    }
                },
            }
        )
    return records


@pytest.fixture
def bundle(tmp_path):
    records = rows([True] * 51)
    tasks, assignments, outcomes, events = [], [], [], []
    for index, job in enumerate(("a", "b")):
        task = {
            "id": f"task-{index + 1}",
            "object_id": f"parcel_0_{job}",
            "destination_id": f"receiver_0_{job}",
            "operation": "pick_place",
            "dependencies": [] if index == 0 else ["task-1"],
        }
        tasks.append(task)
        assignment = {
            "task_id": task["id"],
            "robot_id": "r0",
            "object_id": task["object_id"],
            "destination_id": task["destination_id"],
            "resources": [
                f"object:{task['object_id']}",
                f"destination:{task['destination_id']}",
            ],
        }
        assignments.append(assignment)
        outcomes.append({**assignment, "success": True, "evidence": "records.json"})
        events.extend(
            [
                {**assignment, "type": "assigned", "t": 0.0 if index == 0 else 0.04},
                {
                    "type": "task_finished",
                    "task_id": task["id"],
                    "robot": "r0",
                    "job": job,
                    "t": 0.02 if index == 0 else 1.0,
                    "controller_success": True,
                    "evaluation_success": index == 1,
                },
            ]
        )
    poses = np.zeros((51, 8))
    source = b"# Synthetic validation fixture, not a robot run.\n"
    probe = {
        "schema": "tensorcode.warehouse.physics.jobs.v1",
        "robots": 1,
        "jobs_per_robot": 2,
        "shared_physics": True,
        "seconds": 1.0,
        "requested_duration_s": 1.02,
        "termination_reason": "controller_finished",
        "tasks": tasks,
        "outcomes": outcomes,
        "source_sha256": {"fixture.py": hashlib.sha256(source).hexdigest()},
        "initial_qpos_sha256": hashlib.sha256(poses[0].tobytes()).hexdigest(),
        "evaluation": {"r0": deepcopy(records[-1]["robots"]["r0"]["job_evaluation"])},
        "physical_success": True,
        "success": True,
    }
    np.savez_compressed(tmp_path / "probe.npz", qpos=poses)
    with zipfile.ZipFile(tmp_path / "source.zip", "w") as archive:
        archive.writestr("fixture.py", source)
    for name, value in (
        ("records.json", records),
        ("probe.json", probe),
        ("events.json", events),
    ):
        (tmp_path / name).write_text(json.dumps(value))
    return tmp_path


def edit(path, function):
    value = json.loads(path.read_text())
    function(value)
    path.write_text(json.dumps(value))


def test_current_continuous_settle_does_not_reuse_historical_success():
    records = rows([True] * 50 + [False] + [True] * 2)
    result = physical_scores(records, "r0", "a")
    assert result["ever_physical_success"] is True
    assert result["longest_settled_seconds"] == 1.0
    assert result["current_settled_seconds"] == 0.04
    assert result["physical_success"] is False


def test_successful_controller_can_unlock_before_evaluator_success(bundle):
    result = validate(bundle)
    assert result["validated"] and result["success"] and result["initial_hash_verified"]
    assert result["robots"]["r0"]["b"]["current_settled_seconds"] == 1.02


@pytest.mark.parametrize(
    "field,value",
    [
        ("contacts", ["r0/receiving_marker", "r0/pad_a"]),
        ("object_speed_m_s", 0.1),
        ("hand_clearance_m", 0.05),
        ("object_position", [0.0, 0.0, 0.9]),
        ("minimum_root_height", 0.5),
    ],
)
def test_rejects_invalid_physical_success_claim(field, value):
    records = rows([True] * 51)
    records[-1]["robots"]["r0"]["job_evaluation"]["a"][field] = value
    with pytest.raises(ValueError):
        physical_scores(records, "r0", "a")


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda p: p.update(initial_qpos_sha256="wrong"), "Initial pose"),
        (lambda p: p["source_sha256"].update({"fixture.py": "wrong"}), "Source hash"),
        (lambda p: p.update(seconds=4.0), "duration"),
        (lambda p: p["tasks"][1].update(destination_id="receiver_0_a"), "exact"),
        (lambda p: p["evaluation"]["r0"]["b"].update(streak=0.1), "Probe/final"),
    ],
)
def test_rejects_altered_probe(bundle, mutation, message):
    edit(bundle / "probe.json", mutation)
    with pytest.raises(ValueError, match=message):
        validate(bundle)


def test_rejects_premature_dependency_and_duplicate_assignment(bundle):
    events = json.loads((bundle / "events.json").read_text())
    original = deepcopy(events)
    events[1]["controller_success"] = False
    (bundle / "events.json").write_text(json.dumps(events))
    with pytest.raises(ValueError, match="Dependency"):
        validate(bundle)
    original.append(original[0])
    (bundle / "events.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="Duplicate assigned"):
        validate(bundle)


def test_failed_controller_outcome_is_preserved_even_if_object_physically_settles(
    bundle,
):
    edit(
        bundle / "probe.json",
        lambda p: (p.update(success=False), p["outcomes"][1].update(success=False)),
    )
    edit(bundle / "events.json", lambda e: e[-1].update(controller_success=False))

    def fail_record(records):
        records[-1]["robots"]["r0"].update(
            phase="failed", controller_state={"failed": True}
        )

    edit(bundle / "records.json", fail_record)
    result = validate(bundle)
    assert result["physical_success"] is True
    assert result["controller_outcomes_success"] is False
    assert result["success"] is False
    assert result["failed_outcomes"][0]["task_id"] == "task-2"


def test_legacy_missing_initial_hash_is_explicit_not_claimed_verified(bundle):
    edit(bundle / "probe.json", lambda p: p.pop("initial_qpos_sha256"))
    result = validate(bundle)
    assert result["initial_hash_verified"] is False


def test_malformed_records_have_explicit_validation_error(bundle):
    edit(
        bundle / "records.json",
        lambda records: records[0]["robots"]["r0"].pop("job_evaluation"),
    )
    with pytest.raises(ValueError, match="Malformed"):
        validate(bundle)
