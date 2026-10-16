"""Predeclared capability, lifecycle and freshness cases; no physical outcome claims."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.robotics.warehouse.coordination import RobotState
from examples.robotics.warehouse.jobs import (
    Capability,
    JobScheduler,
    UnsupportedMission,
)
from examples.robotics.warehouse.language import parse_instruction
from examples.robotics.warehouse.mission_receipts import RecordedScheduler
from examples.robotics.warehouse.representation import Entity, SceneGraph


@pytest.fixture
def scene():
    graph = SceneGraph()
    for lane in range(2):
        for kind, prefix, x in (
            ("parcel", "parcel", 1.0),
            ("station", "receiver", 1.2),
        ):
            graph.upsert(
                Entity(
                    f"{prefix}_{lane}",
                    kind,
                    {},
                    (x, lane * 1.6, 0.8),
                    10.0,
                    "rgbd/observed",
                )
            )
    return graph


@pytest.fixture
def capabilities():
    return tuple(
        Capability(f"r{i}", (f"parcel_{i}",), (f"receiver_{i}",)) for i in range(2)
    )


@pytest.fixture
def robots():
    return [RobotState("r0", (0.0, 0.0, 0.8)), RobotState("r1", (0.0, 1.6, 0.8))]


def mission(scene, capabilities, connector="while", substrate="alg0"):
    tree = parse_instruction(
        f"move parcel_0 to receiver_0 {connector} move parcel_1 to receiver_1"
    )
    return JobScheduler.from_instruction(tree, scene, 10.0, capabilities, substrate)


def test_whole_mission_rejected_before_partial_execution(scene, capabilities):
    tree = parse_instruction(
        "move parcel_0 to receiver_0 then move parcel_1 to receiver_0"
    )
    with pytest.raises(UnsupportedMission) as exc:
        JobScheduler.from_instruction(tree, scene, 10.0, capabilities)
    assert exc.value.tasks == ("task-2",)


@pytest.mark.parametrize("substrate", ["alg0", "alg1", "alg2"])
def test_explicit_scope_beats_wrong_nearest_robot_and_gate_explanations(
    scene, capabilities, robots, substrate
):
    pytest.importorskip("torch")
    scheduler = mission(scene, capabilities, "then", substrate)
    robots[0] = replace(robots[0], position=(20.0, 20.0, 0.8))
    assigned = scheduler.dispatch(robots, scene, 10.0)
    assert [(a.task_id, a.robot_id) for a in assigned] == [("task-1", "r0")]
    gates = scheduler.coordinator.last_gates
    assert gates["task-1"]["r1"]["robot_capable"] == 0.0
    answer = scheduler.explain("Why is r1 waiting?")
    assert answer["status"] == "supported"
    assert answer["tasks"][0]["reasons"] == ["dependencies_ready"]
    assert answer["scene_revision"] == scene.revision
    assert answer["checked_at"] == 10.0
    assert "assigned" in scheduler.explain("Why is r0 waiting?")["text"]


def test_failed_execution_never_retries_or_unlocks_dependency(
    scene, capabilities, robots
):
    scheduler = mission(scene, capabilities, "then")
    scheduler.dispatch(robots, scene, 10.0)
    scheduler.finish("task-1", False, "physical/failure.json", 10.1)
    assert scheduler.dispatch(robots, scene, 10.2) == []
    assert scheduler.coordinator.failed == {"task-1"}
    assert scheduler.explain("Why is r0 waiting?")["tasks"][0]["reasons"] == [
        "execution_failed_no_automatic_retry"
    ]
    assert scheduler.explain("Why is r1 waiting?")["tasks"][0]["reasons"] == [
        "dependencies_ready"
    ]


def test_second_job_uses_fresh_revised_position_after_dependency(
    scene, capabilities, robots
):
    scheduler = mission(scene, capabilities, "then")
    scheduler.dispatch(robots, scene, 10.0)
    scheduler.finish("task-1", True, "physical/settled.json", 12.0)
    assert scheduler.dispatch(robots, scene, 15.0) == []
    assert scheduler.explain("Why is r1 waiting?")["tasks"][0]["reasons"] == [
        "evidence_fresh"
    ]
    scene.upsert(
        replace(
            scene.entities["parcel_1"],
            position=(8.0, 1.6, 0.8),
            observed_at=15.0,
            source="rgbd/new-object",
        )
    )
    scene.upsert(
        replace(
            scene.entities["receiver_1"], observed_at=15.0, source="rgbd/new-receiver"
        )
    )
    assigned = scheduler.dispatch(robots, scene, 15.0)
    assert assigned[0].task_id == "task-2"
    assert scheduler.coordinator.last_gates["task-2"]["r1"]["cost"] == pytest.approx(
        64.0
    )


def test_repeated_job_dag_and_failure_of_missing_robot_contract(scene, robots):
    capabilities = (
        Capability("r0", ("parcel_0", "parcel_1"), ("receiver_0", "receiver_1")),
    )
    scheduler = mission(scene, capabilities, "then")
    with pytest.raises(ValueError, match="No explicit capability"):
        scheduler.dispatch(robots, scene, 10.0)
    scheduler.dispatch(robots[:1], scene, 10.0)
    scheduler.finish("task-1", True, "physical/job-1", 10.1)
    assigned = scheduler.dispatch(robots[:1], scene, 10.2)
    assert assigned[0].task_id == "task-2" and assigned[0].robot_id == "r0"


def test_no_evidence_no_explanation_and_uncertain_observation_blocks(
    scene, capabilities, robots
):
    scheduler = mission(scene, capabilities)
    assert scheduler.explain("Why is r0 waiting?")["status"] == "unknown"
    assert scheduler.explain("Why is r9 waiting?")["status"] == "unknown"
    assert scheduler.explain("How heavy is the parcel?")["status"] == "unsupported"
    scene.upsert(replace(scene.entities["parcel_0"], confidence=0.2))
    assigned = scheduler.dispatch(robots, scene, 10.0)
    assert [a.task_id for a in assigned] == ["task-2"]
    assert scheduler.explain("Why is r0 waiting?")["tasks"][0]["reasons"] == [
        "evidence_fresh"
    ]


@pytest.mark.parametrize("substrate", ["alg0", "alg1", "alg2"])
def test_exact_pair_capability_is_not_cartesian_product(scene, robots, substrate):
    pytest.importorskip("torch")
    caps = (
        Capability(
            "r0",
            ("parcel_0", "parcel_1"),
            ("receiver_0", "receiver_1"),
            allowed_pairs=(("parcel_0", "receiver_0"), ("parcel_1", "receiver_1")),
        ),
    )
    invalid = parse_instruction("move parcel_0 to receiver_1")
    with pytest.raises(UnsupportedMission):
        JobScheduler.from_instruction(invalid, scene, 10.0, caps, substrate)
    scheduler = mission(scene, caps, "then", substrate)
    assigned = scheduler.dispatch(robots[:1], scene, 10.0)
    assert (
        assigned[0].object_id == "parcel_0"
        and assigned[0].destination_id == "receiver_0"
    )
    scheduler.finish("task-1", True, "physical/stage-a", 10.1)
    assigned = scheduler.dispatch(robots[:1], scene, 10.2)
    assert (
        assigned[0].object_id == "parcel_1"
        and assigned[0].destination_id == "receiver_1"
    )
    # Test the actual numerical capability gate independently of preflight.
    changed = replace(
        scheduler.tasks[0], id="invalid-pair", destination_id="receiver_1"
    )
    from examples.robotics.warehouse.coordination import Coordinator

    coordinator = Coordinator(substrate=substrate)
    scoped = replace(robots[0], allowed_pairs=caps[0].allowed_pairs)
    assert coordinator.dispatch([changed], [scoped], scene, 10.0) == []
    assert coordinator.last_gates["invalid-pair"]["r0"]["robot_capable"] == 0.0


def test_recorded_receipt_answers_from_actual_dispatch_and_cannot_execute(
    scene, capabilities, robots
):
    scheduler = mission(scene, capabilities, "then")
    scheduler.dispatch(robots, scene, 10.0)
    raw = json.loads(json.dumps(scheduler.snapshot()))
    recorded = RecordedScheduler(raw, "run/scheduler.json")
    answer = recorded.explain("Why is r1 waiting?")
    assert answer["tasks"] == scheduler.explain("Why is r1 waiting?")["tasks"]
    assert answer["tasks"][0]["reasons"] == ["dependencies_ready"]
    assert answer["receipt_source"] == "run/scheduler.json"
    assert not hasattr(recorded, "dispatch") and not hasattr(recorded, "finish")
    scheduler.finish("task-1", True, "physical/complete", 10.1)
    scheduler.dispatch(robots, scene, 10.2)
    assert recorded.explain("Why is r1 waiting?")["tasks"][0]["reasons"] == [
        "dependencies_ready"
    ]
    raw["coordinator"]["completed"] = ["missing-task"]
    with pytest.raises(ValueError, match="Contradictory"):
        RecordedScheduler(raw)
