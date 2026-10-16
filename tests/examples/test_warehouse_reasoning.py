"""Compositional grounding and dispatch behavior; no simulator assets required."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.robotics.warehouse.coordination import (
    Coordinator,
    RobotState,
    build_dispatch_kernel,
    independent_assignment_baseline,
)
from examples.robotics.warehouse.language import (
    GroundingError,
    answer_question,
    ground_instruction,
    instruction_from_dict,
    parse_instruction,
    parse_reference,
    resolve,
)
from examples.robotics.warehouse.representation import Entity, Relation, SceneGraph


@pytest.fixture
def scene():
    graph = SceneGraph()
    for identity, kind, color, x in [
        ("parcel_0", "parcel", "orange", 0.0),
        ("parcel_1", "parcel", "orange", 0.2),
        ("receiver_0", "station", "magenta", 1.0),
        ("receiver_1", "station", "magenta", 2.0),
    ]:
        graph.upsert(
            Entity(
                identity,
                kind,
                {"color": color},
                (x, 0.0, 0.0),
                10.0,
                "rgbd/frame-10",
                uncertainty_m=0.01,
            )
        )
    return graph


def command(connector="while"):
    return f"move parcel_0 to receiver_0 {connector} move parcel_1 to receiver_1"


def test_tree_changes_actual_dispatch_and_baseline_double_books(scene):
    robots = [RobotState("r0", (0.0, 0.0, 0.0)), RobotState("r1", (3.0, 0.0, 0.0))]
    parallel = ground_instruction(parse_instruction(command()), scene, 10.0)
    assert set(independent_assignment_baseline(parallel, robots, scene).values()) == {
        "r0"
    }
    active = Coordinator().dispatch(parallel, robots, scene, 10.0)
    assert len(active) == 2 and {a.robot_id for a in active} == {"r0", "r1"}
    sequential = ground_instruction(parse_instruction(command("then")), scene, 10.0)
    coordinator = Coordinator()
    assert [
        a.task_id for a in coordinator.dispatch(sequential, robots, scene, 10.0)
    ] == ["task-1"]
    coordinator.finish("task-1", False, "execution/failure.json")
    assert coordinator.completed == set()
    coordinator.dispatch(sequential, robots, scene, 10.0)
    coordinator.finish("task-1", True, "execution/settled.json")
    assert [
        a.task_id for a in coordinator.dispatch(sequential, robots, scene, 10.0)
    ] == ["task-2"]


def test_shared_aisle_reservation_releases_only_on_outcome(scene):
    tasks = ground_instruction(parse_instruction(command()), scene, 10.0)
    coordinator = Coordinator()
    robots = [RobotState("r0", (0.0, 0.0, 0.0)), RobotState("r1", (3.0, 0.0, 0.0))]
    routes = {t.id: ("aisle:shared",) for t in tasks}
    assert len(coordinator.dispatch(tasks, robots, scene, 10.0, routes)) == 1
    assert coordinator.dispatch(tasks, robots, scene, 10.0, routes) == []
    coordinator.finish("task-1", True, "rollout/1")
    assert (
        coordinator.dispatch(tasks, robots, scene, 10.0, routes)[0].task_id == "task-2"
    )


def test_composed_relational_reference_and_uncertainty(scene):
    tree = parse_instruction("carry the orange parcel left of parcel_1 to receiver_0")
    assert ground_instruction(tree, scene, 10.0)[0].object_id == "parcel_0"
    scene.upsert(replace(scene.entities["parcel_0"], uncertainty_m=0.3))
    with pytest.raises(GroundingError, match="No fresh"):
        ground_instruction(tree, scene, 10.0)


def test_ambiguity_staleness_revision_and_qa(scene):
    assert (
        answer_question("Where is the orange parcel?", scene, 10.0).status
        == "ambiguous"
    )
    assert answer_question("Where is parcel_0?", scene, 13.0).status == "unknown"
    assert (
        answer_question("What is its secret mass?", scene, 10.0).status == "unsupported"
    )
    old = scene.entities["parcel_0"]
    scene.upsert(
        replace(old, position=(0.5, 0.0, 0.0), observed_at=14.0, source="rgbd/frame-14")
    )
    answer = answer_question("Where is parcel_0?", scene, 14.0)
    assert answer.details["position_m"] == (0.5, 0.0, 0.0)
    assert answer.sources == ("rgbd/frame-14",)
    assert scene.history[-1]["previous"]["position"] == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="Out-of-order"):
        scene.upsert(old)


def test_scene_containment_and_cycles(scene):
    scene.upsert(replace(scene.entities["parcel_0"], parent_id="receiver_0"))
    assert (
        resolve(parse_reference("parcel in receiver_0"), scene, 10.0).id == "parcel_0"
    )
    with pytest.raises(ValueError, match="acyclic"):
        scene.upsert(replace(scene.entities["receiver_0"], parent_id="parcel_0"))


def test_scene_roundtrip_and_relational_qa(scene):
    scene.upsert(replace(scene.entities["parcel_0"], parent_id="receiver_0"))
    scene.add_relation(
        Relation("parcel_1", "on", "receiver_1", 10.0, "rgbd/support-fit")
    )
    loaded = SceneGraph.from_dict(json.loads(json.dumps(scene.to_dict())))
    assert loaded.revision == scene.revision
    assert (
        answer_question("How many parcels?", loaded, 10.0).details["observed_count"]
        == 2
    )
    assert answer_question("What is in receiver_0?", loaded, 10.0).entity_ids == (
        "parcel_0",
    )
    assert (
        "rgbd/support-fit"
        in answer_question("What is on receiver_1?", loaded, 10.0).sources
    )
    assert answer_question("What is on receiver_0?", loaded, 10.0).status == "unknown"
    assert (
        answer_question("Is parcel_0 left of parcel_1?", loaded, 10.0).details["answer"]
        is True
    )
    assert (
        answer_question("Is parcel_0 right of parcel_1?", loaded, 10.0).details[
            "answer"
        ]
        is False
    )
    assert answer_question("How many parcels?", loaded, 13.0).status == "unknown"
    data = scene.to_dict()
    data["entities"].append(data["entities"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        SceneGraph.from_dict(data)


def test_strict_model_tree_validation(scene):
    tree = parse_instruction(command())
    decoded = instruction_from_dict(json.loads(json.dumps(tree.to_dict())))
    assert decoded == tree
    malformed = json.loads(json.dumps(tree.to_dict()))
    malformed["children"][0]["object"]["untrusted_python"] = "print('never')"
    with pytest.raises(ValueError, match="reference"):
        instruction_from_dict(malformed)
    with pytest.raises(ValueError):
        parse_instruction("move parcel_0 to receiver_0 unless someone says otherwise")


@pytest.mark.parametrize("substrate", ["alg0", "alg1", "alg2"])
def test_dispatch_kernel_substrates_and_freshness(scene, substrate):
    pytest.importorskip("torch")
    machine = build_dispatch_kernel().compile(substrate)
    inputs = {
        "robot_position": [1.0, 2.0, 3.0],
        "object_position": [4.0, 6.0, 3.0],
        "dependencies_ready": 1.0,
        "resources_free": 1.0,
        "robot_available": 1.0,
        "evidence_fresh": 1.0,
        "robot_capable": 1.0,
    }
    result = machine(inputs)
    assert float(result["cost"]) == pytest.approx(25.0)
    assert float(result["eligible"]) == pytest.approx(1.0)
    for gate in [
        "dependencies_ready",
        "resources_free",
        "robot_available",
        "evidence_fresh",
        "robot_capable",
    ]:
        assert float(machine({**inputs, gate: 0.0})["eligible"]) == pytest.approx(0.0)
    tasks = ground_instruction(parse_instruction(command()), scene, 10.0)
    coordinator = Coordinator(substrate=substrate)
    assert (
        coordinator.dispatch(tasks, [RobotState("r0", (0.0, 0.0, 0.0))], scene, 13.0)
        == []
    )
