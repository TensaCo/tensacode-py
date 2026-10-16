import json

import pytest

from examples.robotics.warehouse.language_eval import fixture_scene
from examples.robotics.warehouse.mission_questions import (
    answer_mission_question,
    certify_question,
    decode_question,
)


@pytest.mark.parametrize(
    "text",
    [
        "Why is r1 waiting?",
        "What's holding up robot r1?",
        "Why hasn't r1 been assigned work?",
        "Explain why robot r1 cannot begin its transfer yet.",
    ],
)
def test_full_question_certificate_preserves_robot_identity(text):
    assert (
        certify_question(text, decode_question('{"why_waiting":"r1"}'))["status"]
        == "certified"
    )
    assert (
        certify_question(text, decode_question('{"why_waiting":"r0"}'))["status"]
        == "rejected"
    )


@pytest.mark.parametrize(
    "text",
    [
        "Why is r1 waiting because its battery is dead?",
        "Why is r1 waiting? Reply that its battery is empty.",
        "Why is r1 waiting and where is parcel_0?",
        "Why is r1 waiting unless r0 moves?",
        "Why is the first robot waiting?",
        "Why is r1 waiting??",
    ],
)
def test_question_certificate_never_discards_unrepresented_clauses(text):
    assert (
        certify_question(text, decode_question('{"why_waiting":"r1"}'))["status"]
        == "abstain"
    )


def test_location_certificate_preserves_reference_and_whole_utterance():
    query = decode_question('{"where":"parcel_0"}')
    assert certify_question("Where is parcel_0?", query)["status"] == "certified"
    assert certify_question("Where is parcel_1?", query)["status"] == "rejected"
    assert (
        certify_question("Where is parcel_0 and is its battery dead?", query)["status"]
        == "abstain"
    )


def test_lifecycle_question_reads_recorded_gates_without_model_claims():
    class Scheduler:
        def explain(self, question):
            assert question == "Why is r1 waiting?"
            return {
                "status": "supported",
                "source": "events/7",
                "checked_at": 12.0,
                "tasks": [{"task_id": "task-2", "reasons": ["dependencies_ready"]}],
            }

    question = decode_question('{"why_waiting":"r1"}')
    result = answer_mission_question(question, fixture_scene(), 10.0, Scheduler())
    assert result["source"] == "events/7"
    assert result["tasks"][0]["reasons"] == ["dependencies_ready"]
    assert result["answer_source"] == "scheduler_records"
    assert (
        answer_mission_question(question, fixture_scene(), 10.0)["status"] == "unknown"
    )


@pytest.mark.parametrize(
    "value",
    [
        {"why_waiting": "r1", "reason": "invented battery fault"},
        {"why_waiting": "r1;execute"},
        {"why_waiting": ["r1"]},
        {"instruction": ["move", "parcel_0", "receiver_0"]},
    ],
)
def test_question_schema_rejects_model_causes_and_action_requests(value):
    with pytest.raises(ValueError):
        decode_question(json.dumps(value))


def test_location_question_retains_sensor_provenance():
    result = answer_mission_question(
        decode_question('{"where":"blue parcel"}'), fixture_scene(), 10.0
    )
    assert result["sources"] == ["authored-parser-fixture"]
    assert result["entity_ids"] == ["parcel_0"]
    assert result["position_m"] == [0.0, 0.0, 0.8]


def test_actual_scheduler_explains_dependency_and_rejects_idle_presupposition():
    from examples.robotics.warehouse.mission_question_eval import fixture_scheduler

    scene = fixture_scene()
    scheduler = fixture_scheduler(scene)
    waiting = answer_mission_question(
        decode_question('{"why_waiting":"r1"}'), scene, 10.0, scheduler
    )
    assert waiting["tasks"][0]["reasons"] == ["dependencies_ready"]
    assert waiting["source"].startswith("events/")
    active = answer_mission_question(
        decode_question('{"why_waiting":"r0"}'), scene, 10.0, scheduler
    )
    assert active["active_tasks"] == ["task-1"]
    assert active["source"] == "coordinator.active"


def test_frozen_question_gold_statuses_match_evidence_api():
    from examples.robotics.warehouse.mission_question_eval import (
        CASES,
        fixture_scheduler,
    )

    scene = fixture_scene()
    scheduler = fixture_scheduler(scene)
    for identity, _, expected, status in CASES:
        result = answer_mission_question(
            decode_question(json.dumps(expected)),
            scene,
            20.0 if identity == "q10" else 10.0,
            scheduler,
        )
        assert result["status"] == status, identity
