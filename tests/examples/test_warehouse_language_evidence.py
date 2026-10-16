"""Bind claimed physical-scene answers to preserved observations and records."""

import gzip
import hashlib
import json
from pathlib import Path

from examples.robotics.warehouse.mission_questions import (
    answer_mission_question,
    decode_question,
)
from examples.robotics.warehouse.mission_receipts import RecordedScheduler
from examples.robotics.warehouse.representation import SceneGraph

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/results/robotics/warehouse/language"


def test_actual_scene_qa_answers_bind_to_rgbd_receipt_not_evaluator_state():
    evidence = EVIDENCE / "v4-physical-scene-qa"
    protocol = json.loads((evidence / "protocol.json").read_text())
    receipt = json.loads((evidence / "answers.json").read_text())
    source = (ROOT / receipt["scene_receipt"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == receipt["scene_sha256"]
    frame = next(
        f
        for f in json.loads(source)["frames"]
        if f["frame_index"] == protocol["frame_index"]
    )
    scene = SceneGraph.from_dict(frame["scene_graph"])
    assert receipt["historical_observed_at"] == frame["timestamp"]
    for row in receipt["questions"]:
        assert row["certificate"]["status"] == "certified"
        expected = answer_mission_question(
            decode_question(row["proposal"]["raw_text"]), scene, frame["timestamp"]
        )
        assert row["answer"] == expected
        assert expected["status"] == row["expected_status"]
        if expected["status"] == "supported":
            assert all("parcel_track" in s for s in expected["sources"])


def test_actual_waiting_answer_binds_to_failure_and_dependency_records():
    evidence = EVIDENCE / "v4-physical-scheduler-qa"
    receipt = json.loads((evidence / "answers.json").read_text())
    source = gzip.decompress((evidence / "scheduler-final.json.gz").read_bytes())
    assert hashlib.sha256(source).hexdigest() == receipt["scheduler_sha256"]
    model_source = (ROOT / receipt["model_receipt"]).read_bytes()
    assert hashlib.sha256(model_source).hexdigest() == receipt["model_receipt_sha256"]
    scheduler = RecordedScheduler(
        json.loads(source), source=receipt["answer"]["receipt_source"]
    )
    actual = answer_mission_question(
        decode_question(receipt["proposal"]["raw_text"]), None, 0.0, scheduler
    )
    assert actual == receipt["answer"]
    assert actual["tasks"][0]["reasons"] == ["execution_failed_no_automatic_retry"]
    assert actual["tasks"][1]["gates"]["dependencies_ready"] == 0.0
