"""Model boundary tests use data proposals; real model evidence lives in results."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from examples.robotics.warehouse.language_eval import fixture_scene, score
from examples.robotics.warehouse.model_language import (
    decode_request,
    ground_request,
    strict_json,
)


def reference(name="parcel_0"):
    return {"name": name, "color": None, "relation": None, "anchor": None}


def instruction():
    return {
        "kind": "instruction",
        "tree": {
            "operation": "pick_place",
            "object": reference(),
            "destination": reference("receiver_0"),
            "children": [],
        },
    }


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{}\n```",
        "{} trailing",
        "[]",
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '"__import__("os")"',
    ],
)
def test_json_boundary_rejects_nondata_and_duplicate_fields(raw):
    with pytest.raises(ValueError):
        strict_json(raw)


@pytest.mark.parametrize(
    "proposal",
    [
        {"kind": "query", "reference": reference(), "answer": "made up"},
        {"kind": "unsupported", "reason": ""},
        {"kind": "execute", "code": "print(1)"},
        {"kind": "query", "reference": {**reference(), "position": [1, 2, 3]}},
    ],
)
def test_schema_rejects_model_claims_and_executables(proposal):
    with pytest.raises(ValueError):
        decode_request(json.dumps(proposal))


def test_model_query_answers_current_observed_evidence_not_model_belief():
    scene = fixture_scene()
    request = decode_request(json.dumps({"kind": "query", "reference": reference()}))
    first = ground_request(request, scene, 10.0)
    assert first["position_m"] == [0.0, 0.0, 0.8]
    assert first["sources"] == ["authored-parser-fixture"]
    scene.upsert(
        replace(
            scene.entities["parcel_0"],
            position=(4.0, 2.0, 0.9),
            observed_at=11.0,
            source="new-camera-frame",
        )
    )
    second = ground_request(request, scene, 11.0)
    assert second["position_m"] == [4.0, 2.0, 0.9]
    assert second["sources"] == ["new-camera-frame"]
    assert second["scene_revision"] > first["scene_revision"]
    assert ground_request(request, scene, 14.0)["status"] == "unknown"


def test_grounding_does_not_arbitrarily_pick_ambiguous_objects():
    proposal = instruction()
    proposal["tree"]["object"] = {**reference("parcel"), "color": "orange"}
    result = ground_request(decode_request(json.dumps(proposal)), fixture_scene(), 10.0)
    assert result["status"] == "ambiguous"
    assert result["candidates"] == ["parcel_1", "parcel_2"]
    assert "tasks" not in result


def test_unknown_destination_cannot_be_created_by_model():
    proposal = instruction()
    proposal["tree"]["destination"]["name"] = "invented_receiver"
    result = ground_request(decode_request(json.dumps(proposal)), fixture_scene(), 10.0)
    assert result["status"] == "unknown"
    assert "tasks" not in result


def test_eval_scores_ordering_and_dependencies_not_just_object_mentions():
    case = {
        "status": "supported",
        "pairs": [["p", "a"], ["q", "b"]],
        "composition": "sequence",
    }
    tasks = [
        {"id": "t1", "object_id": "p", "destination_id": "a", "dependencies": []},
        {"id": "t2", "object_id": "q", "destination_id": "b", "dependencies": []},
    ]
    assert not score(case, {"status": "supported", "tasks": tasks})
    tasks[1]["dependencies"] = ["t1"]
    assert score(case, {"status": "supported", "tasks": tasks})


@pytest.mark.parametrize("limit", ["0", "13", "-1"])
def test_budget_rejects_invalid_memory_cap_before_starting_process(limit, tmp_path):
    script = Path(__file__).parents[2] / "examples/robotics/g1/budget.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--memory-gib",
            limit,
            "--output",
            str(tmp_path),
            "--",
            "false",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--memory-gib must be between 1 and 12" in result.stderr
    assert not (tmp_path / "stdout.log").exists()


def test_authored_grammar_baseline_preserves_grounding_rejections():
    from examples.robotics.warehouse.language_eval import grammar_result

    result = grammar_result("Move the orange parcel to receiver_0.", fixture_scene())
    assert result["status"] == "ambiguous"
    assert result["candidates"] == ["parcel_1", "parcel_2"]


def test_unknown_model_request_kind_is_not_silently_treated_as_question():
    with pytest.raises(ValueError, match="Only validated"):
        ground_request(
            {"kind": "invent", "reference": reference()}, fixture_scene(), 10.0
        )


@pytest.mark.parametrize(
    "text",
    [
        "Move parcel_0 to receiver_0 while moving parcel_1 to receiver_1",
        "At the same time, move both parcels",
        "Concurrently deliver the parcels",
    ],
)
def test_semantic_guard_rejects_model_serializing_explicit_parallel_request(text):
    from examples.robotics.warehouse.language import Instruction
    from examples.robotics.warehouse.model_language import check_semantic_admission

    request = {"kind": "instruction", "tree": Instruction("sequence")}
    assert check_semantic_admission(text, request)["status"] == "rejected"
    assert request["tree"].operation == "sequence"  # Never silently repaired.


@pytest.mark.parametrize(
    "text",
    [
        "Don't move parcel_0",
        "Do not move parcel_0",
        "Never move parcel_0",
        "Move parcel_0 if the aisle is clear",
        "Move parcel_0 without crossing the aisle",
    ],
)
def test_semantic_guard_rejects_action_that_drops_constraints(text):
    from examples.robotics.warehouse.model_language import check_semantic_admission

    request = decode_request(json.dumps(instruction()))
    assert check_semantic_admission(text, request)["status"] == "rejected"


def test_semantic_guard_does_not_claim_to_verify_all_meaning():
    from examples.robotics.warehouse.language import Instruction
    from examples.robotics.warehouse.model_language import check_semantic_admission

    request = {"kind": "instruction", "tree": Instruction("parallel")}
    result = check_semantic_admission("Move one parcel while moving another", request)
    assert result["status"] == "admitted"
    assert "does not prove semantic fidelity" in result["limitation"]
    assert (
        check_semantic_admission("Move one parcel then move another", request)["status"]
        == "rejected"
    )
    assert (
        check_semantic_admission("Move one while moving another then return", request)[
            "status"
        ]
        == "rejected"
    )


def test_concise_model_tree_expands_to_shared_typed_instruction():
    raw = {
        "instruction": [
            "parallel",
            ["move", "blue carton", "receiver_0"],
            ["move", "parcel_1", "receiver_1"],
        ]
    }
    request = decode_request(json.dumps(raw))
    result = ground_request(request, fixture_scene(), 10.0)
    assert result["status"] == "supported"
    assert [(t["object_id"], t["destination_id"]) for t in result["tasks"]] == [
        ("parcel_0", "receiver_0"),
        ("parcel_1", "receiver_1"),
    ]
    assert all(not t["dependencies"] for t in result["tasks"])


def test_authored_alias_expansion_preserves_identifiers_and_spatial_relations():
    request = decode_request('{"where":"box_0"}')
    assert request["reference"].name == "box_0"
    request = decode_request('{"where":"the package to the left of parcel_1"}')
    assert request["reference"].name == "parcel"
    assert request["reference"].relation == "left of"
    assert ground_request(request, fixture_scene(), 10.0)["entity_ids"] == ["parcel_0"]


@pytest.mark.parametrize(
    "value",
    [
        {"instruction": ["move", "parcel_0"]},
        {"instruction": ["move", "parcel_0", "receiver_0", "execute"]},
        {"instruction": ["parallel", ["move", "parcel_0", "receiver_0"]]},
        {"instruction": ["eval", "print(1)"]},
        {"where": "purple parcel"},
        {"where": "parcel_0", "answer": "invented"},
        {"unsupported": ""},
    ],
)
def test_concise_model_boundary_remains_strict(value):
    with pytest.raises(ValueError):
        decode_request(json.dumps(value))


@pytest.mark.parametrize("relative", ["", "v2-development", "v2-challenge"])
def test_published_language_evidence_scores_replay_without_model(relative):
    import hashlib

    from examples.robotics.warehouse.language_eval import CASES, V2_CHALLENGE

    root = (
        Path(__file__).parents[2]
        / "docs/results/robotics/warehouse/language"
        / relative
    )
    if not (root / "summary.json").exists():
        pytest.skip("This retained model evaluation has not completed yet")
    freeze = json.loads((root / "freeze.json").read_text())
    assert (
        hashlib.sha256((root / "cases.json").read_bytes()).hexdigest()
        == freeze["cases_sha256"]
    )
    assert (
        hashlib.sha256((root / "system_prompt.txt").read_bytes()).hexdigest()
        == freeze["system_prompt_sha256"]
    )
    cases = V2_CHALLENGE if relative == "v2-challenge" else CASES
    rows = json.loads((root / "predictions.json").read_text())
    assert len(rows) == len(cases)
    for case, row in zip(cases, rows, strict=True):
        assert case["text"] == row["text"]
        try:
            result = ground_request(
                decode_request(row["proposal"]["raw_text"]), fixture_scene(), 10.0
            )
        except (ValueError, TypeError, KeyError) as exc:
            result = {"status": "invalid_model_output", "reason": str(exc)}
        result = json.loads(json.dumps(result))
        assert result == row["result"]
        assert score(case, result) == row["correct"]
