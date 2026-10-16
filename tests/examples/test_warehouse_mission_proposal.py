import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from examples.robotics.warehouse.language import instruction_from_dict
from examples.robotics.warehouse.language_review import certify_request, signature
from examples.robotics.warehouse.mission_proposal import propose_mission
from examples.robotics.warehouse.model_language import ModelProposal, decode_request


def test_valid_but_wrong_model_tree_does_not_become_executable_proposal():
    raw = '{"instruction":["parallel",["move","parcel_0","receiver_0"],["move","parcel_1","receiver_1"]]}'
    model = SimpleNamespace(propose=lambda *_: ModelProposal(raw, 1, 0.0, "fixture"))
    receipt, instruction = propose_mission(
        model, "Move parcel_0 to receiver_0 then move parcel_1 to receiver_1", "prompt"
    )
    assert receipt["schema_valid"] is True
    assert receipt["certificate"]["status"] == "rejected"
    assert receipt["dispatch_authorized"] is False
    assert instruction is None
    assert receipt["attempts"][0]["proposal"]["raw_text"] == raw


def test_actual_six_job_artifact_is_unchanged_model_tree_with_exact_hash():
    base = (
        Path(__file__).resolve().parents[2]
        / "docs/results/robotics/warehouse/language/v4-three-robot-mission"
    )
    receipt = json.loads((base / "proposal.json").read_text())
    raw_instruction = (base / "instruction.json").read_bytes()
    assert hashlib.sha256(raw_instruction).hexdigest() == receipt["instruction_sha256"]
    assert (
        hashlib.sha256((base / "protocol.json").read_bytes()).hexdigest()
        == receipt["protocol_sha256"]
    )
    assert len(receipt["attempts"]) == 1
    request = decode_request(receipt["attempts"][0]["proposal"]["raw_text"])
    tree = instruction_from_dict(json.loads(raw_instruction))
    assert signature(request) == signature({"kind": "instruction", "tree": tree})
    assert certify_request(receipt["text"], request)["status"] == "certified"
    assert tree.operation == "sequence"
    assert len(tree.children) == 2
    for job, group in zip(("a", "b"), tree.children, strict=True):
        assert group.operation == "parallel"
        assert [
            (node.object.name, node.destination.name) for node in group.children
        ] == [(f"parcel_{i}_{job}", f"receiver_{i}_{job}") for i in range(3)]
