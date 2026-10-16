"""Recorded model proposals must match the physical executor's exact input."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.robotics.warehouse.validate_jobs import validate_model_mission


@pytest.fixture
def mission(tmp_path):
    source = ROOT / "docs/results/robotics/warehouse/language/v4-three-robot-mission"
    target = tmp_path / "mission"
    target.mkdir()
    for name in ("instruction.json", "proposal.json", "protocol.json"):
        shutil.copyfile(source / name, target / name)
    return tmp_path, {
        "instruction": json.loads((target / "protocol.json").read_text())["text"],
        "instruction_tree": json.loads((target / "instruction.json").read_text()),
    }


def test_actual_model_receipt_binds_execution(mission):
    directory, probe = mission
    result = validate_model_mission(directory, probe)
    assert result["recorded_proposal_binding_verified"]
    assert result["attempts"] == 1


def test_changed_executed_tree_rejected(mission):
    directory, probe = mission
    probe["instruction_tree"]["children"].reverse()
    with pytest.raises(ValueError, match="Executed tree"):
        validate_model_mission(directory, probe)


def test_rehashed_instruction_cannot_replace_model_output(mission):
    directory, probe = mission
    probe["instruction_tree"]["children"].reverse()
    path = directory / "mission/instruction.json"
    path.write_text(json.dumps(probe["instruction_tree"]))
    receipt_path = directory / "mission/proposal.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["instruction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Executed tree"):
        validate_model_mission(directory, probe)


def test_missing_sidecar_rejected(mission):
    directory, probe = mission
    (directory / "mission/proposal.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        validate_model_mission(directory, probe)


def test_non_model_runs_remain_explicit(tmp_path):
    assert validate_model_mission(tmp_path, {}) is None
