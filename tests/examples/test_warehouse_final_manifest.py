import json
from pathlib import Path

import pytest

from examples.robotics.warehouse.learned_policy.data import sha256
from examples.robotics.warehouse.learned_policy.final_manifest import freeze


def test_final_manifest_requires_every_candidate_and_is_immutable(tmp_path):
    protocol = json.loads(
        Path(
            "examples/robotics/warehouse/learned_policy/final-comparison-protocol.json"
        ).read_text()
    )
    with pytest.raises(ValueError, match="every preregistered"):
        freeze({}, tmp_path, tmp_path / "source.zip", tmp_path / "frozen.json")
    policies = {}
    for spec in protocol["policies"]:
        if spec["kind"] == "authored":
            continue
        directory = tmp_path / spec["id"]
        directory.mkdir()
        for name in ("weights.pt", "policy.json", "kinematics.json"):
            (directory / name).write_bytes((spec["id"] + name).encode())
        (directory / "training.json").write_text(
            json.dumps(
                {
                    "weights_sha256": sha256(directory / "weights.pt"),
                    "config_sha256": sha256(directory / "policy.json"),
                }
            )
        )
        policies[spec["id"]] = directory
    for case in protocol["cases"]:
        (tmp_path / f"{case}.json").write_text("{}")
    (tmp_path / "source.zip").write_bytes(b"fixed runtime archive")
    result = freeze(
        policies, tmp_path, tmp_path / "source.zip", tmp_path / "frozen.json"
    )
    assert len(result["candidates"]) == 7
    assert len(result["conditions"]) == 5
    assert result["expected_physical_runs"] == 40
    with pytest.raises(FileExistsError):
        freeze(policies, tmp_path, tmp_path / "source.zip", tmp_path / "frozen.json")
    directory = policies["gru-primary"]
    (directory / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="no longer matches"):
        freeze(policies, tmp_path, tmp_path / "source.zip", tmp_path / "other.json")
