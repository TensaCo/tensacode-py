"""Interrupted evaluations preserve evidence without turning unknowns into failures."""

import hashlib
import importlib.util
import json
import shutil
import struct
from pathlib import Path

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "package_native", HERE / "package_native.py"
)
package_native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_native)


def fixture_arm(root, n=2):
    arm = root / "psi01"
    cell = arm / "nominal"
    cell.mkdir(parents=True)
    protocol_file = HERE / "native_protocol.json"
    protocol = json.loads(protocol_file.read_text())
    ph = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    shutil.copy2(protocol_file, arm / protocol_file.name)
    shutil.copy2(HERE / "native_eval.py", arm / "native_eval.py")
    metadata = {
        "arm": "psi01",
        "task": protocol["task"],
        "action_tracker": protocol["action_tracker"],
        "runner_sha256": hashlib.sha256(
            (arm / "native_eval.py").read_bytes()
        ).hexdigest(),
        "protocol_sha256": ph,
        "checkpoint_sha256": protocol["checkpoint_sha256"]["psi01"],
    }
    (arm / "provenance.json").write_text(json.dumps(metadata))
    rows = []
    for config in range(n):
        success = config == 0
        receipt = {"config": config, "level": 0, "pelvis": [0, 0, 1]}
        native = {
            "seed": config,
            "task": f"simple/{protocol['task']}",
            "env_id": "simple",
            "body": "g1_simple",
            "success_privileged": success,
            "outcome": "success" if success else "timeout",
            "steps": 1,
            "time": 0.02,
            "provenance": {
                "env": {
                    "level": 0,
                    "render": "pt4_iso55",
                    "agent": "Psi0Agent",
                    "sim_mode": "mujoco_isaac",
                    "upstream": {
                        k: {"sha": protocol["source_revisions"][k], "dirty": False}
                        for k in ("psi0", "simple")
                    },
                }
            },
        }
        rows.append(
            {
                "arm": "psi01",
                "condition": "nominal",
                "config": config,
                "repeat": 0,
                "protocol_sha256": ph,
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "task": protocol["task"],
                "level": 0,
                "success": success,
                "outcome": native["outcome"],
                "steps": 1,
                "native_row": native,
                "source": "fixture",
                "initial_reset_audit": {
                    "receipt": receipt,
                    "sha256": hashlib.sha256(
                        json.dumps(receipt, sort_keys=True).encode()
                    ).hexdigest(),
                },
            }
        )
    (cell / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (cell / "policy_inputs.jsonl").write_text(
        json.dumps(
            {
                "instruction": "grasp the object on the tabletop.",
                "rgb_sha256": "a" * 64,
                "state_sha256": "b" * 64,
            }
        )
        + "\n"
    )
    return cell


def test_partial_rows_and_missing_media_are_not_extra_failures(tmp_path):
    cell = fixture_arm(tmp_path / "source")
    # A third outcome was interrupted mid-write: original bytes must survive.
    with (cell / "rows.jsonl").open("a") as f:
        f.write('{"config": 2')
    result = package_native.package(tmp_path / "source", tmp_path / "out")
    count = next(
        c
        for c in result["counts"]
        if c["arm"] == "psi01" and c["condition"] == "nominal"
    )
    assert (
        count["recorded_outcomes"],
        count["successes"],
        count["recorded_failures"],
        count["missing_outcomes"],
    ) == (2, 1, 1, 2)
    assert count["status"] == "incomplete"
    assert all(row["video"] is None for row in result["episodes"])
    assert (tmp_path / "out/psi01/nominal/rows.jsonl").read_bytes() == (
        cell / "rows.jsonl"
    ).read_bytes()
    assert next(c for c in result["counts"] if c["arm"] == "bc")["status"] == "unrun"


def test_started_condition_without_rows_is_incomplete(tmp_path):
    cell = fixture_arm(tmp_path / "source", n=0)
    (cell / "rows.jsonl").unlink()
    (cell / "policy_inputs.jsonl").unlink()
    result = package_native.package(tmp_path / "source", tmp_path / "out")
    count = result["counts"][0]
    assert count["status"] == "incomplete" and count["missing_outcomes"] == 4
    assert count["recorded_failures"] == count["recorded_outcomes"] == 0


def test_invalid_interior_line_and_duplicate_config_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\nbroken\n{}\n")
    with pytest.raises(json.JSONDecodeError):
        package_native.read_jsonl_prefix(path)
    cell = fixture_arm(tmp_path / "source")
    first = (cell / "rows.jsonl").read_text().splitlines()[0]
    (cell / "rows.jsonl").write_text(first + "\n" + first + "\n")
    with pytest.raises(AssertionError, match="duplicate episode"):
        package_native.package(tmp_path / "source", tmp_path / "out")


def test_interrupted_mp4_without_final_index_is_unavailable(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(struct.pack(">I4s", 8, b"mdat"))
    assert not package_native.has_mp4_index(path)
    path.write_bytes(struct.pack(">I4s", 8, b"mdat") + struct.pack(">I4s", 8, b"moov"))
    assert package_native.has_mp4_index(path)
