"""Sensor boundary and episode-disjoint learning regression tests."""

import copy
import json

import numpy as np
import pytest

from examples.robotics.warehouse.learned_policy.contract import (
    FEATURE_NAMES,
    SCHEMA,
    features,
)
from examples.robotics.warehouse.learned_policy.data import (
    load_episode,
    load_manifest,
    sha256,
)


def observation():
    return {
        "joint_position": [0.0] * 31,
        "joint_velocity": [0.0] * 31,
        "imu_quaternion": [1.0, 0, 0, 0],
        "imu_angular_velocity": [0.0] * 3,
        "task_active": True,
        "sensor_state": {
            "root_position": [0, 0, 0.8],
            "object_position": [1, 2, 3],
            "destination_position": [2, 3, 4],
            "object_measurement_age_s": 0,
            "destination_measurement_age_s": 0.1,
            "camera_valid": True,
            "destination_valid": True,
            "camera_confidence": 1,
            "pad_force_n": [0, 0],
            "pad_force_local_n": [[0, 0, 0], [0, 0, 0]],
            "stance_contacts": [True, True],
            "object_covariance": np.eye(3).tolist(),
        },
    }


def episode(path, identifier):
    action = {
        "velocity_command": [0] * 3,
        "arm_target": [0] * 7,
        "grip": 0,
        "arm_feedforward": [0] * 7,
        "arm_active": False,
        "done": False,
    }
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "id": identifier,
                "steps": [
                    {"t": i * 0.02, "observation": observation(), "action": action}
                    for i in range(3)
                ],
            }
        )
    )
    return {"path": path.name, "sha256": sha256(path), "physical_run_id": identifier}


def test_privileged_fields_cannot_change_features():
    original = observation()
    modified = copy.deepcopy(original)
    modified.update(phase="release", t=800, evaluation={"object_position": [300] * 3})
    modified["sensor_state"]["evaluation"] = {"success": True}
    np.testing.assert_array_equal(features(original), features(modified))
    assert len(features(original)) == len(FEATURE_NAMES)


def test_lane_translation_invariant_and_missing_observation_explicit():
    first = observation()
    shifted = copy.deepcopy(first)
    for key in ("root_position", "object_position", "destination_position"):
        shifted["sensor_state"][key][1] += 4
    np.testing.assert_array_equal(features(first), features(shifted))
    shifted["sensor_state"]["object_position"] = None
    shifted["sensor_state"]["object_measurement_age_s"] = None
    x = features(shifted)
    assert x[FEATURE_NAMES.index("object_present")] == 0
    assert x[FEATURE_NAMES.index("object_age")] == 10


def test_rejects_legacy_without_actions(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"steps": []}))
    with pytest.raises(ValueError, match="action provenance"):
        load_episode(path)


def test_rejects_physical_run_split_leakage(tmp_path):
    train = episode(tmp_path / "a.json", "same-run")
    validation = episode(tmp_path / "b.json", "same-run-other-robot")
    validation["physical_run_id"] = train["physical_run_id"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"train": [train], "validation": [validation], "test": []})
    )
    with pytest.raises(ValueError, match="Physical run leaks"):
        load_manifest(manifest)


def test_rejects_corrupt_episode(tmp_path):
    train = episode(tmp_path / "a.json", "train")
    validation = episode(tmp_path / "b.json", "validation")
    (tmp_path / "a.json").write_text("changed")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"train": [train], "validation": [validation], "test": []})
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(manifest)


def test_padded_frames_do_not_affect_training_loss():
    torch = pytest.importorskip("torch")
    from examples.robotics.warehouse.learned_policy.train import loss

    prediction = torch.zeros(1, 3, 20)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    original = loss(prediction, target, mask)
    prediction[:, 2] = 100
    assert loss(prediction, target, mask) == original


def test_saved_policy_preserves_recurrent_sequence_and_reset(tmp_path):
    torch = pytest.importorskip("torch")
    from examples.robotics.warehouse.learned_policy.contract import ACTION_NAMES
    from examples.robotics.warehouse.learned_policy.model import (
        PolicyRuntime,
        SequencePolicy,
    )

    torch.manual_seed(3)
    model = SequencePolicy(hidden=8)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    (tmp_path / "policy.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "hidden": 8,
                "feature_names": FEATURE_NAMES,
                "action_names": ACTION_NAMES,
                "input_mean": [0] * len(FEATURE_NAMES),
                "input_scale": [1] * len(FEATURE_NAMES),
                "action_mean": [0] * len(ACTION_NAMES),
                "action_scale": [1] * len(ACTION_NAMES),
            }
        )
    )
    one, two = PolicyRuntime(tmp_path), PolicyRuntime(tmp_path)
    first = one.step(observation())[1]
    np.testing.assert_array_equal(first, two.step(observation())[1])
    second = one.step(observation())[1]
    np.testing.assert_array_equal(second, two.step(observation())[1])
    assert first != second  # The policy carries its own memory.
    one.reset()
    np.testing.assert_array_equal(first, one.step(observation())[1])


def test_comparison_rejects_different_initial_physics(tmp_path, monkeypatch):
    from examples.robotics.warehouse.learned_policy.compare import compare

    monkeypatch.setattr(
        "examples.robotics.warehouse.learned_policy.compare.validate",
        lambda path: {
            "robots": {
                "r0": {
                    "strict_current_settle_success": True,
                    "current_settled_seconds": 1.2,
                }
            }
        },
    )
    metric = {
        "physical_success": True,
        "controller_declared_completion": True,
        "placement_xy_error_m": 0.001,
        "settled_seconds": 1.2,
        "minimum_root_height": 0.75,
    }
    probe = {
        "scenario": {"id": "case"},
        "initial_qpos_sha256": "same",
        "full_horizon": True,
        "requested_duration_s": 85,
        "mujoco_version": "test",
        "render_backend": "egl",
        "opengl": {"renderer": "same-test-renderer"},
        "source_sha256": {"physics.py": "same-code"},
        "evaluation": {"r0": metric},
        "seconds": 85,
        "termination_reason": "horizon",
        "learned_policy_sha256": {"weights.pt": "weights"},
    }
    teacher, learner = tmp_path / "teacher", tmp_path / "learner"
    for root in (teacher, learner):
        (root / "case").mkdir(parents=True)
        (root / "case" / "probe.json").write_text(json.dumps(probe))
    assert compare(teacher, {"bc": learner}, ["case"])["successes"]["bc"] == 1
    probe["initial_qpos_sha256"] = "different"
    (learner / "case" / "probe.json").write_text(json.dumps(probe))
    with pytest.raises(ValueError, match="Initial physics mismatch"):
        compare(teacher, {"bc": learner}, ["case"])


def test_success_curation_filters_train_only_and_retains_failures(tmp_path):
    from examples.robotics.warehouse.learned_policy.ablation import curate

    entries = [
        {
            "path": f"{name}.json.gz",
            "sha256": name,
            "physical_run_id": name,
            "robot_id": "r0",
        }
        for name in ("train-good", "train-bad", "validation-bad")
    ]
    manifest = {
        "train": entries[:2],
        "validation": entries[2:],
        "test": [],
        "evaluation_only_teacher_outcomes": {
            "train-good/r0": {"strict_current_settle_success": True},
            "train-bad/r0": {
                "physical_success": True,
                "strict_current_settle_success": False,
            },
            "validation-bad/r0": {"strict_current_settle_success": False},
        },
    }
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    output = tmp_path / "curated.json"
    result = curate(source, output)
    assert result["train"] == [entries[0]]
    assert result["validation"] == [entries[2]]
    assert result["curation"]["excluded_train_episodes"][0]["sha256"] == "train-bad"
    assert json.loads(source.read_text()) == manifest


def test_missing_sensor_masks_and_age_keep_contract_units():
    from examples.robotics.warehouse.learned_policy.contract import input_statistics

    train = np.stack([features(observation()) for _ in range(4)])
    mean, scale = input_statistics(train)
    missing = observation()
    missing["sensor_state"]["camera_valid"] = False
    missing["sensor_state"]["object_position"] = None
    missing["sensor_state"]["object_measurement_age_s"] = None
    normalized = (features(missing) - mean) / scale
    assert normalized[FEATURE_NAMES.index("camera_valid")] == 0
    assert normalized[FEATURE_NAMES.index("object_present")] == 0
    assert normalized[FEATURE_NAMES.index("object_age")] == 10
    assert mean[FEATURE_NAMES.index("root_height")] == pytest.approx(0.8)
    assert scale[FEATURE_NAMES.index("root_height")] == pytest.approx(0.01)
