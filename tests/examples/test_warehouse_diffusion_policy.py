"""Diffusion smoke tests: causal windows, DDPM math, learning and checkpoint runtime."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

torch = pytest.importorskip("torch")
from examples.robotics.warehouse.diffusion_policy.data import Windows, normalization
from examples.robotics.warehouse.diffusion_policy.model import (
    SCHEMA,
    Denoiser,
    Diffusion,
    PolicyRuntime,
)
from examples.robotics.warehouse.learned_policy.contract import (
    ACTION_NAMES,
    FEATURE_NAMES,
)


def sensor_observation():
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


def test_windows_are_causal_episode_local_and_padding_masked():
    x = np.arange(5, dtype=np.float32)[:, None].repeat(len(FEATURE_NAMES), axis=1)
    y = np.arange(5, dtype=np.float32)[:, None].repeat(20, axis=1)
    identity = (np.zeros(x.shape[1]), np.ones(x.shape[1]), np.zeros(20), np.ones(20))
    data = Windows(
        [(x, y), (x + 100, y + 100)], identity, history=3, horizon=4, stride=1
    )
    past, future, mask = data[0]
    np.testing.assert_array_equal(past[:, 0], [0, 0, 0])
    np.testing.assert_array_equal(future[:, 0], [0, 1, 2, 3])
    past, future, mask = data[4]
    np.testing.assert_array_equal(past[:, 0], [2, 3, 4])
    np.testing.assert_array_equal(future[:, 0], [4, 4, 4, 4])
    np.testing.assert_array_equal(mask, [1, 0, 0, 0])
    np.testing.assert_array_equal(data[5][0][:, 0], [100, 100, 100])


def test_normalization_binary_flags_and_constant_features():
    x = np.ones((3, len(FEATURE_NAMES)), dtype=np.float32) * 5
    y = np.ones((3, len(ACTION_NAMES)), dtype=np.float32)
    mean, scale, action_mean, action_scale = normalization([(x, y)])
    from examples.robotics.warehouse.learned_policy.contract import UNIT_SCALE_FEATURES

    for index, name in enumerate(FEATURE_NAMES):
        assert mean[index] == (0 if name in UNIT_SCALE_FEATURES else 5)
        assert scale[index] == pytest.approx(1 if name in UNIT_SCALE_FEATURES else 0.01)
    np.testing.assert_array_equal(action_mean[-2:], 0)
    np.testing.assert_array_equal(action_scale[-2:], 1)


def test_forward_and_reverse_ddpm_have_consistent_schedule():
    diffusion = Diffusion(8)
    clean, noise = torch.ones(2, 3, 20), torch.zeros(2, 3, 20)
    t = torch.tensor([0, 7])
    expected = diffusion.cumulative[t].sqrt()[:, None, None] * clean
    torch.testing.assert_close(diffusion.noisy(clean, t, noise), expected)
    assert torch.all(diffusion.cumulative[1:] < diffusion.cumulative[:-1])
    assert diffusion.previous[0] == 1


def test_synthetic_training_reduces_denoising_loss():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model, diffusion = Denoiser(history=2, horizon=2, hidden=32, blocks=1), Diffusion(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    x = torch.randn(8, 2, len(FEATURE_NAMES))
    y = torch.zeros(8, 2, 20)
    mask = torch.ones(8, 2)

    def objective():
        return diffusion.loss(model, x, y, mask, torch.Generator().manual_seed(27))

    first = float(objective().detach())
    for _ in range(100):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        optimizer.step()
    assert float(objective().detach()) < first * 0.15
    sampled = diffusion.sample(model, x, torch.Generator().manual_seed(3))
    assert sampled.shape == y.shape and torch.isfinite(sampled).all()


def checkpoint(directory):
    torch.manual_seed(2)
    architecture = {"history": 2, "horizon": 3, "hidden": 16, "blocks": 1}
    model = Denoiser(**architecture)
    torch.save(model.state_dict(), directory / "weights.pt")
    config = {
        "schema": SCHEMA,
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "architecture": architecture,
        "diffusion_steps": 4,
        "execute": 2,
        "seed": 42,
        "input_mean": [0] * len(FEATURE_NAMES),
        "input_scale": [1] * len(FEATURE_NAMES),
        "action_mean": [0] * 20,
        "action_scale": [1] * 20,
    }
    (directory / "policy.json").write_text(json.dumps(config))


def test_checkpoint_runtime_reset_and_privileged_fields_ignored(tmp_path):
    torch.set_num_threads(1)
    checkpoint(tmp_path)
    one, two = PolicyRuntime(tmp_path), PolicyRuntime(tmp_path)
    original = sensor_observation()
    modified = copy.deepcopy(original)
    modified.update(t=99999, phase="teacher_release", evaluation={"success": True})
    modified["sensor_state"]["ground_truth_object"] = [100, 200, 300]
    first = one.step(original)
    assert first == two.step(modified)
    second = one.step(original)
    assert second == two.step(modified)
    one.reset()
    assert first == one.step(original)
    assert len(first[1]) == 20
    assert first[0]["grip"] in (0, pytest.approx(0.028))
    assert all(
        abs(value) <= limit + 1e-6
        for value, limit in zip(first[0]["velocity_command"], [0.35, 0.35, 0.4])
    )


def test_masked_padding_does_not_affect_loss():
    class Zero(torch.nn.Module):
        def forward(self, noisy, timestep, observation):
            return torch.zeros_like(noisy)

    diffusion = Diffusion(4)
    actions = torch.zeros(1, 3, 20)
    x, mask = torch.zeros(1, 2, 103), torch.tensor([[1.0, 0.0, 0.0]])
    first = diffusion.loss(Zero(), x, actions, mask, torch.Generator().manual_seed(5))
    actions[:, 1:] = 1e8
    second = diffusion.loss(Zero(), x, actions, mask, torch.Generator().manual_seed(5))
    torch.testing.assert_close(first, second)


def test_training_loader_never_opens_final_episode(tmp_path):
    from examples.robotics.warehouse.diffusion_policy.data import load_training_manifest
    from examples.robotics.warehouse.learned_policy.contract import (
        SCHEMA as DATA_SCHEMA,
    )
    from examples.robotics.warehouse.learned_policy.data import sha256

    entries = []
    for identifier in ("train", "validation"):
        source = tmp_path / f"{identifier}.json"
        source.write_text(
            json.dumps(
                {
                    "schema": DATA_SCHEMA,
                    "id": identifier,
                    "steps": [
                        {
                            "t": 0,
                            "observation": sensor_observation(),
                            "action": {
                                "velocity_command": [0] * 3,
                                "arm_target": [0] * 7,
                                "grip": 0,
                                "arm_feedforward": [0] * 7,
                                "arm_active": False,
                                "done": False,
                            },
                        }
                    ],
                }
            )
        )
        entries.append(
            {
                "path": source.name,
                "sha256": sha256(source),
                "physical_run_id": identifier,
            }
        )
    manifest = {
        "train": [entries[0]],
        "validation": [entries[1]],
        "test": [
            {
                "path": "unavailable-final.json",
                "sha256": "0" * 64,
                "physical_run_id": "final",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    _, splits = load_training_manifest(path)
    assert list(splits) == ["train", "validation"]
    assert len(splits["train"]) == len(splits["validation"]) == 1
    manifest["test"][0]["physical_run_id"] = "train"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Physical run leaks"):
        load_training_manifest(path)
