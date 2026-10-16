"""Geometry uses robot calibration and own sensors, never evaluator poses."""

import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from examples.robotics.warehouse.learned_policy.geometry import (
    export_calibration,
    forward,
    geometry_features,
    residual_targets,
)


def test_calibrated_fk_matches_reference_for_nonzero_pivots_and_joint_references(
    tmp_path,
):
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body name="root"><freejoint/><geom type="sphere" size=".1"/><body name="arm" pos=".1 .2 .3" euler="10 20 30"><joint type="hinge" axis="0 1 0" pos=".01 .02 .03" ref="20"/><geom type="sphere" size=".1"/><site name="grasp" pos=".3 -.1 .2" euler="10 0 0"/></body></body></worldbody></mujoco>"""
    )
    calibration = export_calibration(model, tmp_path / "kinematics.json")
    data = mujoco.MjData(model)
    rng = np.random.default_rng(5)
    for _ in range(20):
        data.qpos[:3] = rng.normal(size=3)
        data.qpos[3:7] = Rotation.random(random_state=rng).as_quat(scalar_first=True)
        data.qpos[7] = rng.normal()
        mujoco.mj_forward(model, data)
        p, r = forward(calibration, [[data.qpos[7]]], data.qpos[:3], data.qpos[3:7])
        np.testing.assert_allclose(p[0], data.site_xpos[0], atol=1e-12)
        np.testing.assert_allclose(r[0], data.site_xmat[0].reshape(3, 3), atol=1e-12)


def test_geometry_is_translation_invariant_and_ignores_extra_truth_fields():
    calibration = {
        "chain": [{}],
        "site_position": [0.1, 0.2, 0.3],
        "site_rotation": np.eye(3).tolist(),
    }
    observation = {
        "joint_position": [0.0] * 31,
        "imu_quaternion": [1, 0, 0, 0],
        "sensor_state": {
            "root_position": [1, 2, 3],
            "object_position": [2, 3, 4],
            "destination_position": [3, 4, 5],
        },
    }
    translated = copy.deepcopy(observation)
    for key in ("root_position", "object_position", "destination_position"):
        translated["sensor_state"][key] = np.asarray(
            translated["sensor_state"][key]
        ) + [4, -2, 7]
    translated["ground_truth_object_position"] = [999, 999, 999]
    translated["teacher_phase"] = "complete"
    np.testing.assert_allclose(
        geometry_features(calibration, [observation]),
        geometry_features(calibration, [translated]),
        atol=1e-6,
    )
    missing = copy.deepcopy(observation)
    missing["sensor_state"]["object_position"] = None
    np.testing.assert_array_equal(geometry_features(calibration, [missing])[0, 3:6], 0)


def test_arm_residual_target_reconstructs_original_command():
    targets = np.arange(40, dtype=np.float32).reshape(2, 20)
    observations = [
        {"joint_position": (np.arange(31) + shift).tolist()} for shift in (0, 5)
    ]
    residual = residual_targets(targets, observations)
    restored = residual.copy()
    restored[:, 3:10] += np.asarray([o["joint_position"][22:29] for o in observations])
    np.testing.assert_array_equal(restored, targets)
    np.testing.assert_array_equal(residual[:, :3], targets[:, :3])
    np.testing.assert_array_equal(residual[:, 10:], targets[:, 10:])


def test_geometry_rotation_channels_keep_bounded_units():
    from examples.robotics.warehouse.learned_policy.geometry_data import (
        input_statistics,
    )

    data = np.ones((8, 121), dtype=np.float32)
    mean, scale = input_statistics(data)
    np.testing.assert_array_equal(mean[112:], 0)
    np.testing.assert_array_equal(scale[112:], 1)
