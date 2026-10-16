"""The reset checker must depend on own kinematics, not free object state."""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/robotics"))
from g1.model import ASSETS
from warehouse.physics_jobs_scene import compose_jobs_scene
from warehouse.physics_reset_clearance import ResetClearance


def test_reset_clearance_obstacle_and_object_state_isolation():
    _, xml, _ = compose_jobs_scene(1, 2.5, ASSETS)
    model = mujoco.MjModel.from_xml_string(xml)
    checker = ResetClearance(model)
    assert len(checker.encoder_q) >= 29
    own = model.qpos0.copy()
    target = own[checker.arm_q]
    far = [{"center": np.array([100.0, 100.0, 100.0]), "half": np.ones(3)}]
    assert checker.evaluate(own, target, far)["valid"]
    # A huge obstacle containing the robot must never certify the reset.
    obstacle = [{"center": np.zeros(3), "half": np.ones(3) * 10}]
    assert not checker.evaluate(own, target, obstacle)["valid"]
    baseline = checker.evaluate(own, target, far)
    object_q = model.joint("object_free").qposadr[0]
    own[object_q : object_q + 3] = [333, -777, 100]
    assert checker.evaluate(own, target, far) == baseline
    assert not checker.evaluate(own, target, [])["valid"]


def test_reset_clearance_fails_closed_for_invalid_values():
    from warehouse.physics_reset_clearance import valid_pad_forces

    _, xml, _ = compose_jobs_scene(1, 2.5, ASSETS)
    model = mujoco.MjModel.from_xml_string(xml)
    checker = ResetClearance(model)
    assert len(checker.encoder_q) >= 29
    q = model.qpos0.copy()
    target = q[checker.arm_q]
    box = {"center": np.ones(3) * 100, "half": np.ones(3)}
    for index in [checker.root_q, checker.arm_q[0]]:
        invalid = q.copy()
        invalid[index] = np.nan
        assert not checker.evaluate(invalid, target, [box])["valid"]
    for quaternion in [[0, 0, 0, 0], [2, 0, 0, 0]]:
        invalid = q.copy()
        invalid[checker.root_q + 3 : checker.root_q + 7] = quaternion
        assert not checker.evaluate(invalid, target, [box])["valid"]
    assert not checker.evaluate(q[:-1], target, [box])["valid"]
    assert not checker.evaluate(q, np.full(7, np.nan), [box])["valid"]
    assert not checker.evaluate(q, target[:6], [box])["valid"]
    for bad in [
        {"center": [np.nan, 0, 0], "half": [1, 1, 1]},
        {"center": [0, 0, 0], "half": [1, -1, 1]},
        {"center": [0, 0], "half": [1, 1, 1]},
        {"center": [0, 0, 0], "half": [1, np.inf, 1]},
    ]:
        assert not checker.evaluate(q, target, [bad])["valid"]
    checker.geoms = []
    assert not checker.evaluate(q, target, [box])["valid"]
    for bad in [None, [np.nan, 0], [0, -1], [np.inf, 0], [0], [0, 0, 0]]:
        assert not valid_pad_forces(bad)
    assert valid_pad_forces([0, 0.1])
