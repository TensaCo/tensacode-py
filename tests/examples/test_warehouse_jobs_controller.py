"""Repeated-job transitions use fresh sensed targets and calibrated free-space routing."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples/robotics"))
from warehouse.physics_jobs_controller import RepeatedJobController


class Session:
    def run(self, *_):
        return [np.zeros((1, 29))]


def reset_controller(shift=0, age=0.1):
    c = object.__new__(RepeatedJobController)
    c.transit_waypoints = []
    c.transition_failed = False
    c.pre_reset_backout = False
    c.backed_out_for_reset = False
    c.reset_stable_since = None
    c.reset_started = 0.0
    c.reset_joints = np.zeros(7)
    c.default = np.zeros(29)
    c.action = np.zeros(29)
    c.scale = np.ones(29)
    c.session = Session()
    qpos = np.zeros(45)
    qpos[3] = 1
    c.observer = SimpleNamespace(
        proprio=SimpleNamespace(qpos=qpos, qvel=np.zeros(43), time=3.1)
    )
    c.sensed = {
        "root_position": [0.99 + shift, 0.01, 0.79],
        "object_position": [1.25 + shift, 0.75, 0.808],
        "object_measurement_age_s": age,
    }
    c.current_job = "b"
    return c


def test_fresh_visual_target_creates_backout_before_lateral_motion():
    c = reset_controller()
    c.control(3.1)
    c.control(3.32)
    assert c.phase == "back_out" and c.arm is None
    np.testing.assert_allclose(c.transit_waypoints, [[0.69, 0.01], [0.69, 1.0]])
    # Both route legs stay behind the sensed pickup staging X by30cm.
    assert all(p[0] < 1.25 - 0.25 - 0.29 for p in c.transit_waypoints)


def test_route_translates_with_observed_reference_frame():
    a = reset_controller()
    b = reset_controller(shift=5)
    a.control(3.1)
    b.control(3.1)
    a.control(3.32)
    b.control(3.32)
    np.testing.assert_allclose(
        np.array(b.transit_waypoints) - a.transit_waypoints, [[5, 0], [5, 0]]
    )


def test_stale_or_missing_observation_blocks_next_job_route():
    for age in (None, 2):
        c = reset_controller(age=age)
        c.control(3.1)
        assert c.reset_started is not None and not c.transit_waypoints
        assert c.phase == "reset_arm"


def test_arm_servo_damps_motion_without_changing_other_actuators():
    c = reset_controller()
    c.encoder_q = np.arange(31)
    c.encoder_v = np.arange(31)
    c.kp = np.ones(29)
    c.kd = np.ones(29)
    c.target = np.zeros(29)
    c.grip = 0.0
    c.arm = SimpleNamespace(arm_feedforward=np.zeros(7))
    c.observer.kinematic_data = SimpleNamespace(qfrc_bias=np.zeros(43))
    data = SimpleNamespace(qpos=np.zeros(31), qvel=np.zeros(31))
    data.qvel[22:29] = 0.1
    torque = c.servo(data)
    np.testing.assert_allclose(torque[:22], 0)
    np.testing.assert_allclose(torque[22:29], [-0.4] * 4 + [-0.16] * 3)
    assert np.dot(torque[22:29], data.qvel[22:29]) < 0
    np.testing.assert_allclose(torque[29:], 0)


def test_reset_waits_for_measured_joint_convergence():
    c = reset_controller()
    c.observer.proprio.qpos[29] = 0.2
    c.control(3.1)
    c.control(3.4)
    assert c.phase == "reset_arm" and not c.transit_waypoints
    c.observer.proprio.qpos[29] = 0
    c.control(3.5)
    assert not c.transit_waypoints
    c.control(3.72)
    assert c.phase == "back_out"


def test_reset_timeout_stops_job_without_claiming_success():
    c = reset_controller()
    c.target = np.zeros(29)
    c.control(10.1)
    assert c.phase == "failed" and c.finished and c.arm.failed
    assert c.state["transition_reason"] == "reset_timeout_or_contact"
    np.testing.assert_array_equal(c.command, 0)


def test_second_job_backs_out_with_held_arm_before_joint_reset():
    c = reset_controller()
    c.index = 0
    c.completed_jobs = ["a"]
    c.observers = {"b": c.observer}
    c.job_sensors = {"b": c.sensed}
    c.arm = SimpleNamespace(arm_feedforward=np.zeros(7), failed=False)
    c.reset_started = None
    c.target = np.full(29, 0.2)
    c.assign(
        SimpleNamespace(object_id="parcel_0_b", destination_id="receiver_0_b"), 4.0
    )
    assert c.phase == "back_out" and c.reset_started is None
    c.control(4.02)
    np.testing.assert_allclose(c.target[22:29], 0.2)
    assert c.pre_reset_backout and c.command[0] < 0
    c.sensed["root_position"][0] -= 0.30
    c.control(5.5)
    assert c.phase == "reset_arm" and c.reset_started == 5.5
    assert not c.transit_waypoints and c.backed_out_for_reset
