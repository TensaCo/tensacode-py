"""Sensor-gated receiving-surface handling, without assets or simulator truth."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")

fixture_spec = spec_from_file_location(
    "handling_robot_fixture", Path(__file__).with_name("test_humanoid_planning.py")
)
fixture_module = module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture_module)
robot = fixture_module.robot
planning = fixture_module.planning


@pytest.fixture
def handling_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "planning", planning)
    path = (
        Path(__file__).resolve().parents[2] / "examples/robotics/transfer/handling.py"
    )
    spec = spec_from_file_location("humanoid_handling_example", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def handler(robot, handling_module, monkeypatch):
    model, estimated = robot
    controller = handling_module.PickPlace(model, estimated, np.array([0.5, 0, 0]), 0.0)
    controller.transport_started = True
    controller.transition("locate_destination", 0.0, "test")
    monkeypatch.setattr(
        controller.ik, "solve", lambda *args, **kwargs: (controller.previous.copy(), {})
    )
    # Expose only robot estimated FK and encoders, never object data or world labels.
    estimate = SimpleNamespace(
        qpos=estimated.qpos.copy(),
        site_xpos=estimated.site_xpos.copy(),
        site_xmat=estimated.site_xmat.copy(),
    )
    return controller, estimate


def observations(**kwargs):
    return {"pad_force_n": [1.0, 1.0], **kwargs}


@pytest.mark.parametrize(
    "destination,age",
    [
        (None, 0.0),
        ([1.0, 2.0, 3.0], None),
        ([1.0, 2.0, 3.0], 1.0),
        ([1.0, 2.0, 3.0], 10.0),
    ],
)
def test_missing_or_stale_destination_cannot_start_transport(handler, destination, age):
    controller, estimated = handler
    controller.step(
        estimated,
        observations(
            destination_position=destination, destination_measurement_age_s=age
        ),
        0.1,
    )
    assert controller.phase == "locate_destination"
    assert controller.destination is None
    assert controller.ik.additional_supports == []


def test_fresh_sensor_destination_becomes_copied_goal_and_support(handler):
    controller, estimated = handler
    destination = [1.0, -0.2, 0.8]
    _, _, _, _, diagnostics = controller.step(
        estimated,
        observations(
            destination_position=destination, destination_measurement_age_s=0.1
        ),
        0.1,
    )
    assert controller.phase == "transport"
    np.testing.assert_allclose(controller.destination, destination)
    np.testing.assert_allclose(controller.ik.additional_supports, [destination])
    assert diagnostics["transition_reason"] == "fresh_rgbd_destination"
    destination[0] = 999
    assert controller.destination[0] == 1.0


@pytest.mark.parametrize("forces", [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
def test_transport_contact_loss_aborts(handler, forces):
    controller, estimated = handler
    controller.destination = np.array([1.0, 0.0, 0.8])
    controller.transition("transport", 0.0, "test")
    controller.step(estimated, observations(pad_force_n=forces), 0.2, dt=0.2)
    assert not controller.failed
    _, _, _, grip, diagnostics = controller.step(
        estimated, observations(pad_force_n=forces), 0.4, dt=0.2
    )
    assert controller.failed
    assert controller.phase == "failed"
    assert grip == 0
    assert diagnostics["transition_reason"] == "payload_contact_lost"


def test_lower_waits_for_measured_pose_before_opening(handler):
    controller, estimated = handler
    controller.destination = np.array([1.0, 0.0, 0.8])
    controller.transition("lower", 0.0, "test")
    target = controller.destination + [0, 0, 0.034]
    controller.command = target.copy()  # Command convergence alone is insufficient.
    for step in range(30):
        controller.step(estimated, observations(), (step + 1) * 0.02)
    assert controller.phase == "lower"
    estimated.site_xpos[controller.ik.site] = target
    for step in range(30):
        controller.step(estimated, observations(), 0.6 + (step + 1) * 0.02)
        if controller.phase == "open":
            break
    assert controller.phase == "open"
    assert controller.events[-1]["reason"] == "placement_pose_converged"


@pytest.mark.parametrize("forces", [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
def test_open_requires_tactile_release_before_withdrawal(handler, forces):
    controller, estimated = handler
    controller.destination = np.array([1.0, 0.0, 0.8])
    controller.transition("open", 0.0, "test")
    controller.step(estimated, observations(pad_force_n=forces), 1.0)
    assert controller.phase == "open"
    controller.step(estimated, observations(pad_force_n=[0.0, 0.0]), 1.1)
    assert controller.phase == "withdraw"
    assert controller.events[-1]["reason"] == "tactile_release"


def test_withdraw_requires_measured_target_then_completion_dwell(handler):
    controller, estimated = handler
    controller.destination = np.array([1.0, 0.0, 0.8])
    target = controller.destination + [-0.08, 0.0, 0.19]
    controller.command = target.copy()
    controller.transition("withdraw", 0.0, "test")
    for step in range(20):
        controller.step(
            estimated, observations(pad_force_n=[0.0, 0.0]), (step + 1) * 0.02
        )
    assert controller.phase == "withdraw"
    estimated.site_xpos[controller.ik.site] = target
    for step in range(20):
        controller.step(
            estimated, observations(pad_force_n=[0.0, 0.0]), 0.4 + (step + 1) * 0.02
        )
        if controller.phase == "complete":
            break
    assert controller.phase == "complete"
    assert not controller.done
    controller.step(
        estimated, observations(pad_force_n=[0.0, 0.0]), controller.phase_started + 2.1
    )
    assert controller.done


def test_receiving_plane_clearance_uses_estimated_robot_geometry(robot):
    model, estimated = robot
    solver = planning.BoundedIK(model)
    proxy = SimpleNamespace(qpos=estimated.qpos.copy())
    joints = proxy.qpos[solver.qa]
    pickup = np.array([100.0, 100.0, -100.0])
    assert solver.clearance(proxy, joints, pickup, True) > 0
    palm = estimated.geom_xpos[model.geom("tool_palm").id].copy()
    solver.additional_supports = [palm + [0, 0, 0.01]]
    assert solver.clearance(proxy, joints, pickup, True) < 0
    solver.additional_supports = [palm + [0, 0, -0.20]]
    assert solver.clearance(proxy, joints, pickup, True) > 0


def payload_observations(load):
    return observations(pad_force_local_n=[[0.0, 0.0, -load / 2]] * 2)


@pytest.mark.parametrize(
    "load,expected", [(-3.0, 0.0), (0.6, 0.6), (1.6, 1.6), (20.0, 2.0)]
)
def test_tactile_load_estimate_converges_to_sensor_weight_with_bounds(
    handler, load, expected
):
    controller, estimated = handler
    controller.transition("hold", 0.0, "test")
    for _ in range(60):
        controller.update_payload(estimated, payload_observations(load), 0.1)
        assert 0 <= controller.payload_force_n <= 2.0
    assert controller.payload_force_n == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("phase", ["approach", "align", "descend", "close", "lower"])
def test_payload_estimator_does_not_learn_outside_supported_phases(handler, phase):
    controller, estimated = handler
    controller.transition(phase, 0.0, "test")
    controller.update_payload(estimated, payload_observations(1.2), 0.1)
    assert controller.payload_force_n == 0


@pytest.mark.parametrize("forces", [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
def test_payload_estimator_requires_bilateral_contact(handler, forces):
    controller, estimated = handler
    controller.transition("hold", 0.0, "test")
    controller.update_payload(
        estimated, {**payload_observations(1.2), "pad_force_n": forces}, 0.1
    )
    assert controller.payload_force_n == 0


def test_payload_estimator_ignores_dynamic_motion_then_learns_at_rest(handler):
    controller, estimated = handler
    controller.transition("transport", 0.0, "test")
    estimated.site_xpos[controller.ik.site, 0] += 0.02
    controller.update_payload(estimated, payload_observations(1.2), 0.1)
    assert controller.payload_force_n == 0
    controller.update_payload(estimated, payload_observations(1.2), 0.1)
    assert controller.payload_force_n > 0


@pytest.mark.parametrize("phase", ["open", "withdraw", "complete", "failed"])
def test_payload_compensation_decays_after_release_or_failure(handler, phase):
    controller, estimated = handler
    controller.payload_force_n = 1.5
    controller.transition(phase, 0.0, "test")
    controller.update_payload(estimated, payload_observations(1.5), 0.1)
    assert controller.payload_force_n == pytest.approx(1.5 / np.e)
    for _ in range(15):
        controller.update_payload(estimated, payload_observations(1.5), 0.1)
    assert controller.payload_force_n < 1e-6


def test_payload_feedforward_uses_estimated_jacobian_and_rotated_local_force(handler):
    controller, estimated = handler
    estimated.qpos[:] = 0.3
    probe = controller.ik.data
    probe.qpos[:] = estimated.qpos
    mujoco.mj_forward(controller.ik.model, probe)
    estimated.site_xpos[:] = probe.site_xpos
    estimated.site_xmat[:] = probe.site_xmat
    controller.previous_hand = estimated.site_xpos[controller.ik.site].copy()
    controller.transition("hold", 0.0, "test")
    # Local sensor vectors represent a known downward 1.2 N world load.
    local = [
        probe.geom_xmat[geom].reshape(3, 3).T @ np.array([0.0, 0.0, -0.6])
        for geom in controller.ik.geoms[1:]
    ]
    for _ in range(60):
        controller.update_payload(estimated, observations(pad_force_local_n=local), 0.1)
    assert controller.payload_force_n == pytest.approx(1.2, abs=1e-5)
    jp = np.zeros((3, controller.ik.model.nv))
    jr = np.zeros_like(jp)
    mujoco.mj_jacSite(controller.ik.model, probe, jp, jr, controller.ik.site)
    expected = jp[2, controller.ik.va] * controller.payload_force_n
    assert np.linalg.norm(expected) > 0.01
    np.testing.assert_allclose(controller.arm_feedforward, expected)


@pytest.fixture
def lowering(handler):
    controller, estimated = handler
    hand = estimated.site_xpos[controller.ik.site].copy()
    controller.destination = hand - [0.0, 0.0, 0.066]
    controller.payload_force_n = 1.0
    controller.transition("lower", 0.0, "test")
    return controller, estimated


def test_support_transfer_requires_sustained_load_drop_then_freezes_release_pose(
    lowering,
):
    controller, estimated = lowering
    hand = estimated.site_xpos[controller.ik.site].copy()
    for step in range(5):
        controller.step(
            estimated, payload_observations(0.2), (step + 1) * 0.05, dt=0.05
        )
        assert controller.phase == "lower"
    _, command, _, grip, diagnostics = controller.step(
        estimated, payload_observations(0.2), 0.30, dt=0.05
    )
    assert controller.phase == "open"
    assert diagnostics["transition_reason"] == "tactile_load_transferred_to_support"
    assert grip == 0
    np.testing.assert_allclose(controller.release_position, hand)
    np.testing.assert_allclose(command, hand)
    assert np.linalg.norm(hand - (controller.destination + [0, 0, 0.034])) > 0.02
    # Subsequent open commands hold the measured release pose, not the lower goal.
    for step in range(5):
        _, command, _, _, _ = controller.step(
            estimated, payload_observations(0.2), 0.32 + step * 0.02
        )
        np.testing.assert_allclose(command, hand)


@pytest.mark.parametrize(
    "condition",
    [
        "missing_destination",
        "far_xy",
        "too_high",
        "too_low",
        "small_initial_load",
        "no_load_drop",
        "unilateral",
    ],
)
def test_support_transfer_requires_all_sensor_and_geometry_gates(lowering, condition):
    controller, estimated = lowering
    signals = payload_observations(0.2)
    if condition == "missing_destination":
        controller.destination = None
    elif condition == "far_xy":
        controller.destination[0] += 0.04
    elif condition == "too_high":
        controller.destination[2] -= 0.05
    elif condition == "too_low":
        controller.destination[2] += 0.05
    elif condition == "small_initial_load":
        controller.payload_force_n = 0.15
        signals = payload_observations(0.0)
    elif condition == "no_load_drop":
        signals = payload_observations(0.5)
    elif condition == "unilateral":
        signals["pad_force_n"] = [1.0, 0.0]
    for step in range(14):
        controller.step(estimated, signals, (step + 1) * 0.02)
    assert controller.phase == "lower"
    assert controller.release_position is None
    assert controller.support_transfer_seconds == 0


def test_support_transfer_dwell_resets_when_weight_returns(lowering):
    controller, estimated = lowering
    controller.step(estimated, payload_observations(0.2), 0.2, dt=0.2)
    assert controller.support_transfer_seconds == pytest.approx(0.2)
    controller.step(estimated, payload_observations(1.0), 0.22)
    assert controller.support_transfer_seconds == 0
    controller.step(estimated, payload_observations(0.2), 0.42, dt=0.2)
    assert controller.phase == "lower"
    assert controller.support_transfer_seconds == pytest.approx(0.2)


def test_disabled_compensation_still_estimates_load_without_applying_torque(handler):
    controller, estimated = handler
    controller.load_compensation = False
    controller.transition("hold", 0.0, "test")
    for _ in range(60):
        controller.update_payload(estimated, payload_observations(1.2), 0.1)
    assert controller.payload_force_n == pytest.approx(1.2, abs=1e-5)
    assert controller.supported_force_n == pytest.approx(1.2)
    np.testing.assert_array_equal(controller.arm_feedforward, np.zeros(7))
    controller.load_compensation = True
    controller.update_payload(estimated, payload_observations(1.2), 0.1)
    assert np.linalg.norm(controller.arm_feedforward) > 0.01


@pytest.mark.parametrize(
    "phase,expected", [("withdraw", "complete"), ("lower", "lower")]
)
def test_ten_mm_pose_error_converges_only_for_withdrawal(handler, phase, expected):
    controller, estimated = handler
    hand = estimated.site_xpos[controller.ik.site].copy()
    offset = np.array([-0.08, 0.0, 0.19] if phase == "withdraw" else [0.0, 0.0, 0.034])
    controller.destination = hand - offset + [0.010, 0.0, 0.0]
    controller.transition(phase, 0.0, "test")
    for step in range(26):
        controller.step(estimated, payload_observations(1.0), (step + 1) * 0.02)
        if controller.phase == "complete":
            break
    assert controller.phase == expected
    if phase == "lower":
        assert controller.stable == 0
        assert controller.release_position is None
    else:
        assert controller.events[-1]["reason"] == "hand_withdrawn"
