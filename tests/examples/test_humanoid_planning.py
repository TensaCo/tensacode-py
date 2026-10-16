"""Guarded manipulation contracts without robot assets or a rendering context."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation

path = Path(__file__).resolve().parents[2] / "examples/robotics/transfer/planning.py"
spec = spec_from_file_location("humanoid_planning_example", path)
planning = module_from_spec(spec)
spec.loader.exec_module(planning)


@pytest.fixture
def robot():
    names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    bodies = "".join(
        f'<body pos=".04 0 0"><joint name="right_{name}_joint" axis="{axis}" range="-1 1"/><geom size=".008" mass=".1"/>'
        for name, axis in zip(names, ["0 1 0", "1 0 0", "0 0 1", "0 1 0", "1 0 0", "0 1 0", "0 0 1"])
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>' + bodies +
        '<site name="grasp" pos=".05 0 0"/>'
        '<geom name="tool_palm" type="box" size=".02 .02 .02"/>'
        '<geom name="pad_a" type="box" pos=".05 .04 0" size=".01 .01 .02"/>'
        '<geom name="pad_b" type="box" pos=".05 -.04 0" size=".01 .01 .02"/>' +
        '</body>' * 7 + '</worldbody></mujoco>'
    )
    estimated = mujoco.MjData(model)
    mujoco.mj_forward(model, estimated)
    return model, estimated


@pytest.mark.parametrize("start", [0., .975, -.975, .98, -.98])
def test_ik_bounds_large_unreachable_commands(robot, start):
    model, estimated = robot
    solver = planning.BoundedIK(model)
    previous = np.full(7, start)
    estimated.qpos[solver.qa] = previous
    mujoco.mj_forward(model, estimated)
    command, diagnostics = solver.solve(
        estimated, np.array([10., -10., 10.]), Rotation.from_rotvec([2., 1., -1.]).as_matrix(),
        previous, np.array([100., 100., -100.]),
    )
    assert np.isfinite(command).all()
    assert np.max(np.abs(command - previous)) <= .018 + 1e-9
    assert np.all(command >= solver.limits[:, 0])
    assert np.all(command <= solver.limits[:, 1])
    assert diagnostics["ik_step_rad"] <= .018 + 1e-9


def test_so3_half_turn_remains_nonzero_in_solver(robot, monkeypatch):
    model, estimated = robot
    solver = planning.BoundedIK(model)
    captured = []

    def capture(jac, rhs, **kwargs):
        captured.append(rhs.copy())
        return SimpleNamespace(x=np.zeros(7))

    monkeypatch.setattr(planning, "lsq_linear", capture)
    solver.solve(estimated, estimated.site_xpos[solver.site].copy(),
                 Rotation.from_rotvec([np.pi - 1e-7, 0, 0]).as_matrix(),
                 estimated.qpos[solver.qa], np.array([100., 100., -100.]))
    assert np.linalg.norm(captured[0][3:6]) == pytest.approx(.18 * (np.pi - 1e-7))


def test_clearance_uses_estimated_fk_for_cube_and_support(robot):
    model, estimated = robot
    solver = planning.BoundedIK(model)
    joints = estimated.qpos[solver.qa].copy()
    palm = estimated.geom_xpos[model.geom("tool_palm").id].copy()
    assert solver.clearance(estimated, joints, palm) < 0
    # Palm below the inferred support surface, without touching the cube itself.
    assert solver.clearance(estimated, joints, palm + [0, 0, .10]) < 0
    assert solver.clearance(estimated, joints, palm + [0, 0, -.20]) > 0
    # A data object with only encoder/estimated-root qpos is sufficient.
    encoder_estimate = SimpleNamespace(qpos=estimated.qpos.copy())
    assert solver.clearance(encoder_estimate, joints, palm) < 0


def test_allow_contact_does_not_disable_palm_or_support_collisions(robot):
    model, estimated = robot
    solver = planning.BoundedIK(model)
    palm = estimated.geom_xpos[model.geom("tool_palm").id].copy()
    for anchor in (palm, palm + [0, 0, .10]):
        assert solver.clearance(estimated, estimated.qpos[solver.qa], anchor, allow_contact=True) < 0


@pytest.fixture
def guarded(robot, monkeypatch):
    model, estimated = robot
    controller = planning.GuardedManipulation(model, estimated, np.array([.5, 0, 0]), 0.)
    monkeypatch.setattr(controller.ik, "solve", lambda *args, **kwargs: (controller.previous.copy(), {}))
    return controller, estimated


@pytest.mark.parametrize("forces", [[0., 0.], [1., 0.], [0., 1.], [.25, .25]])
def test_close_cannot_lift_without_bilateral_tactile_contact(guarded, forces):
    controller, estimated = guarded
    controller.transition("close", 0., "test")
    controller.step(estimated, {"pad_force_n": forces}, 1.)
    assert controller.phase == "close"
    assert not any(event["to"] == "lift" for event in controller.events)


def test_bilateral_contact_is_required_before_lift(guarded):
    controller, estimated = guarded
    controller.transition("close", 0., "test")
    controller.step(estimated, {"pad_force_n": [1., 1.]}, .2)
    assert controller.phase == "close"
    controller.step(estimated, {"pad_force_n": [1., 1.]}, .6)
    assert controller.phase == "lift"
    assert controller.events[-1]["reason"] == "bilateral_tactile_contact"


@pytest.mark.parametrize("phase", ["approach", "align", "descend"])
def test_unexpected_early_contact_aborts(guarded, phase):
    controller, estimated = guarded
    controller.transition(phase, 0., "test")
    _, _, _, grip, diagnostics = controller.step(estimated, {"pad_force_n": [1., 0.]}, .1)
    assert controller.phase == "failed"
    assert controller.failed
    assert grip == 0
    assert diagnostics["transition_reason"] == "unexpected_contact"


def test_missing_contact_times_out_instead_of_lifting(guarded):
    controller, estimated = guarded
    controller.transition("close", 0., "test")
    controller.step(estimated, {}, 19.)
    assert controller.failed
    assert controller.phase == "failed"
    assert controller.events[-1]["reason"] == "phase_timeout"


def test_tactile_output_does_not_reveal_contact_counterpart(robot, monkeypatch):
    sensor_spec = spec_from_file_location("planning_sensor_boundary", path.with_name("sensors.py"))
    sensors = module_from_spec(sensor_spec)
    sensor_spec.loader.exec_module(sensors)
    model, _estimated = robot
    observer = sensors.SensorBelief.__new__(sensors.SensorBelief)
    observer.model = model
    observer._odometry = lambda _: None
    observer._last_camera = 0.
    observer.camera_period = .1
    observer.object_position = None
    observer.last_object_measurement_time = None
    observer.root_position = np.zeros(3)
    observer.camera_valid = False
    observer.camera_confidence = 0.
    observer.root_camera_valid = False
    observer.covariance = np.eye(3)
    observer.stance_contacts = [False, False]
    pad = model.geom("pad_a").id

    def force(_model, _data, _index, output):
        output[:] = [3., 4., 0., 0., 0., 0.]

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    first = observer.observe(SimpleNamespace(time=.01, geom_xmat={pad: np.eye(3).ravel()}, contact=[SimpleNamespace(geom1=pad, geom2=0, frame=np.eye(3).ravel())]))
    second = observer.observe(SimpleNamespace(time=.01, geom_xmat={pad: np.eye(3).ravel()}, contact=[SimpleNamespace(geom1=pad, geom2=1, frame=np.eye(3).ravel())]))
    assert first == second
    assert first["pad_force_n"] == [5., 0.]
    assert first["pad_force_local_n"] == [[-3., -4., 0.], [0., 0., 0.]]
    assert not any("counterpart" in key or "contact_position" in key for key in first)


@pytest.mark.parametrize("phase", ["lift", "hold"])
@pytest.mark.parametrize("forces", [[0., 0.], [1., 0.], [0., 1.]])
def test_sustained_loss_of_bilateral_contact_aborts(guarded, phase, forces):
    controller, estimated = guarded
    controller.transition(phase, 0., "test")
    for time in (.1, .2):
        controller.step(estimated, {"pad_force_n": forces}, time, dt=.1)
        assert not controller.failed
    _, _, _, grip, diagnostics = controller.step(
        estimated, {"pad_force_n": forces}, .31, dt=.11
    )
    assert controller.failed
    assert controller.phase == "failed"
    assert grip == 0
    assert diagnostics["transition_reason"] == "tactile_contact_lost"


def test_bilateral_contact_recovery_resets_loss_timer(guarded):
    controller, estimated = guarded
    controller.transition("lift", 0., "test")
    controller.step(estimated, {"pad_force_n": [1., 0.]}, .2, dt=.2)
    controller.step(estimated, {"pad_force_n": [1., 1.]}, .22)
    assert controller.contact_lost_seconds == 0
    controller.step(estimated, {"pad_force_n": [0., 1.]}, .42, dt=.2)
    assert not controller.failed


@pytest.mark.parametrize("age", [None, 1., 20.])
def test_stale_vision_blocks_descent_even_when_pose_converges(guarded, age):
    controller, estimated = guarded
    controller.anchor = estimated.site_xpos[controller.ik.site] - [0., 0., .10]
    controller.transition("align", 0., "test")
    for step in range(20):
        controller.step(estimated, {"object_measurement_age_s": age}, (step + 1) * .02)
    assert controller.phase == "align"
    assert controller.stable == 0
    # A fresh observation permits the same converged pose to descend.
    for step in range(20):
        controller.step(estimated, {"object_measurement_age_s": .1}, .4 + (step + 1) * .02)
        if controller.phase == "descend":
            break
    assert controller.phase == "descend"


def test_swept_clearance_checks_intermediate_joint_commands(robot, monkeypatch):
    model, estimated = robot
    solver = planning.BoundedIK(model)
    sampled = []

    def clearance(_estimated, joints, _anchor, _allow_contact=False):
        sampled.append(float(joints[0]))
        # An intermediate collision band with a clear final configuration.
        return -.01 if .003 < joints[0] < .015 else .01

    monkeypatch.setattr(solver, "clearance", clearance)
    monkeypatch.setattr(planning, "lsq_linear", lambda *args, **kwargs: SimpleNamespace(x=np.array([.018, 0, 0, 0, 0, 0, 0])))
    command, diagnostics = solver.solve(
        estimated, estimated.site_xpos[solver.site], np.eye(3),
        np.zeros(7), np.array([100., 100., -100.])
    )
    assert any(.003 < value < .015 for value in sampled)
    assert command[0] <= .003
    assert diagnostics["clearance_limited"]
    assert diagnostics["clearance_m"] >= 0


def test_hold_cannot_release_on_same_step_as_contact_loss_abort(guarded):
    controller, estimated = guarded
    controller.transition("hold", 0., "test")
    controller.contact_lost_seconds = .29
    controller.step(estimated, {"pad_force_n": [1., 0.]}, 2.1, dt=.02)
    assert controller.phase == "failed"
    assert controller.events[-1]["reason"] == "tactile_contact_lost"
    assert not any(event["to"] == "release" for event in controller.events)


def test_failure_waits_for_its_own_retraction_interval(guarded):
    controller, estimated = guarded
    controller.transition("close", 0., "test")
    controller.step(estimated, {}, 19.)
    assert controller.phase == "failed"
    assert not controller.done
    controller.step(estimated, {}, 20.9)
    assert not controller.done
    controller.step(estimated, {}, 21.1)
    assert controller.done
