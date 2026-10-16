"""DAgger2 records actual externally capped actions, including the final step."""

from types import SimpleNamespace

import numpy as np
import pytest

from examples.robotics.warehouse.learned_policy import dagger2


def test_fixed_schedule_has_twenty_seconds_learner():
    assert sum(not dagger2.expert_step(i / 50) for i in range(4250)) == 1000


def test_finalization_requires_external_cap_and_keeps_labels_separate():
    c = object.__new__(dagger2.Dagger2Controller)
    c.command = np.zeros(3)
    c.target = np.ones(29)
    c.grip = 0.0
    c.arm = None
    c.finished = False
    c.dagger_steps = []
    request = dagger2.policy_action(c)
    c.pending = {
        "selected_requested_action": request,
        "previous_executed_arm_target": [0.0] * 7,
        "action": {"grip": 0.028},
    }
    with pytest.raises(ValueError, match="boundary"):
        c.finalize_pending()
    c.target[22:29] = 0.018
    c.finalize_pending()
    assert c.pending is None
    assert c.dagger_steps[0]["action"]["grip"] == 0.028
    assert c.dagger_steps[0]["executed_action"]["grip"] == 0
    np.testing.assert_array_equal(c.previous_executed, [0.018] * 7)


def test_physics_preflight_rejects_missing_common_boundary():
    args = SimpleNamespace(
        arm_slew_rad=0.018, bc_record=True, full_horizon=True, grasp_recovery=True
    )
    dagger2.require_boundary(args)
    args.arm_slew_rad = None
    with pytest.raises(ValueError):
        dagger2.require_boundary(args)


def test_shadow_uses_actual_feedback_and_memory_advances_on_expert_steps(monkeypatch):
    c = object.__new__(dagger2.Dagger2Controller)
    c.pending = None
    c.previous_executed = None
    c.dagger_steps = []
    c.target = np.ones(29) * 9
    c.action = np.ones(29)
    c.grip = 0.0
    c.sensed = {}
    c.shadow = SimpleNamespace(
        arm=SimpleNamespace(previous=None), phase="approach", state={}, finished=False
    )
    observed = {"joint_position": [0.0] * 31}
    monkeypatch.setattr(dagger2, "capture_observation", lambda _: observed)
    seen = []

    def expert(shadow, time):
        seen.append(shadow.arm.previous.copy())
        shadow.target = np.ones(29)
        shadow.action = np.ones(29)
        shadow.command = np.zeros(3)
        shadow.grip = 0.028
        shadow.arm.arm_feedforward = np.zeros(7)

    def learner(controller, time):
        controller.target = np.ones(29) * -1
        controller.command = np.zeros(3)
        controller.action = np.zeros(29)
        controller.grip = 0.0
        controller.arm = None
        controller.finished = False
        controller.state = {}
        controller.last_policy_action = {"learner_memory_advanced": True}

    monkeypatch.setattr(dagger2.RobotController, "control", expert)
    monkeypatch.setattr(dagger2.GeometryController, "control", learner)
    c.control(0.0)
    np.testing.assert_array_equal(seen[0], np.zeros(7))
    assert c.pending["learner_prediction"]["learner_memory_advanced"]
    assert c.pending["executed_source"] == "expert"
    assert not c.dagger_steps
    c.target[22:29] = 0.018  # external common actuator boundary
    c.control(0.02)
    np.testing.assert_array_equal(seen[1], np.ones(7) * 0.018)
    assert c.dagger_steps[0]["executed_action"]["grip"] == 0.028
    np.testing.assert_array_equal(c.pending["action"]["arm_target"], np.ones(7) * 0.036)
    c.target[22:29] = 0.036
    shadow_arm = c.shadow.arm
    c.control(10.0)
    assert c.arm is None
    assert c.shadow.arm is shadow_arm
    assert c.pending["executed_source"] == "learner"
    assert c.pending["observation"] == observed
    assert "expert_metadata" not in c.pending["observation"]
