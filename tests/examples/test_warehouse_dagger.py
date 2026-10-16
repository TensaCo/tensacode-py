"""DAgger attribution and actual-execution feedback, without a physics rollout."""

from types import SimpleNamespace

import numpy as np
import pytest

from examples.robotics.warehouse.learned_policy import dagger


def test_assistance_blocks_are_seeded_and_final_round_is_unassisted():
    assert dagger.expert_block(7, 0, 3.01, 0.5) == dagger.expert_block(7, 0, 3.99, 0.5)
    assert all(not dagger.expert_block(7, 0, t, 0) for t in range(85))
    assert all(dagger.expert_block(7, 0, t, 1) for t in range(85))
    with pytest.raises(ValueError):
        dagger.expert_block(7, 0, 0, 1.1)


@pytest.mark.parametrize("assisted", [False, True])
def test_shadow_receives_executed_feedback_and_records_separate_labels(
    monkeypatch, assisted
):
    controller = object.__new__(dagger.DaggerController)
    controller.index = 0
    controller.dagger_config = {"seed": 7, "expert_probability": 0.5}
    controller.action = np.ones(29) * 2
    controller.target = np.ones(29) * 3
    controller.command = np.ones(3) * 4
    controller.grip = 0.028
    controller.sensed = {"camera_valid": True}
    controller.shadow = SimpleNamespace(
        arm=SimpleNamespace(previous=np.ones(7) * 99), finished=False
    )
    controller.dagger_steps = []
    seen = {}

    def expert(shadow, time):
        seen["prior_action"] = shadow.action.copy()
        seen["prior_target"] = shadow.arm.previous.copy()
        seen["prior_grip"] = shadow.grip
        shadow.target = np.ones(29) * 5
        shadow.action = np.ones(29) * 6
        shadow.command = np.ones(3) * 7
        shadow.grip = 0.0
        shadow.arm.arm_feedforward = np.ones(7)

    def learner(self, time):
        self.target = np.ones(29) * 8
        self.action = np.ones(29) * 9
        self.command = np.ones(3) * 10
        self.grip = 0.028
        self.arm = None
        self.finished = False
        self.state = {}
        self.last_policy_action = {"sentinel": "learner"}

    monkeypatch.setattr(dagger, "capture_observation", lambda _: {"allowed": True})
    monkeypatch.setattr(dagger.RobotController, "control", expert)
    monkeypatch.setattr(dagger.LearnedController, "control", learner)
    monkeypatch.setattr(dagger, "expert_block", lambda *args: assisted)
    controller.control(1.0)
    assert np.all(seen["prior_action"] == 2)
    assert np.all(seen["prior_target"] == 3)
    assert seen["prior_grip"] == 0.028
    row = controller.dagger_steps[0]
    assert row["observation"] == {"allowed": True}
    assert row["action"]["arm_target"] == [5] * 7
    assert row["executed_action"]["arm_target"] == [5 if assisted else 8] * 7
    assert row["executed_source"] == ("expert" if assisted else "learner")
    assert controller.finished is False
