"""Use learned commands in the same sensor, frozen walker and servo boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..physics import RobotController, observation
from ..physics_recording import policy_observation
from .model import PolicyRuntime


def capture_observation(controller):
    """Capture own virtual encoders/IMU, never shared world state."""
    return policy_observation(controller)


class LearnedController(RobotController):
    """No PickPlace invocation: all high-level actuator commands are learned.

    Common navigation/manipulation task labels are not supplied to the network.
    ``phase`` below is telemetry of predicted activation/termination only.
    """

    def __init__(
        self, *args, policy_directory, runtime_factory=PolicyRuntime, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.policy = runtime_factory(policy_directory)
        self.command = np.zeros(3)
        self.predicted_done = False
        self.last_policy_observation = None
        self.last_policy_action = None
        # Actual joint calibration limits are shared with the teacher IK policy.
        joints = [
            self.model.joint(i).id
            for i in range(self.model.njnt)
            if self.model.jnt_qposadr[i] in range(29, 36)
        ]
        self.arm_limits = self.model.jnt_range[joints].copy()

    def control(self, time):
        self.last_policy_observation = capture_observation(self)
        action, raw = self.policy.step(self.last_policy_observation)
        self.last_policy_action = action
        self.command = np.asarray(action["velocity_command"])
        self.predicted_done = bool(action["done"])
        # An unassigned robot has no task. This common dispatch guard is not a
        # teacher movement trajectory or a replacement for a failed learned arm.
        active = self.assignment is not None and action["arm_active"]
        if self.assignment is None:
            self.command[:] = 0
        proprio = self.observer.proprio
        obs = observation(proprio, self.action, self.command, self.default)
        if active:
            obs[33:40] = 0
            obs[62:69] = 0
        self.action = self.session.run(None, {"obs": obs[None]})[0][0]
        self.target = self.default + self.scale * self.action
        self.arm = None
        self.grip = 0.0
        if active:
            self.target[22:29] = np.clip(
                np.asarray(action["arm_target"]),
                self.arm_limits[:, 0] + 0.02,
                self.arm_limits[:, 1] - 0.02,
            )
            self.grip = 0.028 if action["grip"] >= 0.014 else 0.0
            self.arm = SimpleNamespace(
                arm_feedforward=np.asarray(action["arm_feedforward"]), failed=False
            )
        self.phase = "learned_terminal" if self.predicted_done else "learned_control"
        self.state = {
            "learned_policy": True,
            "predicted_done": self.predicted_done,
            "arm_active": bool(active),
            "raw_prediction": raw,
            "failed": False,
        }
        # Evaluation must run the predeclared physical horizon. Predicted done
        # is logged, never substituted for physical placement success.
        self.finished = False
