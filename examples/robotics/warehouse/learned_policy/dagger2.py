"""Separate training-only geometry DAgger qualification with delayed actuation audit."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np

from ..physics import RobotController
from ..physics_recording import policy_action
from .adapter import capture_observation
from .contract import SCHEMA
from .data import sha256
from .geometry_adapter import GeometryController

CAP = 0.018


def expert_step(time):
    """Ten-second warm start, then fixed 5s learner / 15s expert blocks."""
    return time < 10 - 1e-8 or (time - 10 + 1e-8) % 20 >= 5


def project_action(action, previous):
    result = copy.deepcopy(action)
    arm = np.asarray(previous) + np.clip(
        np.asarray(action["arm_target"]) - previous, -CAP, CAP
    )
    result["arm_target"] = arm.tolist()
    result["joint_target"][22:29] = arm.tolist()
    return result


def require_boundary(args):
    if not (
        args.arm_slew_rad == CAP
        and args.bc_record
        and args.full_horizon
        and args.grasp_recovery
    ):
        raise ValueError(
            "DAgger2 requires .018 cap, BC recording, full horizon and recovery"
        )


class Dagger2Controller(GeometryController):
    def __init__(self, *args, policy_directory, **kwargs):
        super().__init__(*args, policy_directory=policy_directory, **kwargs)
        directory = Path(policy_directory)
        self.dagger_config = json.loads((directory / "dagger2.json").read_text())
        if self.dagger_config["experiment"] != "geometry-dagger2-qualification-v1":
            raise ValueError("Unknown DAgger2 experiment")
        if not self.grasp_recovery:
            raise ValueError("Recovery expert required")
        self.shadow = copy.copy(self)
        self.shadow.state = {}
        self.dagger_steps = []
        self.pending = None
        self.previous_executed = None
        self.dagger_weights_sha256 = sha256(directory / "weights.pt")
        if self.dagger_weights_sha256 != self.dagger_config["initial_weights_sha256"]:
            raise ValueError("Initial learner differs from preregistration")

        for filename, key in (
            ("policy.json", "initial_policy_sha256"),
            ("kinematics.json", "kinematics_sha256"),
        ):
            if sha256(directory / filename) != self.dagger_config[key]:
                raise ValueError("Initial learner configuration differs from protocol")

    def assign(self, assignment, time):
        super().assign(assignment, time)
        RobotController.assign(self.shadow, assignment, time)

    def finalize_pending(self):
        """Called only after the external common limiter has executed."""
        if self.pending is None:
            return
        actual = copy.deepcopy(policy_action(self))
        expected = project_action(
            self.pending["selected_requested_action"],
            np.asarray(self.pending["previous_executed_arm_target"]),
        )
        if not np.allclose(
            actual["arm_target"], expected["arm_target"], atol=1e-12, rtol=0
        ):
            raise ValueError(
                "Missing or mismatched post-control common actuator boundary"
            )
        self.pending["executed_action"] = actual
        self.previous_executed = np.asarray(actual["arm_target"])
        self.dagger_steps.append(self.pending)
        self.pending = None

    def control(self, time):
        self.finalize_pending()
        observed = capture_observation(self)
        if self.previous_executed is None:
            self.previous_executed = np.asarray(observed["joint_position"][22:29])
        previous = self.previous_executed.copy()
        self.shadow.sensed = self.sensed
        self.shadow.action = self.action.copy()
        self.shadow.grip = self.grip
        self.shadow.target = self.target.copy()
        self.shadow.target[22:29] = previous
        if self.shadow.arm is not None:
            self.shadow.arm.previous = previous.copy()
        RobotController.control(self.shadow, time)
        requested = copy.deepcopy(policy_action(self.shadow))
        label = project_action(requested, previous)
        # Memory advances on every actual sensed state, including expert steps.
        super().control(time)
        prediction = copy.deepcopy(self.last_policy_action)
        assisted = expert_step(time)
        if assisted:
            for name in ("target", "action", "command"):
                setattr(self, name, getattr(self.shadow, name).copy())
            self.grip = self.shadow.grip
            self.arm = self.shadow.arm
        self.finished = False
        self.state["dagger_training_only"] = True
        self.state["dagger_executed_source"] = "expert" if assisted else "learner"
        self.pending = {
            "t": float(time),
            "observation": observed,
            "action": label,
            "expert_requested_action": requested,
            "learner_prediction": prediction,
            "selected_requested_action": copy.deepcopy(policy_action(self)),
            "previous_executed_arm_target": previous.tolist(),
            "executed_source": "expert" if assisted else "learner",
            "expert_metadata": {
                "phase": self.shadow.phase,
                "state": copy.deepcopy(self.shadow.state),
            },
        }

    def save_dagger(self, path):
        self.finalize_pending()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as stream:
            json.dump(
                {
                    "schema": SCHEMA,
                    "robot_id": self.id,
                    "scenario": self.scenario,
                    "dagger": self.dagger_config,
                    "learner_weights_sha256": self.dagger_weights_sha256,
                    "label_provenance": "sensor_only_shadow_requested_command_projected_from_actual_previous_target",
                    "execution_boundary": "recorded_after_external_common_arm_slew_limit",
                    "action_is_expert_label_not_executed_command": True,
                    "steps": self.dagger_steps,
                },
                stream,
            )
