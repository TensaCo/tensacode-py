"""Training-only sensor shadow expert; assistance is explicit and recorded."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path

from ..physics import RobotController
from ..physics_recording import policy_action
from .adapter import LearnedController, capture_observation
from .contract import SCHEMA
from .data import sha256


def expert_block(seed, robot_index, time, probability):
    """Fixed one-second blocks, independent of performance and sensor outcomes."""
    if not 0 <= probability <= 1:
        raise ValueError("Assistance probability outside [0,1]")
    key = f"{seed}:{robot_index}:{int((time + 1e-8) // 1)}".encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return value < probability


class DaggerController(LearnedController):
    def __init__(self, *args, policy_directory, **kwargs):
        super().__init__(*args, policy_directory=policy_directory, **kwargs)
        self.dagger_config = json.loads(
            (Path(policy_directory) / "dagger.json").read_text()
        )
        if not self.grasp_recovery:
            raise ValueError("DAgger requires the preregistered recovery expert")
        if self.dagger_config["round"] not in (1, 2, 3):
            raise ValueError("Unexpected DAgger round")
        probability = (0.5, 0.25, 0.0)[self.dagger_config["round"] - 1]
        if self.dagger_config["expert_probability"] != probability:
            raise ValueError("Assistance differs from frozen protocol")
        # A shallow copy deliberately shares only the real sensor/kinematic
        # observer and immutable calibration. It constructs no second renderer.
        self.shadow = copy.copy(self)
        self.shadow.state = {}
        self.dagger_steps = []
        self.dagger_weights_sha256 = sha256(Path(policy_directory) / "weights.pt")

    def assign(self, assignment, time):
        super().assign(assignment, time)
        RobotController.assign(self.shadow, assignment, time)

    def control(self, time):
        observed = capture_observation(self)
        self.shadow.sensed = self.sensed
        self.shadow.action = self.action.copy()
        self.shadow.grip = self.grip
        self.shadow.target = self.target.copy()
        if self.shadow.arm is not None:
            # IK regularization follows the actually executed prior joint target.
            # Cartesian plans advance only through measured tracking/contact gates.
            self.shadow.arm.previous = self.target[22:29].copy()
        RobotController.control(self.shadow, time)
        expert = copy.deepcopy(policy_action(self.shadow))
        # Always advance the learner memory, including assisted blocks.
        super().control(time)
        predicted = copy.deepcopy(self.last_policy_action)
        assistance = expert_block(
            self.dagger_config["seed"],
            self.index,
            time,
            self.dagger_config["expert_probability"],
        )
        if assistance:
            for name in ("target", "action", "command"):
                setattr(self, name, getattr(self.shadow, name).copy())
            self.grip = self.shadow.grip
            self.arm = self.shadow.arm
        # Shadow phase/done is supervision metadata, never an input feature.
        self.finished = False
        self.state["dagger_executed_source"] = "expert" if assistance else "learner"
        self.state["dagger_training_only"] = True
        self.dagger_steps.append(
            {
                "t": float(time),
                "observation": observed,
                "action": expert,
                "learner_prediction": predicted,
                "executed_action": copy.deepcopy(policy_action(self)),
                "executed_source": "expert" if assistance else "learner",
            }
        )

    def save_dagger(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as stream:
            json.dump(
                {
                    "schema": SCHEMA,
                    "robot_id": self.id,
                    "scenario": self.scenario,
                    "label_provenance": "sensor_only_shadow_expert_on_actual_visited_state",
                    "action_is_expert_label_not_executed_command": True,
                    "dagger": self.dagger_config,
                    "learner_weights_sha256": self.dagger_weights_sha256,
                    "steps": self.dagger_steps,
                },
                stream,
            )
