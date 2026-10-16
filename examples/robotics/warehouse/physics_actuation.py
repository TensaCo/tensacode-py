"""Optional common arm command slew bound, using only own encoder initialization."""

from __future__ import annotations

import numpy as np


class ArmSlewLimiter:
    """Bound each arm target increment at the shared 50 Hz control boundary.

    This constrains commanded joint motion, not actual joint velocity or force.
    It is not a collision guarantee, a learned-policy fallback or a task planner.
    """

    def __init__(self, maximum_delta_rad):
        if not np.isfinite(maximum_delta_rad) or maximum_delta_rad <= 0:
            raise ValueError("Arm slew bound must be positive and finite")
        self.maximum_delta_rad = float(maximum_delta_rad)
        self.previous = None
        self.steps = 0
        self.clipped_steps = 0
        self.clipped_joints = 0
        self.maximum_adjustment_rad = 0.0

    def apply(self, controller):
        requested = np.asarray(controller.target[22:29], dtype=float).copy()
        if requested.shape != (7,) or not np.isfinite(requested).all():
            raise ValueError("Invalid requested arm target")
        if self.previous is None:
            mapping = sorted(controller.observer._joint_map)
            sensed = controller.observer.proprio.qpos
            self.previous = np.array([sensed[row[0]] for row in mapping][22:29])
            if not np.isfinite(self.previous).all():
                raise ValueError("Invalid own encoder initialization")
        executed = self.previous + np.clip(
            requested - self.previous,
            -self.maximum_delta_rad,
            self.maximum_delta_rad,
        )
        adjusted = np.abs(requested - executed)
        clipped = adjusted > 1e-12
        controller.target[22:29] = executed
        self.previous = executed.copy()
        self.steps += 1
        self.clipped_steps += int(clipped.any())
        self.clipped_joints += int(clipped.sum())
        self.maximum_adjustment_rad = max(
            self.maximum_adjustment_rad, float(adjusted.max())
        )
        return {
            "requested_arm_target": requested.tolist(),
            "executed_arm_target": executed.tolist(),
            "clipped_joints": int(clipped.sum()),
            "maximum_adjustment_rad": float(adjusted.max()),
        }

    @property
    def summary(self):
        return {
            "maximum_delta_rad_per_20ms": self.maximum_delta_rad,
            "steps": self.steps,
            "clipped_steps": self.clipped_steps,
            "clipped_step_fraction": self.clipped_steps / max(self.steps, 1),
            "clipped_joint_fraction": self.clipped_joints / max(7 * self.steps, 1),
            "maximum_adjustment_rad": self.maximum_adjustment_rad,
            "initialization": "own sensed arm encoders; thereafter previous executed target",
            "scope": "command slew only, not actual velocity, force or collision certification",
        }
