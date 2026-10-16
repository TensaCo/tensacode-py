"""Bounded tactile reacquisition for early unilateral contact during descent.

The only contact identity is the robot's own pad index. No contacted-object
identity, object pose, mass, or contact world position enters this controller.
"""

from __future__ import annotations

import numpy as np
from handling import PickPlace

RECOVERY_PROTOCOL = {
    "version": "tactile-reacquisition-v1",
    "maximum_attempts": 2,
    "retract_m": 0.06,
    "lateral_correction_per_attempt_m": 0.005,
    "maximum_contact_force_n": 8.0,
    "maximum_observation_age_s": 0.3,
    "retract_speed_m_s": 0.04,
    "retract_timeout_s": 8.0,
    "scope": "Early unilateral pad contact during visually targeted descent only; bounded retreat and pad-side correction, no counterpart identity",
}


def recovery_correction(pad_centers, pad_index, distance=0.005):
    """Shift the grasp center toward the touched pad, away from its inner edge."""
    centers = np.asarray(pad_centers, dtype=float)
    direction = centers[pad_index] - centers.mean(axis=0)
    direction[2] = 0
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError("Pad separation is not observable in the horizontal plane")
    return direction / norm * distance


class TactilePickPlace(PickPlace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_attempts = 0
        self.recovery_events = []
        self.recovery_target = None
        self.recovery_shift = None

    def _telemetry(self, diagnostics):
        diagnostics["recovery_attempts"] = self.recovery_attempts
        diagnostics["recovery_events"] = list(self.recovery_events)
        diagnostics["recovery_protocol"] = RECOVERY_PROTOCOL["version"]
        return diagnostics

    def step(self, estimated, sensors, time, dt=0.02):
        hand = estimated.site_xpos[self.ik.site].copy()
        forces = np.asarray(sensors.get("pad_force_n", [0, 0]), dtype=float)
        if self.phase == "reacquire_retract":
            elapsed = time - self.phase_started
            tracking = float(np.linalg.norm(hand - self.command))
            if tracking < 0.018:
                delta = self.recovery_target - self.command
                self.command += delta * min(
                    1, 0.04 * dt / max(np.linalg.norm(delta), 1e-12)
                )
            age = sensors.get("object_measurement_age_s")
            fresh = age is not None and age < 0.3
            error = float(np.linalg.norm(hand - self.recovery_target))
            if forces.max() > 8 or elapsed > 8:
                self.failed = True
                self.transition("failed", time, "reacquisition_force_or_timeout")
                return super().step(estimated, sensors, time, dt)
            if error < 0.008 and forces.max() < 0.25 and elapsed > 0.3 and fresh:
                self.anchor += self.recovery_shift
                self.command = hand.copy()
                self.transition("align", time, "tactile_reacquisition_clear")
                return super().step(estimated, sensors, time, dt)
            joints, diagnostics = self.ik.solve(
                estimated,
                self.command,
                self.rotation,
                self.previous,
                self.anchor,
                False,
            )
            self.previous = joints.copy()
            diagnostics.update(
                phase=self.phase,
                tracking_error_m=tracking,
                target_error_m=error,
                transition_reason=self.transition_reason,
                pad_force_n=forces.tolist(),
                failed=False,
            )
            return (
                joints,
                self.command.copy(),
                self.rotation.copy(),
                0.0,
                self._telemetry(diagnostics),
            )
        before = self.phase
        result = super().step(estimated, sensors, time, dt)
        age = sensors.get("object_measurement_age_s")
        unilateral = np.count_nonzero(forces > 0.25) == 1
        if (
            before == "descend"
            and self.failed
            and self.transition_reason == "unexpected_contact"
            and unilateral
            and forces.max() <= 8
            and age is not None
            and age < 0.3
            and self.recovery_attempts < 2
        ):
            pad = int(np.argmax(forces))
            centers = estimated.geom_xpos[self.ik.geoms[1:]].copy()
            self.recovery_shift = recovery_correction(centers, pad)
            self.recovery_target = hand + [0, 0, 0.06]
            self.command = hand.copy()
            self.previous = estimated.qpos[self.ik.qa].copy()
            self.failed = self.done = False
            self.recovery_attempts += 1
            self.recovery_events.append(
                {
                    "time": float(time),
                    "pad_index": pad,
                    "force_n": float(forces[pad]),
                    "correction_xyz": self.recovery_shift.tolist(),
                    "attempt": self.recovery_attempts,
                }
            )
            self.transition("reacquire_retract", time, "early_unilateral_contact")
            diagnostics = result[-1]
            diagnostics.update(
                phase=self.phase, failed=False, transition_reason=self.transition_reason
            )
            return (
                self.previous.copy(),
                self.command.copy(),
                self.rotation.copy(),
                0.0,
                self._telemetry(diagnostics),
            )
        self._telemetry(result[-1])
        return result
