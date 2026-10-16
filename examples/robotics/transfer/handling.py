"""Visually targeted pickup, transport, placement and withdrawal for G1.

The destination is a calibrated-color 20 cm support observed through RGB-D.
No simulator target pose or object/world label enters this controller.
"""

from __future__ import annotations

import mujoco
import numpy as np
from planning import GuardedManipulation
from scipy.spatial.transform import Rotation


class PickPlace(GuardedManipulation):
    def __init__(self, *args, load_compensation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.destination = None
        self.transport_started = False
        self.load_compensation = load_compensation
        self.payload_force_n = 0.0
        self.arm_feedforward = np.zeros(7)
        self.previous_hand = self.command.copy()
        self.supported_force_n = 0.0
        self.support_transfer_seconds = 0.0
        self.release_position = None

    def update_payload(self, estimated, sensors, dt):
        """Infer supported weight from tactile reactions in calibrated pad frames."""
        hand = estimated.site_xpos[self.ik.site]
        speed = float(np.linalg.norm(hand - self.previous_hand) / dt)
        self.previous_hand = hand.copy()
        self.ik.data.qpos[:] = estimated.qpos
        mujoco.mj_forward(self.ik.model, self.ik.data)
        forces = np.asarray(sensors.get("pad_force_local_n", np.zeros((2, 3))))
        bilateral = np.asarray(sensors.get("pad_force_n", [0.0, 0.0])) > 0.25
        world = sum(
            (
                self.ik.data.geom_xmat[geom].reshape(3, 3) @ force
                for geom, force in zip(self.ik.geoms[1:], forces)
            ),
            start=np.zeros(3),
        )
        self.supported_force_n = float(np.clip(-world[2], 0.0, 2.0))
        if (
            self.phase in ("lift", "hold", "locate_destination", "transport")
            and bilateral.all()
            and speed < 0.06
        ):
            measured = float(np.clip(-world[2], 0.0, 2.0))
            self.payload_force_n += (1 - np.exp(-dt / 0.4)) * (
                measured - self.payload_force_n
            )
        if self.phase in ("open", "withdraw", "complete", "failed"):
            self.payload_force_n *= np.exp(-dt / 0.10)
        jp = np.zeros((3, self.ik.model.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jacSite(self.ik.model, self.ik.data, jp, jr, self.ik.site)
        self.arm_feedforward = jp[:, self.ik.va].T @ np.array(
            [0.0, 0.0, self.payload_force_n]
        )
        if not self.load_compensation:
            self.arm_feedforward[:] = 0.0

    def step(self, estimated, sensors, time, dt=0.02):
        self.update_payload(estimated, sensors, dt)
        if not self.transport_started:
            result = super().step(estimated, sensors, time, dt)
            if self.phase == "hold" and not self.failed:
                self.transport_started = True
                self.transition("locate_destination", time, "payload_lifted")
                result[-1]["phase"] = self.phase
            result[-1]["payload_force_n"] = self.payload_force_n
            result[-1]["payload_mass_estimate_kg"] = self.payload_force_n / 9.81
            return result
        if self.phase == "failed":
            # Shared failure policy opens and attempts to retract; no hidden retry.
            return super().step(estimated, sensors, time, dt)
        hand = estimated.site_xpos[self.ik.site]
        rotation = estimated.site_xmat[self.ik.site].reshape(3, 3)
        elapsed = time - self.phase_started
        tactile = np.asarray(sensors.get("pad_force_n", [0.0, 0.0])) > 0.25
        destination = sensors.get("destination_position")
        age = sensors.get("destination_measurement_age_s")
        if (
            self.phase == "locate_destination"
            and destination is not None
            and age is not None
            and age < 1.0
        ):
            self.destination = np.asarray(destination).copy()
            self.ik.additional_supports = [self.destination.copy()]
            self.transition("transport", time, "fresh_rgbd_destination")
        target = self.command.copy()
        if self.destination is not None:
            target = (
                self.destination
                + {
                    "transport": [0, 0, 0.20],
                    "lower": [0, 0, 0.034],
                    "open": [0, 0, 0.034],
                    "withdraw": [-0.08, 0, 0.19],
                    "complete": [-0.08, 0, 0.19],
                    "locate_destination": [0, 0, 0.20],
                }[self.phase]
            )
        if self.phase == "open" and self.release_position is not None:
            target = self.release_position.copy()
        tracking = float(np.linalg.norm(hand - self.command))
        orientation = float(
            Rotation.from_matrix(self.rotation @ rotation.T).magnitude()
        )
        speed = 0.02 if self.phase == "lower" else 0.04
        tracking_limit = 0.010 if self.phase == "lower" else 0.012
        if tracking < tracking_limit and orientation < 0.15:
            delta = target - self.command
            self.command += delta * min(
                1.0, speed * dt / max(np.linalg.norm(delta), 1e-12)
            )
        error = float(np.linalg.norm(hand - target))
        # Withdrawal is a clearance maneuver, not a precision placement pose.
        tolerance = 0.012 if self.phase in ("withdraw", "complete") else 0.008
        self.stable = self.stable + dt if error < tolerance and orientation < 0.10 else 0.0
        # The held cube may settle within the fingers. Detect load transfer to
        # the visually located support instead of assuming a fixed grasp offset.
        support_transfer = (
            self.phase == "lower"
            and self.destination is not None
            and np.linalg.norm(hand[:2] - self.destination[:2]) < 0.025
            and 0.025 < hand[2] - self.destination[2] < 0.09
            and self.payload_force_n > 0.15
            and self.supported_force_n < 0.35 * self.payload_force_n
            and tactile.all()
        )
        self.support_transfer_seconds = (
            self.support_transfer_seconds + dt if support_transfer else 0.0
        )
        if self.phase == "lower" and self.support_transfer_seconds > 0.25:
            self.release_position = hand.copy()
            self.command = hand.copy()
            self.transition("open", time, "tactile_load_transferred_to_support")
        elif self.phase == "transport" and self.stable > 0.30:
            self.transition("lower", time, "above_receiving_surface")
        elif self.phase == "lower" and self.stable > 0.40:
            self.transition("open", time, "placement_pose_converged")
        elif self.phase == "open" and elapsed > 0.8 and not tactile.any():
            self.transition("withdraw", time, "tactile_release")
        elif self.phase == "withdraw" and self.stable > 0.3:
            self.transition("complete", time, "hand_withdrawn")
        elif self.phase == "complete" and elapsed > 2.0:
            self.done = True
        must_hold = self.phase in ("locate_destination", "transport") or (
            self.phase == "lower" and error > 0.025
        )
        if must_hold:
            self.contact_lost_seconds = (
                0.0 if tactile.all() else self.contact_lost_seconds + dt
            )
            if self.contact_lost_seconds > 0.30:
                self.failed = True
                self.transition("failed", time, "payload_contact_lost")
        if time - self.phase_started > 22.0 and self.phase not in (
            "complete",
            "failed",
        ):
            self.failed = True
            self.transition("failed", time, "handling_phase_timeout")
        grip = (
            0.028 if self.phase in ("locate_destination", "transport", "lower") else 0.0
        )
        q, diagnostics = self.ik.solve(
            estimated, self.command, self.rotation, self.previous, self.anchor, True
        )
        self.previous = q.copy()
        diagnostics.update(
            phase=self.phase,
            tracking_error_m=tracking,
            orientation_error_rad=orientation,
            target_error_m=error,
            transition_reason=self.transition_reason,
            pad_force_n=sensors.get("pad_force_n", [0.0, 0.0]),
            destination_position=None
            if self.destination is None
            else self.destination.tolist(),
            destination_measurement_age_s=age,
            failed=self.failed,
            payload_force_n=self.payload_force_n,
            supported_force_n=self.supported_force_n,
            support_transfer_seconds=self.support_transfer_seconds,
            payload_mass_estimate_kg=self.payload_force_n / 9.81,
            load_compensation=self.load_compensation,
        )
        return q, self.command.copy(), self.rotation.copy(), grip, diagnostics
