"""Isolated v6 development controller: sensed clearance gates arm reset."""

from __future__ import annotations

import math

import numpy as np
from warehouse.physics_jobs_controller import RepeatedJobController, observation
from warehouse.physics_reset_clearance import (
    ResetClearance,
    fixture_box,
    valid_pad_forces,
)


class ClearanceJobController(RepeatedJobController):
    def __init__(self, index, local, shared, offset, session, cfg, **kwargs):
        super().__init__(index, local, shared, offset, session, cfg, **kwargs)
        self.reset_clearance = ResetClearance(local)
        self.pickup_fixture_anchors = {}
        self.backout_start = None
        self.reset_fixture_job = None
        self.clearance_receipt = None

    def assign(self, assignment, time):
        previous_job = self.current_job
        super().assign(assignment, time)
        if self.current_job not in self.pickup_fixture_anchors:
            self.pickup_fixture_anchors[self.current_job] = np.asarray(
                self.sensed["object_position"]
            ).copy()
        if self.pre_reset_backout:
            self.backout_start = np.asarray(self.sensed["root_position"][:2]).copy()
            self.reset_fixture_job = previous_job

    def _clearance(self):
        # Nearby completed cell only; remote next-cell visibility is not required.
        sensed = self.job_sensors[self.reset_fixture_job]
        if not valid_pad_forces(self.sensed.get("pad_force_n")):
            return {"valid": False, "reason": "invalid_pad_forces"}
        age = sensed.get("destination_measurement_age_s")
        fresh = (
            age is not None
            and 0 <= age <= 0.3
            and sensed.get("destination_valid", False)
            and self.sensed.get("root_camera_valid", False)
        )
        if not fresh:
            return {"valid": False, "reason": "stale_nearby_fixture_or_root"}
        fixtures = [
            fixture_box(self.pickup_fixture_anchors[self.reset_fixture_job], -0.025),
            fixture_box(sensed["destination_position"], 0.050),
        ]
        result = self.reset_clearance.evaluate(
            self.observer.proprio.qpos, self.default[22:29], fixtures
        )
        result["static_pickup_anchor_scope"] = (
            "Initial RGB-D pickup observation; fixed-support prior"
        )
        result["destination_age_s"] = age
        return result

    def control(self, time):
        if self.transition_failed:
            return super().control(time)
        if (
            self.pre_reset_backout or self.reset_started is not None
        ) and not valid_pad_forces(self.sensed.get("pad_force_n")):
            self._fail_transition("invalid_pad_forces")
            self._standing_hold()
            return
        if self.reset_started is not None:
            self.clearance_receipt = self._clearance()
            if not self.clearance_receipt["valid"]:
                self._fail_transition("reset_clearance_lost")
                self.state["reset_clearance"] = self.clearance_receipt
                self._standing_hold()
                return
            super().control(time)
            self.state["reset_clearance"] = self.clearance_receipt
            return
        if not self.pre_reset_backout:
            return super().control(time)
        sensed = self.sensed
        if max(sensed.get("pad_force_n", [0, 0])) > 0.25:
            self._fail_transition("backout_unexpected_contact")
            self._standing_hold()
            return
        if time - self.transit_started > 25:
            self._fail_transition("transit_timeout")
            self.state["reset_clearance"] = self.clearance_receipt
            self._standing_hold()
            return
        root = np.asarray(sensed["root_position"][:2])
        goal = self.transit_waypoints[0]
        delta = goal - root
        progress = float(self.backout_start[0] - root[0])
        self.clearance_receipt = self._clearance()
        if progress >= 0.240 and self.clearance_receipt["valid"]:
            self.transit_waypoints.pop(0)
            self.pre_reset_backout = False
            self.backed_out_for_reset = True
            self.reset_started = time
            self.reset_joints = self.observer.proprio.qpos[29:36].copy()
            self.phase, self.phase_start = "reset_arm", time
            self.command = np.zeros(3)
            self.state = {
                "phase": self.phase,
                "job": self.current_job,
                "backward_progress_m": progress,
                "reset_clearance": self.clearance_receipt,
            }
            return
        quat = self.observer.proprio.qpos[3:7]
        yaw = math.atan2(
            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
        )
        rotation = np.array(
            [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
        )
        world_velocity = np.clip(delta * 1.5, [-0.35, -0.12], [0.35, 0.12])
        self.command = np.r_[rotation @ world_velocity, np.clip(-1.5 * yaw, -0.4, 0.4)]
        age = sensed.get("object_measurement_age_s")
        if (
            age is None
            or not 0 <= age < 1
            or not sensed.get("root_camera_valid", False)
        ):
            self.command[:] = 0
        obs = observation(
            self.observer.proprio, self.action, self.command, self.default
        )
        held_arm = self.target[22:29].copy()
        obs[33:40] = 0
        obs[62:69] = 0
        self.action = self.session.run(None, {"obs": obs[None]})[0][0]
        self.target = self.default + self.scale * self.action
        self.target[22:29] = held_arm
        self.grip = 0.0
        self.phase = "back_out"
        self.state = {
            "phase": self.phase,
            "job": self.current_job,
            "sensor_derived_waypoint": goal.tolist(),
            "waypoint_error_m": float(np.linalg.norm(delta)),
            "backward_progress_m": progress,
            "root_camera_valid": bool(sensed.get("root_camera_valid", False)),
            "reset_clearance": self.clearance_receipt,
            "command": self.command.tolist(),
        }
