"""Repeated sensor-selected pickup skills in one robot's calibrated workcell."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from warehouse.physics import RobotController, observation
from warehouse.physics_jobs_sensors import JobSensorBelief
from warehouse.representation import Entity

ARM_SERVO_PROTOCOL = {
    "version": "repeated-job-arm-servo-v3",
    "frequency_hz": 500,
    "kp": [60, 60, 60, 60, 24, 24, 24],
    "kd": [4, 4, 4, 4, 1.6, 1.6, 1.6],
    "scope": "Own-joint disturbance rejection; existing force limits and tactile retry bounds unchanged",
}
TRANSITION_PROTOCOL = {
    "version": "repeated-job-transitions-v5",
    "reset_order": "Hold withdrawn arm; back body away30cm; reset arm; traverse",
    "reset_max_joint_error_rad": 0.08,
    "reset_max_joint_speed_rad_s": 0.3,
    "reset_stable_seconds": 0.2,
    "reset_timeout_seconds": 10,
    "reset_contact_abort_n": 0.25,
    "transit_timeout_seconds": 25,
    "maximum_object_age_seconds": 1,
    "require_optical_root_at_transit_completion": True,
}


class RepeatedJobController(RobotController):
    """Select a fresh sensed parcel, reset the arm, then walk/reach again.

    Each observer continuously updates its own color-conditioned posterior.
    Completed parcels remain in the same physical world throughout later jobs.
    """

    def __init__(self, index, local, shared, offset, session, cfg, **kwargs):
        super().__init__(index, local, shared, offset, session, cfg, **kwargs)
        self.observer.close()
        self.observers = {
            job: JobSensorBelief(
                local,
                shared,
                self.id + "/",
                offset,
                job=job,
                sensor_interventions=self.scenario["sensors"],
                noise_seed=self.scenario["seed"] + index,
            )
            for job in ("a", "b")
        }
        self.current_job = "a"
        self.observer = self.observers["a"]
        self.job_sensors = {}
        self.completed_jobs = []
        self.reset_started = None
        self.reset_joints = None
        self.transit_waypoints = []
        self.reset_stable_since = None
        self.transit_started = None
        self.pre_reset_backout = False
        self.backed_out_for_reset = False
        self.transition_failed = False

    def observe(self, data):
        self.job_sensors = {
            job: observer.observe(data) for job, observer in self.observers.items()
        }
        self.observer = self.observers[self.current_job]
        self.sensed = self.job_sensors[self.current_job]
        return self.sensed

    def servo(self, shared_data):
        torque = super().servo(shared_data)
        if self.arm is not None:
            # Only own scalar encoders are read here. The Cartesian planner,
            # contact guards and actuator force limits remain unchanged.
            q = shared_data.qpos[self.encoder_q][22:29]
            v = shared_data.qvel[self.encoder_v][22:29]
            torque[22:29] = (
                np.asarray(ARM_SERVO_PROTOCOL["kp"]) * (self.target[22:29] - q)
                - np.asarray(ARM_SERVO_PROTOCOL["kd"]) * v
                + self.observer.kinematic_data.qfrc_bias[28:35]
                + self.arm.arm_feedforward
            )
        return torque

    def scene_entities(self, now):
        entities = []
        for job, sensed in self.job_sensors.items():
            for kind, key, agekey, color in [
                (
                    "parcel",
                    "object_position",
                    "object_measurement_age_s",
                    "orange" if job == "a" else "green",
                ),
                (
                    "receiver",
                    "destination_position",
                    "destination_measurement_age_s",
                    "magenta" if job == "a" else "blue",
                ),
            ]:
                if sensed[key] is not None and sensed[agekey] is not None:
                    entities.append(
                        Entity(
                            f"{kind}_{self.index}_{job}",
                            "parcel" if kind == "parcel" else "station",
                            {"color": color, "lane": str(self.index), "job": job},
                            tuple(sensed[key]),
                            now - sensed[agekey],
                            f"{self.id}/{job}/shared_rgbd",
                            0.95,
                        )
                    )
        return entities

    def assign(self, assignment, time):
        job = assignment.object_id.rsplit("_", 1)[-1]
        if (
            job not in self.observers
            or assignment.object_id != f"parcel_{self.index}_{job}"
            or assignment.destination_id != f"receiver_{self.index}_{job}"
            or job in self.completed_jobs
        ):
            raise ValueError("Only a fresh calibrated own-workcell job is supported")
        self.current_job = job
        self.observer = self.observers[job]
        self.sensed = self.job_sensors[job]
        self.assignment = assignment
        self.finished = False
        self.reset_stable_since = None
        self.transition_failed = False
        self.object_goal = None
        self.state = {}
        if self.arm is not None:
            root = np.asarray(self.sensed["root_position"][:2])
            self.transit_waypoints = [root + [-0.30, 0.0]]
            self.transit_started = time
            self.pre_reset_backout = True
            self.phase = "back_out"
        else:
            self.phase = "walk"
        self.phase_start = time

    def release_assignment(self):
        if not self.finished or self.phase != "complete" or self.arm.failed:
            return False
        self.completed_jobs.append(self.current_job)
        self.assignment = None
        self.finished = False
        return True

    def control(self, time):
        if self.transition_failed:
            self._standing_hold()
            return
        if self.transit_waypoints:
            if (
                self.pre_reset_backout
                and max(self.sensed.get("pad_force_n", [0, 0])) > 0.25
            ):
                self._fail_transition("backout_unexpected_contact")
                self._standing_hold()
                return
            if time - self.transit_started > 25:
                self._fail_transition("transit_timeout")
                self._standing_hold()
                return
            root = np.asarray(self.sensed["root_position"][:2])
            goal = self.transit_waypoints[0]
            delta = goal - root
            quaternion = self.observer.proprio.qpos[3:7]
            yaw = math.atan2(
                2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
                1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2),
            )
            rotation = np.array(
                [[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]
            )
            self.command = np.r_[
                np.clip(rotation @ delta * 1.5, -0.35, 0.35),
                np.clip(-1.5 * yaw, -0.4, 0.4),
            ]
            age = self.sensed.get("object_measurement_age_s")
            fresh = age is not None and age < 1.0
            if not fresh:
                self.command[:] = 0
            obs = observation(
                self.observer.proprio, self.action, self.command, self.default
            )
            held_arm = self.target[22:29].copy()
            if self.pre_reset_backout:
                obs[33:40] = 0
                obs[62:69] = 0
            self.action = self.session.run(None, {"obs": obs[None]})[0][0]
            self.target = self.default + self.scale * self.action
            if self.pre_reset_backout:
                self.target[22:29] = held_arm
            self.grip = 0.0
            self.phase = (
                "back_out"
                if self.pre_reset_backout or len(self.transit_waypoints) == 2
                else "traverse"
            )
            self.state = {
                "phase": self.phase,
                "job": self.current_job,
                "sensor_derived_waypoint": goal.tolist(),
                "waypoint_error_m": float(np.linalg.norm(delta)),
                "object_measurement_age_s": age,
                "root_camera_valid": bool(self.sensed.get("root_camera_valid", False)),
            }
            if (
                np.linalg.norm(delta) < 0.06
                and fresh
                and (
                    len(self.transit_waypoints) > 1
                    or self.pre_reset_backout
                    or self.sensed.get("root_camera_valid", False)
                )
            ):
                self.transit_waypoints.pop(0)
                if self.pre_reset_backout:
                    self.pre_reset_backout = False
                    self.backed_out_for_reset = True
                    self.reset_started = time
                    self.reset_joints = self.observer.proprio.qpos[29:36].copy()
                    self.phase, self.phase_start = "reset_arm", time
                    return
                if not self.transit_waypoints:
                    self.phase, self.phase_start = "walk", time
            return
        if self.reset_started is not None:
            if (
                time - self.reset_started > 10
                or max(self.sensed.get("pad_force_n", [0, 0])) > 0.25
            ):
                self._fail_transition("reset_timeout_or_contact")
                self._standing_hold()
                return
            fraction = min(1.0, (time - self.reset_started) / 3.0)
            self.command = np.zeros(3)
            proprio = self.observer.proprio
            obs = observation(proprio, self.action, self.command, self.default)
            obs[33:40] = 0
            obs[62:69] = 0
            self.action = self.session.run(None, {"obs": obs[None]})[0][0]
            self.target = self.default + self.scale * self.action
            self.target[22:29] = (
                1 - fraction
            ) * self.reset_joints + fraction * self.default[22:29]
            self.arm = SimpleNamespace(arm_feedforward=np.zeros(7), failed=False)
            self.grip = 0.0
            self.phase = "reset_arm"
            self.state = {
                "phase": "reset_arm",
                "reset_progress": fraction,
                "job": self.current_job,
            }
            age = self.sensed.get("object_measurement_age_s")
            joint_error = float(
                np.max(np.abs(proprio.qpos[29:36] - self.default[22:29]))
            )
            joint_speed = float(np.max(np.abs(proprio.qvel[28:35])))
            ready = fraction >= 1 and joint_error < 0.08 and joint_speed < 0.3
            if not ready:
                self.reset_stable_since = None
            elif self.reset_stable_since is None:
                self.reset_stable_since = time
            self.state.update(
                reset_joint_error_rad=joint_error, reset_joint_speed_rad_s=joint_speed
            )
            stable = (
                self.reset_stable_since is not None
                and time - self.reset_stable_since >= 0.2
            )
            if stable and age is not None and age < 1:
                root = np.asarray(self.sensed["root_position"][:2])
                next_staging = np.asarray(self.sensed["object_position"][:2]) + [
                    -0.25,
                    0.25,
                ]
                if self.backed_out_for_reset:
                    self.transit_waypoints = [np.array([root[0], next_staging[1]])]
                else:
                    back_x = min(root[0], next_staging[0]) - 0.30
                    self.transit_waypoints = [
                        np.array([back_x, root[1]]),
                        np.array([back_x, next_staging[1]]),
                    ]
                self.reset_started = None
                self.transit_started = time
                self.arm = None
                self.phase = "back_out"
                self.phase_start = time
            return
        # A completed robot holds its last arm command until another assignment.
        if self.assignment is None and self.completed_jobs:
            self.command = np.zeros(3)
            proprio = self.observer.proprio
            obs = observation(proprio, self.action, self.command, self.default)
            obs[33:40] = 0
            obs[62:69] = 0
            previous = self.target[22:29].copy()
            self.action = self.session.run(None, {"obs": obs[None]})[0][0]
            self.target = self.default + self.scale * self.action
            self.target[22:29] = previous
            self.grip = 0.0
            return
        super().control(time)
        self.state["job"] = self.current_job

    def _fail_transition(self, reason):
        self.transition_failed = True
        self.finished = True
        self.phase = "failed"
        self.state = {
            "phase": "failed",
            "job": self.current_job,
            "transition_reason": reason,
        }
        self.arm = SimpleNamespace(arm_feedforward=np.zeros(7), failed=True)
        self.target[22:29] = self.observer.proprio.qpos[29:36]
        self.grip = 0.0

    def _standing_hold(self):
        self.command = np.zeros(3)
        obs = observation(
            self.observer.proprio, self.action, self.command, self.default
        )
        obs[33:40] = 0
        obs[62:69] = 0
        previous = self.target[22:29].copy()
        self.action = self.session.run(None, {"obs": obs[None]})[0][0]
        self.target = self.default + self.scale * self.action
        self.target[22:29] = previous
