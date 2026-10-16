"""Sensor-driven, guarded Cartesian manipulation for the G1 example.

This is a local bounded controller with geometric clearance checks, not a global
collision-free planner. All geometry below is either robot calibration or an
explicit upright-cube/fixture prior positioned by the RGB-D object estimate.
"""

from __future__ import annotations

import itertools
import time

import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


class BoundedIK:
    """Continuous joint-limited damped least squares with sampled tool clearance."""

    def __init__(self, model, backend="scipy"):
        initialized = time.perf_counter()
        self.backend = backend
        self.compiled_solver = None
        self.fallback_count = 0
        if backend != "scipy":
            from box_qp import BoxQPSolver

            self.compiled_solver = BoxQPSolver(substrate=backend)
        self.model = model
        self.data = mujoco.MjData(model)
        names = [
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
        joints = [model.joint("right_" + name + "_joint").id for name in names]
        self.qa = np.array([model.jnt_qposadr[j] for j in joints])
        self.va = np.array([model.jnt_dofadr[j] for j in joints])
        self.limits = model.jnt_range[joints].copy() + [0.02, -0.02]
        self.site = model.site("grasp").id
        self.geoms = [model.geom(n).id for n in ("tool_palm", "pad_a", "pad_b")]
        self.corners = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
        self.initialization_seconds = time.perf_counter() - initialized

    def clearance(self, estimated, joints, anchor, allow_contact=False):
        self.data.qpos[:] = estimated.qpos
        self.data.qpos[self.qa] = joints
        mujoco.mj_forward(self.model, self.data)
        gaps = []
        for i, geom in enumerate(self.geoms):
            center = self.data.geom_xpos[geom]
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            # Separating-axis test between the calibrated tool OBB and upright SKU.
            half = self.model.geom_size[geom]
            axes = np.vstack(
                [
                    np.eye(3),
                    rotation.T,
                    np.cross(np.eye(3)[:, None, :], rotation.T[None, :, :]).reshape(
                        -1, 3
                    ),
                ]
            )
            norms = np.linalg.norm(axes, axis=1)
            axes = axes[norms > 1e-8] / norms[norms > 1e-8, None]
            separation = (
                np.abs(axes @ (center - anchor))
                - np.abs(axes @ rotation) @ half
                - np.abs(axes).sum(axis=1) * 0.025
            )
            if i == 0 or not allow_contact:
                gaps.append(max(separation))
            # Known square support footprint/shape; its pose derives from vision.
            points = center + (self.corners * half) @ rotation.T
            lo, hi = points.min(axis=0), points.max(axis=0)
            xy_gap = max(
                lo[0] - (anchor[0] + 0.10),
                anchor[0] - 0.10 - hi[0],
                lo[1] - (anchor[1] + 0.10),
                anchor[1] - 0.10 - hi[1],
            )
            gaps.append(max(xy_gap, lo[2] - (anchor[2] - 0.025)))
        return float(min(gaps))

    def solve(
        self, estimated, desired, rotation, previous, anchor, allow_contact=False
    ):
        solve_started = time.perf_counter()
        q = np.clip(previous, self.limits[:, 0], self.limits[:, 1])
        lower = np.maximum(self.limits[:, 0], q - 0.018)
        upper = np.minimum(self.limits[:, 1], q + 0.018)
        start = q.copy()
        for _ in range(4):
            self.data.qpos[:] = estimated.qpos
            self.data.qpos[self.qa] = q
            mujoco.mj_forward(self.model, self.data)
            poserr = desired - self.data.site_xpos[self.site]
            roterr = Rotation.from_matrix(
                rotation @ self.data.site_xmat[self.site].reshape(3, 3).T
            ).as_rotvec()
            jp = np.zeros((3, self.model.nv))
            jr = np.zeros_like(jp)
            mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
            jac = np.vstack([jp[:, self.va], 0.18 * jr[:, self.va], 0.025 * np.eye(7)])
            rhs = np.r_[poserr, 0.18 * roterr, np.zeros(7)]
            # Numerical epsilon keeps least-squares bounds strictly separated.
            lo, hi = lower - q - 1e-10, upper - q + 1e-10
            delta = None
            if self.compiled_solver is not None:
                candidate = self.compiled_solver.solve(jac[:6], rhs[:6], lo, hi)
                check = self.compiled_solver.last_diagnostics
                if (
                    np.isfinite(candidate).all()
                    and check.get("converged", False)
                    and check["bound_violation"] < 1e-9
                ):
                    delta = candidate
                else:
                    self.fallback_count += 1
            if delta is None:
                delta = lsq_linear(jac, rhs, bounds=(lo, hi), tol=1e-6, max_iter=20).x
            q = np.clip(q + delta, lower, upper)
            if np.linalg.norm(delta) < 1e-5:
                break
        current_gap = self.clearance(estimated, start, anchor, allow_contact)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0):
            candidate = start + fraction * (q - start)
            gaps = [
                self.clearance(
                    estimated, start + a * (candidate - start), anchor, allow_contact
                )
                for a in (0.25, 0.5, 0.75, 1.0)
            ]
            # A currently intersecting estimate may only move toward recovery.
            if min(gaps) >= min(0.002, current_gap - 1e-6):
                q = candidate
                accepted = True
                break
        if not accepted:
            q = start
        return q, {
            "clearance_m": self.clearance(estimated, q, anchor, allow_contact),
            "ik_step_rad": float(np.max(np.abs(q - start))),
            "clearance_limited": fraction < 1.0,
            "solver": self.backend,
            "solver_fallback_count": self.fallback_count,
            "solve_seconds": time.perf_counter() - solve_started,
            "initialization_seconds": self.initialization_seconds,
        }


class GuardedManipulation:
    """Feedback transitions and geometric waypoints; no simulator data interface."""

    def __init__(self, model, estimated, anchor, time, backend="scipy"):
        self.ik = BoundedIK(model, backend)
        self.anchor = np.asarray(anchor).copy()
        self.phase = "approach"
        self.started = self.phase_started = time
        self.events = []
        self.previous = estimated.qpos[self.ik.qa].copy()
        self.command = estimated.site_xpos[self.ik.site].copy()
        self.rotation = estimated.site_xmat[self.ik.site].reshape(3, 3).copy()
        self.initial_rotation = Rotation.from_matrix(self.rotation).as_rotvec()
        self.transition_reason = "start"
        self.stable = 0.0
        self.failed = False
        self.done = False
        self.recoveries = 0
        self.contact_lost_seconds = 0.0

    def transition(self, phase, time, reason):
        self.events.append(
            {"t": float(time), "from": self.phase, "to": phase, "reason": reason}
        )
        self.phase = phase
        self.phase_started = time
        self.stable = 0.0
        self.transition_reason = reason

    def step(self, estimated, sensors, time, dt=0.02):
        hand = estimated.site_xpos[self.ik.site]
        actual_rot = estimated.site_xmat[self.ik.site].reshape(3, 3)
        elapsed = time - self.phase_started
        contact = np.asarray(sensors.get("pad_force_n", [0.0, 0.0])) > 0.25
        target = (
            self.anchor
            + {
                "approach": [-0.16, 0, 0.14],
                "align": [0, 0, 0.10],
                "descend": [0, 0, 0.009],
                "close": [0, 0, 0.009],
                "lift": [-0.04, 0, 0.169],
                "hold": [-0.04, 0, 0.169],
                "release": [-0.04, 0, 0.169],
                "retract": [-0.10, 0, 0.14],
                "failed": [-0.10, 0, 0.14],
            }[self.phase]
        )
        # Targets move at 4 cm/s, but only while measured tracking is close.
        tracking = float(np.linalg.norm(hand - self.command))
        orient = float(Rotation.from_matrix(self.rotation @ actual_rot.T).magnitude())
        tracking_limit = 0.005 if self.phase == "descend" else 0.018
        speed = 0.02 if self.phase == "descend" else 0.04
        if tracking < tracking_limit and orient < 0.20:
            delta = target - self.command
            self.command += delta * min(
                1.0, speed * dt / max(np.linalg.norm(delta), 1e-12)
            )
        progress = min(elapsed / 3.0, 1.0) if self.phase == "approach" else 1.0
        self.rotation = Rotation.from_rotvec(
            (1 - progress) * self.initial_rotation
        ).as_matrix()
        error = float(np.linalg.norm(hand - target))
        age = sensors.get("object_measurement_age_s")
        fresh = age is not None and age < 1.0
        ready = error < 0.009 and orient < 0.10 and (self.phase != "align" or fresh)
        self.stable = self.stable + dt if ready else 0.0
        if self.phase in ("approach", "align") and self.stable > 0.25:
            self.transition(
                "align" if self.phase == "approach" else "descend",
                time,
                "pose_converged",
            )
        elif self.phase == "descend" and self.stable > 0.25:
            self.transition("close", time, "grasp_pose_converged")
        elif self.phase == "close" and contact.all() and elapsed > 0.5:
            self.transition("lift", time, "bilateral_tactile_contact")
        elif self.phase == "lift" and self.stable > 0.25:
            self.transition("hold", time, "lift_pose_converged")
        elif self.phase == "hold" and elapsed > 2.0 and contact.all():
            self.transition("release", time, "hold_complete")
        elif self.phase == "release" and elapsed > 2.0:
            self.done = True
        if (
            self.phase in ("approach", "align")
            or (self.phase == "descend" and error > 0.015)
        ) and contact.any():
            self.failed = True
            self.transition("failed", time, "unexpected_contact")
        if self.phase in ("lift", "hold"):
            self.contact_lost_seconds = (
                0.0 if contact.all() else self.contact_lost_seconds + dt
            )
            if self.contact_lost_seconds > 0.30:
                self.failed = True
                self.transition("failed", time, "tactile_contact_lost")
        if elapsed > 18.0 and self.phase not in ("release", "failed"):
            self.failed = True
            self.transition("failed", time, "phase_timeout")
        if self.phase == "failed" and time - self.phase_started > 2.0:
            self.done = True
        grip = 0.028 if self.phase in ("close", "lift", "hold") else 0.0
        joints, diagnostics = self.ik.solve(
            estimated,
            self.command,
            self.rotation,
            self.previous,
            self.anchor,
            self.phase in ("close", "lift", "hold", "release"),
        )
        self.previous = joints.copy()
        diagnostics.update(
            phase=self.phase,
            tracking_error_m=tracking,
            orientation_error_rad=orient,
            target_error_m=error,
            transition_reason=self.transition_reason,
            pad_force_n=sensors.get("pad_force_n", [0.0, 0.0]),
            failed=self.failed,
            object_measurement_age_s=age,
        )
        return joints, self.command.copy(), self.rotation.copy(), grip, diagnostics
