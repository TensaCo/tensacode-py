"""Sensor-only workcell belief: RGB-D object detection and contact/IMU odometry.

The simulator is read only through rendered RGB/depth, joint encoders, orientation
and binary foot contacts. Camera calibration, robot geometry, object half-size,
and the initial robot pose are supplied calibration priors, not live pose labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


class SensorBelief:
    def __init__(
        self,
        model,
        initialposition=(0, 0, 0.79),
        camera_period=0.1,
        calibration_bias=(0, 0, 0),
        width=320,
        height=240,
    ):
        self.model = model
        self.root_position = np.asarray(initialposition, dtype=float).copy()
        self.kinematic_data = mujoco.MjData(model)
        self._relative_data = mujoco.MjData(model)
        self._root_qadr = int(model.jnt_qposadr[model.joint("floating_base_joint").id])
        # Only scalar encoders are copied; object free-joint pose never crosses
        # this boundary. The only copied free-joint coordinates are IMU attitude.
        self._encoders = [
            int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if int(model.jnt_type[j])
            in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        ]
        self._feet = [model.body(s + "_ankle_roll_link").id for s in ("left", "right")]
        self._foot_geoms = [
            {
                g
                for g in range(model.ngeom)
                if (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                ).startswith(s + "_foot")
            }
            for s in ("left", "right")
        ]
        self._localizer = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "localizer"
        )
        self.root_camera_valid = False
        self.root_camera_updates = 0
        self._floor = model.geom("floor").id
        self._anchors = [None, None]
        self.stance_contacts = [False, False]
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = [1.1, -0.25, 0.65]
        self.camera.distance = 2.5
        self.camera.azimuth = 90
        self.camera.elevation = -78
        self.width, self.height = width, height
        self.camera_period = camera_period
        self.calibration_bias = np.asarray(calibration_bias, dtype=float)
        self._last_camera = -np.inf
        self._last_observe = None
        self.object_position = None
        self.covariance = np.eye(3) * 0.01
        self.camera_valid = False
        self.camera_confidence = 0.0
        self.latest_rgb = self.latest_depth = self.latest_mask = None
        self.first_rgb = self.first_depth = self.first_mask = None
        self.latest_measurement = None
        self.camera_calibration = None
        self.accepted_measurements = self.rejected_measurements = 0

    def _odometry(self, data):
        relative = self._relative_data
        root = self._root_qadr
        relative.qpos[root : root + 3] = 0.0
        relative.qpos[root + 3 : root + 7] = data.qpos[root + 3 : root + 7]
        relative.qpos[self._encoders] = data.qpos[self._encoders]
        mujoco.mj_forward(self.model, relative)
        contacts = [False, False]
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._floor in pair:
                for i, geoms in enumerate(self._foot_geoms):
                    contacts[i] |= bool(pair & geoms)
        candidates = []
        for i, foot in enumerate(self._feet):
            if contacts[i] and self._anchors[i] is not None:
                candidates.append(self._anchors[i] - relative.xpos[foot])
        if candidates:
            self.root_position = np.mean(candidates, axis=0)
        # Initialize newly planted feet only after updating from existing stance
        # feet, so a contact handoff does not discard one frame of translation.
        for i, foot in enumerate(self._feet):
            if not contacts[i]:
                self._anchors[i] = None
            elif self._anchors[i] is None:
                self._anchors[i] = self.root_position + relative.xpos[foot]
        # With no stance contact, hold translation and grow no fabricated motion.
        # This is a stance odometer, not a flight-state inertial navigation system.
        self.stance_contacts = contacts
        self.kinematic_data.qpos[:] = relative.qpos
        self.kinematic_data.qpos[root : root + 3] = self.root_position
        mujoco.mj_forward(self.model, self.kinematic_data)

    def _camera_measurement(self, data):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera)
        rgb = self.renderer.render().copy()
        cams = self.renderer.scene.camera
        position = np.mean([np.asarray(c.pos) for c in cams], axis=0)
        forward = np.mean([np.asarray(c.forward) for c in cams], axis=0)
        forward /= np.linalg.norm(forward)
        up = np.mean([np.asarray(c.up) for c in cams], axis=0)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        r, g, b = np.moveaxis(rgb.astype(float), -1, 0)
        mask = (
            (r > 65)
            & (r > 0.92 * g)
            & (g > 1.8 * b)
            & (g > 25)
            & np.isfinite(depth)
            & (depth > 0)
        )
        self.latest_rgb, self.latest_depth, self.latest_mask = rgb, depth, mask
        fovy = float(self.model.vis.global_.fovy)
        focal = 0.5 * self.height / np.tan(np.deg2rad(fovy) * 0.5)
        self.camera_calibration = {
            "position": position.tolist(),
            "forward": forward.tolist(),
            "up": up.tolist(),
            "right": right.tolist(),
            "focal_pixels": focal,
            "width": self.width,
            "height": self.height,
            "translation_bias": self.calibration_bias.tolist(),
        }

        def backproject(pixel_mask):
            rows, cols = np.where(pixel_mask)
            z = depth[rows, cols]
            x = (cols + 0.5 - self.width / 2) * z / focal
            y = (self.height / 2 - rows - 0.5) * z / focal
            return (
                position + z[:, None] * forward + x[:, None] * right + y[:, None] * up
            )

        self.root_camera_valid = False
        if self._localizer >= 0:
            marker_mask = (r < 0.3 * b) & (g < 1.08 * b) & (g > 55) & np.isfinite(depth)
            # Reject antialiased boundary pixels before fitting the known sphere.
            interior = marker_mask.copy()
            interior[1:-1, 1:-1] &= (
                marker_mask[:-2, 1:-1]
                & marker_mask[2:, 1:-1]
                & marker_mask[1:-1, :-2]
                & marker_mask[1:-1, 2:]
            )
            cloud = backproject(interior)
            if len(cloud) >= 6:
                origin = cloud.mean(axis=0)
                local = cloud - origin
                design = np.column_stack([2 * local, np.ones(len(local))])
                solution, _, rank, _ = np.linalg.lstsq(
                    design, (local * local).sum(axis=1), rcond=None
                )
                center = origin + solution[:3]
                radius = np.sqrt(max(0.0, solution[3] + solution[:3] @ solution[:3]))
                expected_radius = float(self.model.geom_size[self._localizer, 0])
                if (
                    rank == 4
                    and 0.65 * expected_radius < radius < 1.35 * expected_radius
                ):
                    rotation = np.empty(9)
                    root = self._root_qadr
                    mujoco.mju_quat2Mat(
                        rotation, self._relative_data.qpos[root + 3 : root + 7]
                    )
                    measured_root = (
                        center
                        - rotation.reshape(3, 3) @ self.model.geom_pos[self._localizer]
                    )
                    measured_root += self.calibration_bias
                    # Optical fiducial corrects stance drift; no world pose label.
                    correction = 0.8 * (measured_root - self.root_position)
                    if np.linalg.norm(correction) < 0.3:
                        self.root_position += correction
                        self._anchors = [
                            None if a is None else a + correction for a in self._anchors
                        ]
                        self.kinematic_data.qpos[root : root + 3] = self.root_position
                        mujoco.mj_forward(self.model, self.kinematic_data)
                        self.root_camera_valid = True
                        self.root_camera_updates += 1
        rows, cols = np.where(mask)
        if len(rows) < 6:
            return None, 0.0
        z = depth[rows, cols]
        x = (cols + 0.5 - self.width / 2) * z / focal
        y = (self.height / 2 - rows - 0.5) * z / focal
        points = position + z[:, None] * forward + x[:, None] * right + y[:, None] * up
        # Known upright cube size is a shape prior. The top face supplies its
        # height; lateral coordinates come only from visible RGB-D points.
        top_height = np.quantile(points[:, 2], 0.8)
        top = points[np.abs(points[:, 2] - top_height) < 0.006]
        if len(top) < 4:
            return None, 0.0
        measurement = np.median(top, axis=0)
        measurement[2] = top_height - 0.025
        measurement += self.calibration_bias
        confidence = min(1.0, len(top) / 30.0)
        if self.first_rgb is None:
            self.first_rgb, self.first_depth, self.first_mask = (
                rgb.copy(),
                depth.copy(),
                mask.copy(),
            )
        return measurement, confidence

    def observe(self, data):
        """Return a posterior from sensor channels; missing vision never uses truth."""
        self._odometry(data)
        now = float(data.time)
        if now - self._last_camera >= self.camera_period - 1e-9:
            elapsed = (
                self.camera_period
                if not np.isfinite(self._last_camera)
                else now - self._last_camera
            )
            self._last_camera = now
            measurement, confidence = self._camera_measurement(data)
            self.camera_valid = False
            self.camera_confidence = confidence
            self.covariance += np.eye(3) * (0.015**2) * max(elapsed, 1e-3)
            if measurement is not None:
                self.latest_measurement = measurement.copy()
                noise = np.eye(3) * (0.006**2) / max(confidence, 0.2)
                if self.object_position is None:
                    self.object_position = measurement.copy()
                    self.covariance = noise.copy()
                    self.camera_valid = True
                else:
                    innovation = measurement - self.object_position
                    total = self.covariance + noise
                    distance = float(innovation @ np.linalg.solve(total, innovation))
                    if distance <= 25.0:
                        gain = np.linalg.solve(total.T, self.covariance.T).T
                        self.object_position += gain @ innovation
                        self.covariance = (np.eye(3) - gain) @ self.covariance
                        self.camera_valid = True
                if self.camera_valid:
                    self.accepted_measurements += 1
                else:
                    self.rejected_measurements += 1
        return {
            "object_position": None
            if self.object_position is None
            else self.object_position.tolist(),
            "root_position": self.root_position.tolist(),
            "camera_valid": self.camera_valid,
            "camera_confidence": self.camera_confidence,
            "root_camera_valid": self.root_camera_valid,
            "object_covariance": self.covariance.tolist(),
            "stance_contacts": list(self.stance_contacts),
        }

    def save_evidence(self, directory):
        """Save observed pixels, calibration, and sensor-contract metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.first_rgb is not None:
            from PIL import Image

            Image.fromarray(self.first_rgb).save(directory / "sensor-rgb.png")
            np.savez_compressed(
                directory / "sensor-depth.npz",
                depth=self.first_depth,
                mask=self.first_mask,
            )
        (directory / "sensor-contract.json").write_text(
            json.dumps(
                {
                    "measurement": "Rendered RGB-D orange-color segmentation and calibrated pinhole backprojection",
                    "estimator": "Random-walk Gaussian object posterior with Mahalanobis innovation gate",
                    "odometry": "IMU orientation, scalar joint encoders, binary stance contacts and fixed-foot FK anchors",
                    "priors": [
                        "Initial root position",
                        "Calibrated fixed camera extrinsics/intrinsics",
                        "Robot kinematic model",
                        "Optional pelvis sphere fiducial local offset and radius",
                        "Upright 5 cm orange cube shape/color",
                    ],
                    "excluded": [
                        "Live object pose",
                        "Live root translation",
                        "Object segmentation IDs",
                        "Contact world positions",
                        "Simulator object velocities",
                    ],
                    "limitations": [
                        "Color-based perception; no semantic visual foundation model",
                        "Stance odometry drifts under foot slip and holds during flight",
                        "No explicit occlusion-motion model; posterior can become stale",
                    ],
                    "camera": self.camera_calibration,
                    "root_camera_updates": self.root_camera_updates,
                    "accepted_measurements": self.accepted_measurements,
                    "rejected_measurements": self.rejected_measurements,
                },
                indent=2,
            )
            + "\n"
        )

    def close(self):
        self.renderer.close()
