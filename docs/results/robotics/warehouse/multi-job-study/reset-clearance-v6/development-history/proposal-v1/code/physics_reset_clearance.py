"""Conservative sampled arm-reset bounds in an independent calibrated FK model.

This is not continuous collision certification: 31 configurations, authored static
fixture dimensions, RGB-D anchors, and inflated robot/fixture AABBs are used.
"""

from __future__ import annotations

import mujoco
import numpy as np

RESET_CLEARANCE_PROTOCOL = {
    "samples": 31,
    "fixture_inflation_m": 0.020,
    "robot_inflation_m": 0.010,
    "required_separation_m": 0.010,
    "minimum_backwards_progress_m": 0.240,
    "maximum_lateral_velocity_m_s": 0.120,
    "maximum_fixture_age_s": 0.300,
    "maximum_pad_force_n": 0.250,
}


class ResetClearance:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        shoulder = model.body("right_shoulder_pitch_link").id
        self.geoms = []
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            while body and body != shoulder:
                body = int(model.body_parentid[body])
            if body == shoulder:
                self.geoms.append(geom)
        self.arm_q = np.array(
            [
                model.joint(f"right_{name}_joint").qposadr[0]
                for name in (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                )
            ]
        )
        self.encoder_q = [
            int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if model.jnt_type[j]
            in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ]
        self.root_q = int(model.joint("floating_base_joint").qposadr[0])

    def evaluate(self, own_qpos, target_joints, fixtures):
        """Fixtures are sensed centers/half extents; no simulation contacts/poses."""
        if not fixtures:
            return {"valid": False, "minimum_separation_m": None, "samples": 0}
        q = self.model.qpos0.copy()
        q[self.encoder_q] = np.asarray(own_qpos)[self.encoder_q]
        q[self.root_q : self.root_q + 7] = own_qpos[self.root_q : self.root_q + 7]
        initial = q[self.arm_q].copy()
        minimum = float("inf")
        for fraction in np.linspace(0, 1, RESET_CLEARANCE_PROTOCOL["samples"]):
            self.data.qpos[:] = q
            self.data.qpos[self.arm_q] = initial + fraction * (target_joints - initial)
            mujoco.mj_forward(self.model, self.data)
            for geom in self.geoms:
                rotation = self.data.geom_xmat[geom].reshape(3, 3)
                center = (
                    self.data.geom_xpos[geom]
                    + rotation @ self.model.geom_aabb[geom, :3]
                )
                half = np.abs(rotation) @ self.model.geom_aabb[geom, 3:] + 0.010
                for box in fixtures:
                    gap = np.max(
                        np.abs(center - box["center"]) - half - box["half"] - 0.020
                    )
                    minimum = min(minimum, float(gap))
        return {
            "valid": minimum >= 0.010,
            "minimum_separation_m": minimum,
            "samples": RESET_CLEARANCE_PROTOCOL["samples"],
            "arm_geometries": len(self.geoms),
            "fixtures": len(fixtures),
        }


def fixture_box(position, top_offset=0.0):
    """Known 20cm upright support, including 5cm placed-parcel envelope."""
    position = np.asarray(position, dtype=float)
    top = float(position[2] + top_offset)
    return {
        "center": np.array([position[0], position[1], top / 2]),
        "half": np.array([0.10, 0.10, top / 2]),
    }
