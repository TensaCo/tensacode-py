"""Calibrated own-arm forward kinematics, with no scene/evaluator information."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import FEATURE_NAMES, features

GEOMETRY_NAMES = [
    f"hand_{reference}_{axis}"
    for reference in ("root", "object", "destination")
    for axis in ("x", "y", "z")
] + [f"hand_rotation_{i}{j}" for i in range(3) for j in range(3)]
GEOMETRY_FEATURE_NAMES = FEATURE_NAMES + GEOMETRY_NAMES


def export_calibration(model, path):
    """Extract only the robot's calibrated ancestry of the own grasp site."""
    site = model.site("grasp").id
    body = int(model.site_bodyid[site])
    ancestors = []
    while body:
        ancestors.append(body)
        body = int(model.body_parentid[body])
    ancestors.reverse()
    own = sorted(
        int(model.jnt_qposadr[j])
        for j in range(model.njnt)
        if int(model.jnt_type[j]) in (2, 3)
    )
    chain = []
    for body in ancestors:
        joints = []
        for j in range(
            int(model.body_jntadr[body]),
            int(model.body_jntadr[body] + model.body_jntnum[body]),
        ):
            kind = int(model.jnt_type[j])
            if kind == 0:
                continue
            if kind != 3:
                raise ValueError(
                    "Expected own arm ancestry to contain hinge joints only"
                )
            q = int(model.jnt_qposadr[j])
            joints.append(
                {
                    "input_index": own.index(q),
                    "axis": model.jnt_axis[j].tolist(),
                    "pivot": model.jnt_pos[j].tolist(),
                    "reference": float(model.qpos0[q]),
                }
            )
        chain.append(
            {
                "name": model.body(body).name,
                "position": model.body_pos[body].tolist(),
                "rotation": Rotation.from_quat(model.body_quat[body], scalar_first=True)
                .as_matrix()
                .tolist(),
                "joints": joints,
            }
        )
    calibration = {
        "version": "own-grasp-fk-v1",
        "root": "own IMU quaternion and sensor-estimated root translation",
        "chain": chain,
        "site_position": model.site_pos[site].tolist(),
        "site_rotation": Rotation.from_quat(model.site_quat[site], scalar_first=True)
        .as_matrix()
        .tolist(),
    }
    Path(path).write_text(json.dumps(calibration, indent=2) + "\n")
    return calibration


def forward(calibration, joint_positions, root_positions, root_quaternions):
    """Vectorized own-site pose; translation never comes from simulator truth."""
    joint_positions = np.atleast_2d(joint_positions)
    position = np.atleast_2d(root_positions).astype(float, copy=True)
    rotation = Rotation.from_quat(
        np.atleast_2d(root_quaternions), scalar_first=True
    ).as_matrix()
    for body in calibration["chain"][1:]:
        position += np.einsum("nij,j->ni", rotation, body["position"])
        rotation = rotation @ np.asarray(body["rotation"])
        for joint in body["joints"]:
            angle = joint_positions[:, joint["input_index"]] - joint["reference"]
            turn = Rotation.from_rotvec(
                angle[:, None] * np.asarray(joint["axis"])
            ).as_matrix()
            pivot = np.asarray(joint["pivot"])
            position += np.einsum(
                "nij,nj->ni", rotation, pivot - np.einsum("nij,j->ni", turn, pivot)
            )
            rotation = rotation @ turn
    position += np.einsum("nij,j->ni", rotation, calibration["site_position"])
    return position, rotation @ np.asarray(calibration["site_rotation"])


def geometry_features(calibration, observations):
    roots = np.asarray([o["sensor_state"]["root_position"] for o in observations])
    hand, rotation = forward(
        calibration,
        [o["joint_position"] for o in observations],
        roots,
        [o["imu_quaternion"] for o in observations],
    )
    pieces = [hand - roots]
    for key in ("object_position", "destination_position"):
        value = np.asarray(
            [
                o["sensor_state"].get(key)
                if o["sensor_state"].get(key) is not None
                else hand[i]
                for i, o in enumerate(observations)
            ]
        )
        pieces.append(hand - value)
    return np.concatenate([*pieces, rotation.reshape(-1, 9)], axis=1).astype(np.float32)


def extended_features(calibration, observations):
    return np.concatenate(
        [
            np.stack([features(o) for o in observations]),
            geometry_features(calibration, observations),
        ],
        axis=1,
    )


def residual_targets(targets, observations):
    result = targets.copy()
    result[:, 3:10] -= np.asarray([o["joint_position"][22:29] for o in observations])
    return result
