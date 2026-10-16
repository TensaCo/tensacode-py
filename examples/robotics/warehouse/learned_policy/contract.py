"""Explicit observation/action boundary; evaluator and teacher phase are ignored."""

from __future__ import annotations

import numpy as np

SCHEMA = "tensorcode.warehouse.sensor_bc.v1"
ACTION_NAMES = (
    [f"velocity_command_{i}" for i in range(3)]
    + [f"arm_target_{i}" for i in range(7)]
    + ["grip"]
    + [f"arm_feedforward_{i}" for i in range(7)]
    + ["arm_active", "done"]
)
FEATURE_NAMES = (
    [f"joint_position_{i}" for i in range(31)]
    + [f"joint_velocity_{i}" for i in range(31)]
    + [f"imu_quaternion_{i}" for i in range(4)]
    + [f"imu_angular_velocity_{i}" for i in range(3)]
    + [f"{kind}_relative_{i}" for kind in ("object", "destination") for i in range(3)]
    + ["root_height", "object_present", "destination_present"]
    + [
        "object_age",
        "destination_age",
        "camera_valid",
        "destination_valid",
        "camera_confidence",
    ]
    + [f"pad_force_{i}" for i in range(2)]
    + [f"pad_local_force_{i}" for i in range(6)]
    + [f"stance_contact_{i}" for i in range(2)]
    + [f"object_covariance_{i}" for i in range(9)]
    + ["task_active"]
)


def vector(value, length, name):
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def features(observation):
    """Only allowlisted sensor fields enter the model; no phase/time/GT fields.

    Positions are relative to the sensor-estimated root. Sensor absence is an
    explicit mask and bounded age, rather than a privileged replacement value.
    Ideal joint encoders and IMU are shared with the algorithmic controller.
    """
    s = observation["sensor_state"]
    root = vector(s["root_position"], 3, "estimated root")
    points, present = [], []
    for kind in ("object", "destination"):
        point = s.get(f"{kind}_position")
        present.append(float(point is not None))
        points.extend(vector(point, 3, kind) - root if point is not None else [0] * 3)
    age = [
        min(10.0, max(0.0, float(s.get(f"{kind}_measurement_age_s") or 0)))
        if s.get(f"{kind}_measurement_age_s") is not None
        else 10.0
        for kind in ("object", "destination")
    ]
    x = np.concatenate(
        [
            vector(observation["joint_position"], 31, "joint_position"),
            vector(observation["joint_velocity"], 31, "joint_velocity"),
            vector(observation["imu_quaternion"], 4, "imu_quaternion"),
            vector(observation["imu_angular_velocity"], 3, "imu_angular_velocity"),
            points,
            [root[2], *present],
            age,
            [
                bool(s.get("camera_valid")),
                bool(s.get("destination_valid")),
                s.get("camera_confidence", 0),
            ],
            vector(s["pad_force_n"], 2, "pad_force_n"),
            vector(s["pad_force_local_n"], 6, "pad_force_local_n"),
            vector(s["stance_contacts"], 2, "stance_contacts"),
            vector(s["object_covariance"], 9, "object_covariance"),
            [bool(observation["task_active"])],
        ]
    ).astype(np.float32)
    return vector(x, len(FEATURE_NAMES), "features")


def actions(action):
    return vector(
        np.concatenate(
            [
                vector(action["velocity_command"], 3, "velocity_command"),
                vector(action["arm_target"], 7, "arm_target"),
                [action["grip"]],
                vector(action["arm_feedforward"], 7, "arm_feedforward"),
                [bool(action["arm_active"]), bool(action["done"])],
            ]
        ),
        len(ACTION_NAMES),
        "action",
    )


def decode_action(values):
    """Decode physical values; runtime applies recorded calibration bounds."""
    v = vector(values, len(ACTION_NAMES), "predicted action")
    return {
        "velocity_command": v[:3].tolist(),
        "arm_target": v[3:10].tolist(),
        "grip": float(v[10]),
        "arm_feedforward": v[11:18].tolist(),
        "arm_active": bool(v[18] >= 0.5),
        "done": bool(v[19] >= 0.5),
    }


# Explicit sensor semantics, fixed before training; not inferred from which
# features happen to be constant in a particular dataset.
UNIT_SCALE_FEATURES = frozenset(
    {
        "object_present",
        "destination_present",
        "camera_valid",
        "destination_valid",
        "camera_confidence",
        "stance_contact_0",
        "stance_contact_1",
        "task_active",
        "object_age",
        "destination_age",
    }
)


def input_statistics(train_features):
    """Fit continuous channels on TRAIN; preserve mask/probability/age units."""
    x = np.asarray(train_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Expected finite TRAIN feature matrix")
    mean, scale = x.mean(0), np.maximum(x.std(0), 0.01)
    for name in UNIT_SCALE_FEATURES:
        index = FEATURE_NAMES.index(name)
        mean[index], scale[index] = 0, 1
    return mean, scale
