"""Teacher collection with an explicit pre-observation/post-command boundary."""

from __future__ import annotations

import copy
import gzip
import json

import numpy as np

SCHEMA = "tensorcode.warehouse.sensor_bc.v1"


def policy_observation(controller):
    observer = controller.observer
    proprio = observer.proprio
    mapping = sorted(observer._joint_map)
    return {
        "joint_position": [float(proprio.qpos[row[0]]) for row in mapping],
        "joint_velocity": [float(proprio.qvel[row[2]]) for row in mapping],
        "imu_quaternion": proprio.qpos[
            observer._root_qadr + 3 : observer._root_qadr + 7
        ].tolist(),
        "imu_angular_velocity": proprio.qvel[
            observer._local_root_dadr + 3 : observer._local_root_dadr + 6
        ].tolist(),
        "sensor_state": copy.deepcopy(controller.sensed),
        "task_active": controller.assignment is not None,
    }


def policy_action(controller):
    arm = controller.arm
    return {
        "velocity_command": controller.command.tolist(),
        "arm_target": controller.target[22:29].tolist(),
        "joint_target": controller.target.tolist(),
        "grip": float(controller.grip),
        "arm_feedforward": (
            np.zeros(7) if arm is None else arm.arm_feedforward
        ).tolist(),
        "arm_active": arm is not None,
        "done": bool(controller.finished),
    }


def save_episode(path, steps, *, robot_id, scenario, physical_success):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "schema": SCHEMA,
                "robot_id": robot_id,
                "scenario": scenario,
                "evaluation": {"physical_success": bool(physical_success)},
                "observation_boundary": "Before teacher action; own encoders/IMU and sensor posterior only. No time, phase, task progress or evaluator truth in observation.",
                "steps": steps,
            },
            stream,
        )
