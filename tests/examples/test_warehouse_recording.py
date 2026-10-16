"""Policy collection preserves temporal alignment and excludes evaluator state."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "recording",
    Path(__file__).parents[2] / "examples/robotics/warehouse/physics_recording.py",
)
recording = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recording)


def test_observation_uses_only_proprio_and_copies_sensor_snapshot():
    sensed = {"root_position": [1, 2, 3], "pad_force_n": [0.2, 0.3]}
    observer = SimpleNamespace(
        proprio=SimpleNamespace(qpos=np.arange(45.0), qvel=np.arange(43.0)),
        _joint_map=[(7 + i, 100 + i, 6 + i, 200 + i) for i in range(31)],
        _root_qadr=0,
        _local_root_dadr=0,
    )
    controller = SimpleNamespace(
        observer=observer,
        sensed=sensed,
        assignment=object(),
        finished=False,
        phase="secret_teacher_phase",
        evaluation={"object_truth": [9, 9, 9]},
    )
    observation = recording.policy_observation(controller)
    sensed["root_position"][0] = 100
    assert observation["sensor_state"]["root_position"] == [1, 2, 3]
    assert len(observation["joint_position"]) == 31
    assert observation["joint_position"][0] == 7
    assert observation["imu_quaternion"] == [3, 4, 5, 6]
    assert "phase" not in observation and "evaluation" not in observation


def test_post_action_records_commands_not_realized_targets():
    controller = SimpleNamespace(
        command=np.array([0.1, 0.2, 0.3]),
        target=np.arange(29.0),
        grip=0.01,
        arm=SimpleNamespace(arm_feedforward=np.ones(7)),
        finished=True,
    )
    action = recording.policy_action(controller)
    assert action["arm_target"] == list(range(22, 29))
    assert action["velocity_command"] == [0.1, 0.2, 0.3]
    assert action["arm_active"] and action["done"]
