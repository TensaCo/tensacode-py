"""Fleet scoring must use each robot's actual released-payload contacts."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.robotics.warehouse.validate import placement


def frames(valid=True, count=60):
    return [
        {
            "t": i * 0.02,
            "robots": {
                "r0": {
                    "phase": "complete",
                    "controller_state": {},
                    "grip_command": 0.0,
                    "evaluation": {
                        "object_position": [1.0, 0.0, 0.925],
                        "receiving_surface_xyz": [1.0, 0.0, 0.9],
                        "contacts": ["r0/receiving_marker"]
                        if valid
                        else ["r1/receiving_marker"],
                        "object_speed_m_s": 0.0,
                        "hand_clearance_m": 0.2,
                        "placement_xy_error_m": 0.0,
                        "streak": (i + 1) * 0.02 if valid else 0.0,
                        "settled_seconds": (i + 1) * 0.02 if valid else 0.0,
                        "minimum_root_height": 0.78,
                    },
                }
            },
        }
        for i in range(count)
    ]


def test_foreign_robot_receiver_cannot_satisfy_placement():
    assert placement(frames(), "r0")["success"]
    assert not placement(frames(False), "r0")["success"]


def test_short_contact_does_not_count_as_settled():
    assert not placement(frames(count=20), "r0")["success"]


def test_fabricated_reported_settling_is_rejected():
    rows = frames()
    altered = copy.deepcopy(rows)
    altered[10]["robots"]["r0"]["evaluation"]["contacts"] += ["r0/pad_a"]
    with pytest.raises(AssertionError):
        placement(altered, "r0")


def test_pose_error_must_match_recorded_position():
    rows = frames()
    rows[-1]["robots"]["r0"]["evaluation"]["placement_xy_error_m"] = 0.02
    with pytest.raises(AssertionError):
        placement(rows, "r0")
