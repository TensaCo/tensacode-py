"""Physical scoring must not depend on authored phases or completion claims."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.robotics.warehouse.validate_v4 import physical_scores


def records(validity):
    rows = []
    streak = longest = 0
    for index, valid in enumerate(validity):
        streak = streak + 1 if valid else 0
        longest = max(longest, streak)
        score = bool(valid and longest >= 50)
        rows.append(
            {
                "t": index * 0.02,
                "robots": {
                    "r0": {
                        "phase": "learned",
                        "grip_command": 0.0,
                        "controller_state": {},
                        "evaluation": {
                            "object_position": [0, 0, 0.825],
                            "receiving_surface_xyz": [0, 0, 0.8],
                            "object_speed_m_s": 0.0,
                            "hand_clearance_m": 0.15,
                            "minimum_root_height": 0.79,
                            "placement_xy_error_m": 0.0,
                            "contacts": ["r0/receiving_marker"] if valid else [],
                            "streak": streak * 0.02,
                            "settled_seconds": longest * 0.02,
                            "physical_success": score,
                            "success": score,
                            "controller_declared_completion": False,
                        },
                    }
                },
            }
        )
    return rows


def test_learned_phase_and_false_done_do_not_prevent_physical_success():
    result = physical_scores(records([True] * 60), "r0")
    assert result["physical_success"]
    assert result["strict_current_settle_success"]
    assert not result["authored_phase_completed"]
    assert not result["reported_controller_completion"]


def test_previous_settling_does_not_count_as_current_continuous_settling():
    result = physical_scores(records([True] * 60 + [False] + [True]), "r0")
    assert result["physical_success"]  # Historical runner's weaker criterion.
    assert result["ever_physical_success"]
    assert not result["strict_current_settle_success"]
    assert result["current_settled_seconds"] == 0.02


@pytest.mark.parametrize("contact", ["r1/receiving_marker", "r0/pad_a", "floor"])
def test_wrong_contact_cannot_support_claimed_settling(contact):
    rows = records([True] * 60)
    rows[-1]["robots"]["r0"]["evaluation"]["contacts"] = [contact]
    with pytest.raises(ValueError):
        physical_scores(rows, "r0")


def test_completion_claim_cannot_turn_failed_contact_into_placement():
    rows = records([False] * 60)
    for row in rows:
        row["robots"]["r0"]["phase"] = "complete"
        row["robots"]["r0"]["evaluation"]["controller_declared_completion"] = True
    result = physical_scores(rows, "r0")
    assert result["authored_phase_completed"]
    assert not result["physical_success"]


def test_corrupt_xy_metric_is_rejected():
    rows = copy.deepcopy(records([True] * 60))
    rows[20]["robots"]["r0"]["evaluation"]["placement_xy_error_m"] = 0.03
    with pytest.raises(ValueError, match="Placement error"):
        physical_scores(rows, "r0")
