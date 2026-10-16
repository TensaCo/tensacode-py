"""Placement claims require sustained, released support contact and hand clearance."""

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "examples/robotics/transfer/validate.py"
spec = spec_from_file_location("handling_evidence", path)
validator = module_from_spec(spec)
spec.loader.exec_module(validator)


def settled():
    return {
        "object": [1.3, -0.4, 0.825],
        "receiving_surface_xyz": [1.3, -0.4, 0.8],
        "hand": [1.3, -0.4, 1.0],
        "contacts": ["cube", "receiving_marker"],
        "receiving_contact": True,
        "object_speed_m_s": 0.001,
        "controller_state": {"phase": "withdraw"},
        "grip_command": 0,
    }


def test_settled_duration_is_contiguous_not_accumulated():
    good = settled()
    bad = deepcopy(good)
    bad["contacts"].append("pad_a")
    assert validator.placement_metrics([good] * 60)["placement_settled_seconds"] == 1.2
    assert (
        validator.placement_metrics([good] * 30 + [bad] + [good] * 30)[
            "placement_settled_seconds"
        ]
        == 0.6
    )


@pytest.mark.parametrize(
    "change",
    [
        {"controller_state": {"phase": "transport"}},
        {"contacts": ["cube"], "receiving_contact": False},
        {"contacts": ["cube", "receiving_marker", "pad_b"]},
        {"object_speed_m_s": 0.04},
        {"grip_command": 0.02},
        {"hand": [1.3, -0.4, 0.9]},
        {"object": [1.38, -0.4, 0.825]},
        {"object": [1.3, -0.4, 0.85]},
    ],
)
def test_unsettled_or_unreleased_object_cannot_pass(change):
    row = settled() | change
    assert validator.placement_metrics([row] * 60)["placement_settled_seconds"] == 0


def test_contact_label_and_reported_distance_are_cross_checked():
    with pytest.raises(AssertionError):
        validator.placement_metrics([settled() | {"receiving_contact": False}])
    with pytest.raises(AssertionError):
        validator.placement_metrics([settled() | {"hand_object_distance_m": 5.0}])
