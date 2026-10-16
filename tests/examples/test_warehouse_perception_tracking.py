import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/robotics/warehouse/perception_tracking.py"
)
_SPEC = importlib.util.spec_from_file_location("warehouse_perception_tracking", _PATH)
m = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = m
_SPEC.loader.exec_module(m)


def observation(position=(0.0, 0.0, 0.0), axes=(0, 1, 2), timestamp=0.0, name="p"):
    return SimpleNamespace(
        kind="parcel",
        color="orange",
        dimensions_xyz=(0.05, 0.05, 0.05),
        center_xyz=position,
        constrained_axes=axes,
        position_ambiguity_m=(0.003, 0.003, 0.003),
        residual_m=0.001,
        observation_id=name,
        timestamp=timestamp,
        provenance={"frame_sha256": name},
    )


def test_complementary_views_resolve_missing_axes_with_evidence():
    tracker = m.SceneTracker()
    first = tracker.update([observation(axes=(1, 2))], 0)[0]
    assert not first["localized"]
    assert first["axis_evidence_age_s"][0] is None
    second = tracker.update([observation(axes=(0, 2), name="second")], 0)[0]
    assert second["localized"]
    assert len(second["sources"]) == 2
    assert second["confidence_score"] > 0.7


def test_repeated_unobservable_coordinate_cannot_become_confident():
    tracker = m.SceneTracker()
    for i in range(10):
        result = tracker.update(
            [observation(axes=(1, 2), timestamp=i * 0.02)], i * 0.02
        )[0]
    assert not result["localized"]
    assert result["position_std_xyz_m"][0] > 0.04
    assert result["confidence_score"] < 0.5


def test_absent_measurements_force_abstention_and_uncertainty_growth():
    tracker = m.SceneTracker()
    initial = tracker.update([observation()], 0)[0]
    stale = tracker.update([], 1.0)[0]
    assert initial["localized"] and not stale["localized"]
    assert stale["position_std_xyz_m"][0] > initial["position_std_xyz_m"][0]
    assert stale["axis_evidence_age_s"] == [1.0, 1.0, 1.0]


def test_reordered_instances_associate_by_geometry_not_component_id():
    tracker = m.SceneTracker()
    a, b = tracker.update(
        [observation(name="a"), observation((0.2, 0, 0), name="b")], 0
    )
    results = tracker.update(
        [
            observation((0.201, 0, 0), timestamp=0.1, name="a"),
            observation((0.001, 0, 0), timestamp=0.1, name="b"),
        ],
        0.1,
    )
    assert results[0]["id"] == a["id"]
    assert results[1]["id"] == b["id"]
    assert results[0]["center_xyz"][0] < 0.01
    assert results[1]["center_xyz"][0] > 0.19


def test_identity_ambiguity_does_not_silently_assign():
    tracker = m.SceneTracker(m.TrackingConfig(association_gate_mahalanobis=1000.0))
    tracker.update([observation((-0.02, 0, 0)), observation((0.02, 0, 0))], 0)
    result = tracker.update([observation((0, 0, 0), timestamp=0.1)], 0.1)
    assert tracker.events[-1]["event"] == "identity_ambiguous"
    assert all(t["observations"] == 1 for t in result)


def test_duplicate_time_allowed_but_out_of_order_rejected():
    tracker = m.SceneTracker()
    tracker.update([observation(timestamp=1.0)], 1.0)
    with pytest.raises(ValueError):
        tracker.update([], 0.0)
    with pytest.raises(ValueError):
        tracker.update([observation(timestamp=0.0)], 1.0)


def test_covariance_remains_positive_and_has_systematic_floor():
    tracker = m.SceneTracker()
    for i in range(200):
        output = tracker.update([observation(name=str(i))], 0)[0]
    assert np.linalg.eigvalsh(tracker.tracks[0].covariance).min() >= 0
    assert min(output["position_std_xyz_m"]) >= tracker.config.systematic_floor_m
