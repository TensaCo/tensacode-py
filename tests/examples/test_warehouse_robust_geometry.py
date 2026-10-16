"""Robust geometric consensus retains evidence of rejected sensor pixels."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/robotics"))
from warehouse.perception import observe_rgbd
from warehouse.perception_learning.robust_geometry import observe_robust_rgbd


def fixture():
    spec = importlib.util.spec_from_file_location(
        "inverse_geometry_test_fixture",
        Path(__file__).with_name("test_warehouse_perception.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.frame()


def test_silhouette_depth_outlier_is_reported_not_hidden():
    rgb, depth, cal = fixture()
    rows, cols = np.where(np.isfinite(depth))
    depth[rows[len(rows) // 2], cols[len(rows) // 2]] += 0.3
    baseline = observe_rgbd(rgb, depth, cal, 0)[0]
    robust = observe_robust_rgbd(rgb, depth, cal, 0)[0]
    assert robust.provenance["rejected_pixels"] >= 1
    assert robust.provenance["inlier_fraction"] > 0.99
    assert robust.residual_m < baseline.residual_m / 10
    assert np.linalg.norm(np.array(robust.center_xyz) - [0, 0, 0.025]) < 0.001
    assert "original_frame_sha256" in robust.provenance


def test_clean_geometry_not_changed_by_consensus():
    rgb, depth, cal = fixture()
    old = observe_rgbd(rgb, depth, cal, 0)[0]
    new = observe_robust_rgbd(rgb, depth, cal, 0)[0]
    assert np.allclose(old.center_xyz, new.center_xyz)
    assert new.provenance["rejected_pixels"] == 0


def test_multiview_adapter_rejects_duplicate_or_asynchronous_views():
    import pytest
    from warehouse.perception_learning.scene_observer import observe_views

    rgb, depth, cal = fixture()
    frame = {"rgb": rgb, "depth": depth, "calibration": cal, "timestamp": 0.0}
    with pytest.raises(ValueError, match="Distinct camera"):
        observe_views([frame, frame], 0.0, workcell_center_xy=[0, 0])
    shifted = {
        **frame,
        "calibration": {**cal, "position": [0.5, 0, 1]},
        "timestamp": 1.0,
    }
    with pytest.raises(ValueError, match="same capture timestamp"):
        observe_views([frame, shifted], 0.0, workcell_center_xy=[0, 0])


def test_green_yellow_priors_exclude_cyan_fiducial_and_keep_color_identity():
    from warehouse.perception_learning.colored_objects import (
        observe_colored_boxes,
        semantic_color_mask,
    )

    rgb, depth, cal = fixture()
    mask = np.isfinite(depth)
    rgb[mask] = [35, 220, 45]
    result = observe_colored_boxes(rgb, depth, cal, 0)
    assert (
        len(result) == 1 and result[0].color == "green" and result[0].kind == "parcel"
    )
    assert result[0].provenance["actual_color_prior"] == "green"
    colors = np.array(
        [[[20, 170, 220], [220, 190, 25], [230, 120, 20]]], dtype=np.uint8
    )
    assert not semantic_color_mask(colors, "green")[0, 0]
    assert not semantic_color_mask(colors, "blue")[0, 0]
    assert semantic_color_mask(np.array([[[20, 45, 230]]]), "blue")[0, 0]
    assert semantic_color_mask(colors, "yellow")[0, 1]
    assert semantic_color_mask(colors, "orange")[0, 1]  # saturation overlap is retained
    assert semantic_color_mask(colors, "orange")[0, 2]


def test_saturated_orange_parcel_remains_a_geometry_hypothesis():
    from warehouse.perception_learning.colored_objects import observe_colored_boxes

    rgb, depth, cal = fixture()
    rgb[np.isfinite(depth)] = [255, 255, 63]
    hypotheses = observe_colored_boxes(rgb, depth, cal, 0)
    parcels = [h for h in hypotheses if h.kind == "parcel" and h.color == "orange"]
    assert len(parcels) == 1
    assert np.linalg.norm(np.asarray(parcels[0].center_xyz) - [0, 0, 0.025]) < 0.001
