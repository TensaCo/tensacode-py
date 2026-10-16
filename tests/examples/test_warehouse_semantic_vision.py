"""Result-critical checks: calibrated depth, abstention and honest partial recall."""

import numpy as np
import pytest

from examples.robotics.warehouse.semantic_vision.adapter import visible_surface_sample
from examples.robotics.warehouse.semantic_vision.study import score_partial_labels
from examples.robotics.warehouse.semantic_vision.tiles import (
    detect_tiled,
    overlapping_tiles,
)


@pytest.fixture
def camera():
    return {
        "width": 20,
        "height": 20,
        "focal_pixels": 10,
        "position": [1, 2, 3],
        "forward": [0, 1, 0],
        "right": [1, 0, 0],
        "up": [0, 0, 1],
    }


def test_depth_backprojection_uses_camera_basis_and_pixel_centers(camera):
    result = visible_surface_sample([0, 0, 20, 20], np.full((20, 20), 2.0), camera)
    assert result["status"] == "surface_sample"
    np.testing.assert_allclose(result["visible_surface_xyz"], [1, 4, 3])
    assert "not object center" in result["interpretation"]


@pytest.mark.parametrize("bad", [np.nan, 0, np.inf, 11])
def test_invalid_depth_abstains(camera, bad):
    result = visible_surface_sample([0, 0, 20, 20], np.full((20, 20), bad), camera)
    assert result["status"] == "abstained"
    assert "visible_surface_xyz" not in result


def test_depth_discontinuity_abstains(camera):
    depth = np.ones((20, 20))
    depth[:, 10:] = 3
    result = visible_surface_sample([0, 0, 20, 20], depth, camera)
    assert result["reason"] == "mixed depth surfaces"


def test_tiny_box_abstains(camera):
    assert (
        visible_surface_sample([3, 3, 4, 4], np.ones((20, 20)), camera)["status"]
        == "abstained"
    )


def test_invalid_calibration_rejected(camera):
    camera["up"] = camera["forward"]
    with pytest.raises(ValueError, match="Orthonormal"):
        visible_surface_sample([0, 0, 20, 20], np.ones((20, 20)), camera)


def test_one_detection_cannot_count_twice_against_partial_labels():
    row = {"frame": "f", "query": "a box", "detections": [{"box_xyxy": [0, 0, 10, 10]}]}
    annotations = {
        "frames": [
            {
                "path": "f",
                "boxes": [{"bbox_xyxy": [0, 0, 10, 10]}, {"bbox_xyxy": [0, 0, 10, 10]}],
            }
        ]
    }
    result = score_partial_labels([row], annotations)
    assert result["matched_at_iou_0_5"] == 1
    assert result["recall_at_iou_0_5"] == 0.5
    assert "precision" not in {key for key in result}


def test_tiles_cover_all_pixels_without_target_position():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    coverage = np.zeros(image.shape[:2])
    for crop, (x, y) in overlapping_tiles(image):
        assert crop.shape == (24, 36, 3)
        coverage[y : y + crop.shape[0], x : x + crop.shape[1]] += 1
    assert np.min(coverage) == 1
    assert np.max(coverage) == 4


def test_tiled_predictions_map_back_to_original_image():
    class Detector:
        def detect(self, rgb, query, **kwargs):
            assert rgb.shape == (24, 36, 3)
            assert query == "a box"
            return [{"box_xyxy": [1, 2, 5, 6], "score": 0.8, "query": query}]

    detections = detect_tiled(
        Detector(), np.zeros((40, 60, 3), dtype=np.uint8), "a box"
    )
    assert [d["box_xyxy"] for d in detections] == [
        [1, 2, 5, 6],
        [25, 2, 29, 6],
        [1, 18, 5, 22],
        [25, 18, 29, 22],
    ]
