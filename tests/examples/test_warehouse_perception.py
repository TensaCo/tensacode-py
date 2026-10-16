"""Analytic ray-rendered RGB-D fixtures; truth stays in scoring, outside fitter."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = (
    Path(__file__).resolve().parents[2] / "examples/robotics/warehouse/perception.py"
)
_SPEC = importlib.util.spec_from_file_location("warehouse_perception", _PATH)
p = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = p
_SPEC.loader.exec_module(p)


def frame(center=(0.0, 0.0, 0.025), noise=0.0, occlusion=False):
    size = np.array([0.05] * 3)
    camera = np.array([0.22, -0.27, 0.24])
    forward = np.array(center) - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    calibration = {
        "position": camera.tolist(),
        "forward": forward.tolist(),
        "right": right.tolist(),
        "up": up.tolist(),
        "focal_pixels": 400.0,
        "width": 160,
        "height": 120,
    }
    rays = p.backproject(np.ones((120, 160)), calibration) - camera
    with np.errstate(divide="ignore"):
        a = (np.array(center) - size / 2 - camera) / rays
        b = (np.array(center) + size / 2 - camera) / rays
    near = np.max(np.minimum(a, b), axis=2)
    far = np.min(np.maximum(a, b), axis=2)
    visible = (near <= far) & (near > 0)
    if occlusion:
        visible[:, :70] = False  # A foreground occluder removes part of the silhouette.
    depth = np.where(visible, near, np.nan)
    rng = np.random.default_rng(7)
    depth[visible] += rng.normal(0, noise, visible.sum())
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[visible] = [230, 120, 20]
    return rgb, depth, calibration


@pytest.mark.parametrize(
    "noise,occlusion", [(0.0, False), (0.0001, False), (0.0001, True), (0.001, True)]
)
def test_inverse_geometry_beats_visible_centroid(noise, occlusion):
    rgb, depth, calibration = frame(noise=noise, occlusion=occlusion)
    truth = np.array([0.0, 0.0, 0.025])
    fits = p.observe_rgbd(rgb, depth, calibration, 1.0)
    assert len(fits) == 1
    fit = fits[0]
    points = p.backproject(depth, calibration)[np.isfinite(depth)]
    baseline_error = np.linalg.norm(points.mean(axis=0) - truth)
    fit_error = np.linalg.norm(np.array(fit.center_xyz) - truth)
    assert fit_error < 0.001
    assert fit_error < baseline_error / 3
    assert fit.observed_face_axes == (0, 1, 2)
    assert fit.provenance["source"] == "calibrated_rgbd_pixels"
    assert len(fit.provenance["frame_sha256"]) == 64


def test_top_only_occlusion_does_not_claim_localized_center():
    x, y = np.meshgrid(np.linspace(-0.005, 0.005, 20), np.linspace(-0.005, 0.005, 20))
    points = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.05)])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    fit = p.fit_box(points, normals, [0, 0, 1], p.DEFAULT_PRIORS[0])
    assert fit.constrained_axes == (2,)
    assert fit.confidence_score < 0.5
    assert fit.position_ambiguity_m[0] > 0.019


def test_invalid_calibration_and_dimension_mismatch_rejected():
    rgb, depth, calibration = frame()
    calibration["focal_pixels"] = -1
    with pytest.raises(ValueError):
        p.observe_rgbd(rgb, depth, calibration, 0)
    with pytest.raises(ValueError):
        p.BoxPrior("parcel", "orange", (-1, 1, 1))


def test_missing_pixels_and_wrong_color_produce_no_hypotheses():
    rgb, depth, calibration = frame()
    assert p.observe_rgbd(np.zeros_like(rgb), depth, calibration, 0) == []
    assert p.observe_rgbd(rgb, np.full_like(depth, np.nan), calibration, 0) == []


def test_relations_are_candidates_not_contact_assertions():
    rng = np.random.default_rng(8)

    def fit(center, size, kind, color, name):
        points = rng.uniform(-0.45, 0.45, (400, 3)) * size + center
        points[:, 2] = center[2] + size[2] / 2
        return p.fit_box(
            points,
            np.tile([0.0, 0.0, 1.0], (400, 1)),
            [0, 0, 1],
            p.BoxPrior(kind, color, tuple(size)),
            observation_id=name,
        )

    item = fit(
        np.array([0.0, 0.0, 0.031]), np.array([0.05] * 3), "parcel", "orange", "p"
    )
    tray = fit(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.2, 0.2, 0.012]),
        "receiver",
        "magenta",
        "r",
    )
    relations = p.relation_candidates([item, tray])
    assert relations[0]["predicate"] == "possibly_supported_by"
    assert "not_contact" in relations[0]["source"]


def load_audit():
    sys.modules["perception"] = p
    spec = importlib.util.spec_from_file_location(
        "warehouse_perception_audit", _PATH.with_name("perception_audit.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_inference_does_not_require_evaluation_files(tmp_path):
    import json

    from PIL import Image

    rgb, depth, calibration = frame()
    Image.fromarray(rgb).save(tmp_path / "sensor-rgb.png")
    np.savez(tmp_path / "sensor-depth.npz", depth=depth)
    (tmp_path / "sensor-contract.json").write_text(json.dumps({"camera": calibration}))
    fits, baselines, provenance = load_audit().capture_hypotheses(tmp_path)
    assert len(fits) == 1
    assert fits[0].observation_id in baselines
    assert "sensor-rgb.png" in provenance


def test_real_initial_capture_reports_baseline_regression_honestly():
    capture = _PATH.parents[3] / "docs/results/robotics/handling/runs/high-receiver-v5"
    report = load_audit().audit_capture(capture)
    rows = {row["kind"]: row for row in report["entities"]}
    assert (
        rows["parcel"]["center_error_m"]
        < rows["parcel"]["baseline"]["size_corrected_centroid_error_m"]
    )
    # The receiver midpoint fit is less accurate than the centroid on this frame.
    assert (
        rows["receiver"]["center_error_m"]
        > rows["receiver"]["baseline"]["size_corrected_centroid_error_m"]
    )
