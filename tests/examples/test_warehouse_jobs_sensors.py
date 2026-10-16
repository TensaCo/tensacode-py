"""Job selection must not fabricate evidence or leak another color's targets."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/robotics"))
from warehouse.physics_jobs_sensors import (
    JobColorRenderer,
    JobSensorBelief,
    SharedSensorBelief,
    select_job_colors,
)


def pixels():
    return np.array(
        [
            [
                [230, 150, 50],
                [30, 180, 55],
                [160, 20, 200],
                [10, 35, 210],
                [5, 160, 210],
                [100, 100, 100],
            ]
        ],
        dtype=np.uint8,
    )


@pytest.mark.parametrize("job,parcel,receiver", [("a", 0, 2), ("b", 1, 3)])
def test_disjoint_job_masks_preserve_raw_and_cyan(job, parcel, receiver):
    raw = pixels()
    before = raw.copy()
    canonical, masks = select_job_colors(raw, job)
    np.testing.assert_array_equal(raw, before)
    assert np.flatnonzero(masks["parcel"]).tolist() == [parcel]
    assert np.flatnonzero(masks["receiver"]).tolist() == [receiver]
    np.testing.assert_array_equal(canonical[0, 4], raw[0, 4])
    np.testing.assert_array_equal(canonical[0, parcel], [220, 120, 20])
    np.testing.assert_array_equal(canonical[0, receiver], [170, 10, 200])
    for index in set(range(4)) - {parcel, receiver}:
        assert len(set(canonical[0, index])) == 1


def test_depth_wrapper_does_not_modify_sensor_depth():
    class Renderer:
        depth = False

        def enable_depth_rendering(self):
            self.depth = True

        def disable_depth_rendering(self):
            self.depth = False

        def render(self):
            return np.array([[0.8, 0.9]]) if self.depth else pixels()

    wrapper = JobColorRenderer(Renderer(), "b")
    wrapper.render()
    np.testing.assert_array_equal(wrapper.raw_rgb, pixels())
    wrapper.enable_depth_rendering()
    depth = wrapper.render()
    np.testing.assert_array_equal(depth, [[0.8, 0.9]])
    np.testing.assert_array_equal(wrapper.raw_depth, depth)
    depth[0, 0] = 99
    assert wrapper.raw_depth[0, 0] == 0.8


def test_camera_evidence_is_original_even_when_detector_normalizes(monkeypatch):
    belief = JobSensorBelief.__new__(JobSensorBelief)
    raw = pixels()
    canonical, _ = select_job_colors(raw, "b")
    belief.renderer = SimpleNamespace(raw_rgb=raw.copy())
    belief.first_rgb = None

    def measurement(self, data):
        self.latest_rgb = canonical.copy()
        self.first_rgb = canonical.copy()
        return np.array([1.0, 2.0, 3.0]), 0.7

    monkeypatch.setattr(SharedSensorBelief, "_camera_measurement", measurement)
    result = belief._camera_measurement(None)
    assert result[1] == 0.7
    np.testing.assert_array_equal(belief.first_rgb, raw)
    np.testing.assert_array_equal(belief.latest_rgb, raw)
    assert not np.array_equal(belief.latest_rgb, canonical)


def test_invalid_job_rejected_before_allocating_renderer():
    with pytest.raises(ValueError, match="job"):
        JobSensorBelief(None, None, "r0", 0.0, job="unknown")


def test_saturated_orange_top_is_retained():
    # Bright top-face color observed in the physical scene must not be dropped
    # merely because red and green approach saturation together.
    rgb = np.array(
        [[[255, 240, 90], [220, 120, 20], [10, 70, 210], [5, 160, 210]]], dtype=np.uint8
    )
    _, masks_a = select_job_colors(rgb, "a")
    _, masks_b = select_job_colors(rgb, "b")
    assert masks_a["parcel"].tolist() == [[True, True, False, False]]
    assert masks_b["receiver"].tolist() == [[False, False, True, False]]
    assert masks_a["localizer"].tolist() == [[False, False, False, True]]


@pytest.mark.parametrize("job,shift", [("a", 0.0), ("b", 1.0)])
def test_fixed_camera_offset_is_calibration_not_live_tracking(
    monkeypatch, tmp_path, job, shift
):
    import json

    def initialize(self, local, shared, prefix, offset, **kwargs):
        self.camera = SimpleNamespace(
            lookat=np.array([1.1, 2.25, 0.65]),
            distance=2.5,
            azimuth=90.0,
            elevation=-78.0,
        )
        self.renderer = SimpleNamespace()

    monkeypatch.setattr(SharedSensorBelief, "__init__", initialize)
    belief = JobSensorBelief(None, None, "r0", 2.5, job=job)
    np.testing.assert_allclose(belief.camera.lookat, [1.1, 2.25 + shift, 0.65])
    assert (belief.camera.distance, belief.camera.azimuth, belief.camera.elevation) == (
        2.5,
        90.0,
        -78.0,
    )

    def save(self, directory):
        (directory / "sensor-contract.json").write_text("{}")

    monkeypatch.setattr(SharedSensorBelief, "save_evidence", save)
    belief.save_evidence(tmp_path)
    contract = json.loads((tmp_path / "sensor-contract.json").read_text())
    assert contract["fixed_job_camera_offset_xyz_m"] == [0.0, shift, 0.0]
    assert "no robot or target tracking" in contract["camera_mount"]
