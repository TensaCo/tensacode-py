"""Offline camera-study math is independently testable without a GL context."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy")
path = (
    Path(__file__).resolve().parents[2]
    / "examples/robotics/warehouse/physics_camera_study.py"
)
spec = spec_from_file_location("warehouse_camera_study", path)
study = module_from_spec(spec)
spec.loader.exec_module(study)


def camera():
    return {
        "position": [0, 0, 3],
        "forward": [0, 0, -1],
        "right": [1, 0, 0],
        "up": [0, 1, 0],
        "width": 16,
        "height": 16,
        "focal_pixels": 1000.0,
    }


def test_pixel_center_reconstruction_and_depth_convention():
    depth = np.full((16, 16), 2.0)
    points, rays = study.reconstruct(depth, camera())
    np.testing.assert_allclose(points[..., 2], 1.0)
    np.testing.assert_allclose(points[7, 7], [-0.001, 0.001, 1.0])
    np.testing.assert_allclose(rays[7, 7], [-0.0005, 0.0005, -1.0])


def test_image_only_cube_estimate_uses_visible_top_and_shape_prior():
    depth = np.full((16, 16), 2.0)
    points, _ = study.reconstruct(depth, camera())
    rgb = np.full((16, 16, 3), [200, 110, 25], dtype=np.uint8)
    estimate, mask = study.image_cube_estimate(rgb, depth, points)
    np.testing.assert_allclose(estimate, [0.0, 0.0, 0.975])
    assert mask.all()
    estimate, mask = study.image_cube_estimate(np.zeros_like(rgb), depth, points)
    assert estimate is None
    assert not mask.any()


def test_offline_plane_score_recovers_known_depth_bias():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="r0/cube" type="box" size=".025 .025 .025" pos="0 0 .975"/>
      <geom name="floor" type="plane" size="3 3 .1"/>
      <geom name="r0/receiving_marker" type="box" size=".1 .1 .002" pos="1 0 .8"/>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    depth = np.full((16, 16), 2.001)
    points, rays = study.reconstruct(depth, camera())
    segmentation = np.zeros((16, 16, 2), dtype=int)
    segmentation[..., 0] = model.geom("r0/cube").id
    segmentation[..., 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    scores = study.score_planes(
        model, data, "r0/", depth, segmentation, points, rays, camera()
    )
    cube = scores["r0/cube"]
    assert cube["depth_error"]["count"] == 144
    assert cube["depth_error"]["mean_m"] == pytest.approx(0.001)
    assert cube["plane_normal_error"]["mean_m"] == pytest.approx(-0.001)
    assert scores["floor"]["depth_error"]["count"] == 0
