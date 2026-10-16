"""Destination estimates use image/depth observations, including missing vision."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
path = Path(__file__).resolve().parents[2] / "examples/robotics/transfer/sensors.py"
spec = spec_from_file_location("destination_sensors", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def image_fixture():
    # Calibrated downward pinhole camera: f=100 px, optical depth=1 m.
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    rgb[30:50, 30:50] = [160, 12, 195]
    depth = np.ones((80, 80))

    def backproject(mask):
        row, col = np.where(mask)
        z = depth[row, col]
        return np.column_stack(
            [
                1.4 + (col + 0.5 - 40) * z / 100,
                -0.4 + (40 - row - 0.5) * z / 100,
                1.8 - z,
            ]
        )

    return rgb, depth, backproject


def observer():
    sensor = module.SensorBelief.__new__(module.SensorBelief)
    sensor.calibration_bias = np.zeros(3)
    return sensor


def test_destination_top_center_from_rgbd():
    sensor = observer()
    rgb, depth, project = image_fixture()
    sensor._destination_measurement(rgb, depth, project, 1.0)
    assert sensor.destination_valid
    np.testing.assert_allclose(sensor.destination_position, [1.4, -0.4, 0.8])
    assert sensor.last_destination_measurement_time == 1.0


def test_occlusion_cannot_shift_cached_destination():
    sensor = observer()
    rgb, depth, project = image_fixture()
    sensor._destination_measurement(rgb, depth, project, 1.0)
    rgb[:, :40] = 0
    sensor._destination_measurement(rgb, depth, project, 2.0)
    assert not sensor.destination_valid
    np.testing.assert_allclose(sensor.destination_position, [1.4, -0.4, 0.8])
    assert sensor.last_destination_measurement_time == 1.0


def test_missing_marker_has_no_fabricated_destination():
    sensor = observer()
    rgb, depth, project = image_fixture()
    rgb[:] = [200, 140, 10]  # Orange object is not the magenta destination.
    sensor._destination_measurement(rgb, depth, project, 1.0)
    assert not sensor.destination_valid
    assert getattr(sensor, "destination_position", None) is None


def test_camera_outage_skips_acquisition_but_continues_odometry():
    sensor = observer()
    sensor.camera_outage = (1.0, 2.0)
    sensor.camera_period = 0.1
    sensor._last_camera = -np.inf
    sensor.covariance = np.eye(3)
    sensor.object_position = None
    sensor.last_object_measurement_time = None
    sensor.root_position = np.zeros(3)
    sensor.stance_contacts = [False, False]
    sensor.root_camera_valid = False
    sensor.model = mujoco.MjModel.from_xml_string("<mujoco/>")
    acquisitions, odometry = [], []
    sensor._camera_measurement = lambda d: (acquisitions.append(d.time), 0.0)
    sensor._odometry = lambda d: odometry.append(d.time)
    for time in (1.0, 1.5, 1.99):
        result = sensor.observe(SimpleNamespace(time=time, contact=[]))
        assert not result["camera_valid"]
        assert not result["destination_valid"]
        assert result["destination_position"] is None
    assert acquisitions == []
    assert odometry == [1.0, 1.5, 1.99]
    sensor.observe(SimpleNamespace(time=2.1, contact=[]))
    assert acquisitions == [2.1]


@pytest.mark.parametrize("swapped", [False, True])
def test_pad_vector_reaction_is_invariant_to_contact_order(monkeypatch, swapped):
    sensor = observer()
    sensor.camera_period = 0.1
    sensor._last_camera = 0.0
    sensor.covariance = np.eye(3)
    sensor.object_position = None
    sensor.last_object_measurement_time = None
    sensor.root_position = np.zeros(3)
    sensor.stance_contacts = [False, False]
    sensor.root_camera_valid = False
    sensor.camera_valid = False
    sensor.camera_confidence = 0.0
    sensor.model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
        <geom name="pad_a" size=".01"/><geom name="pad_b" size=".01"/>
        <geom name="anonymous" size=".01"/></worldbody></mujoco>""")
    sensor._odometry = lambda _: None
    # Deliberately non-symmetric rotation catches frame-vs-frame.T mistakes.
    frame = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    local_force = np.array([2.0, -3.0, 4.0])
    # Swap geom ordering and reverse the normal and first tangent, preserving
    # a right-handed frame and the same physical force on the pad.
    flip = np.diag([-1.0, -1.0, 1.0])
    used_frame = flip @ frame if swapped else frame
    used_force = -flip @ local_force if swapped else local_force
    contacts = [
        SimpleNamespace(
            geom1=2 if swapped else 0,
            geom2=0 if swapped else 2,
            frame=used_frame.ravel(),
        )
    ]

    def contact_force(model, data, index, result):
        result[:3] = used_force

    monkeypatch.setattr(mujoco, "mj_contactForce", contact_force)
    pad_rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = sensor.observe(
        SimpleNamespace(
            time=0.01, contact=contacts, geom_xmat=np.tile(pad_rotation.ravel(), (3, 1))
        )
    )
    np.testing.assert_allclose(
        result["pad_force_local_n"][0], -pad_rotation.T @ frame.T @ local_force
    )
    np.testing.assert_allclose(result["pad_force_local_n"][1], [0.0, 0.0, 0.0])
    assert result["pad_force_n"][0] == pytest.approx(np.linalg.norm(local_force))


def test_actual_resting_payload_reacts_downward_on_tactile_pad():
    # MuJoCo convention: https://mujoco.readthedocs.io/en/stable/computation/
    # Positive contact force acts towards geom2; test real physics, not a mock.
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep=".002" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="pad_a" type="plane" size="1 1 .01"/>
        <geom name="pad_b" type="sphere" size=".01" pos="5 5 5"/>
        <body pos="0 0 .06"><freejoint/>
          <geom type="box" size=".05 .05 .05" mass=".2"/>
        </body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(1500):
        mujoco.mj_step(model, data)
    sensor = observer()
    sensor.model = model
    sensor.camera_period = 0.1
    sensor._last_camera = float(data.time)
    sensor.covariance = np.eye(3)
    sensor.object_position = None
    sensor.last_object_measurement_time = None
    sensor.root_position = np.zeros(3)
    sensor.stance_contacts = [False, False]
    sensor.root_camera_valid = False
    sensor.camera_valid = False
    sensor.camera_confidence = 0.0
    sensor._odometry = lambda _: None
    result = sensor.observe(data)
    np.testing.assert_allclose(
        result["pad_force_local_n"][0], [0.0, 0.0, -0.2 * 9.81], atol=0.002
    )
    np.testing.assert_allclose(result["pad_force_local_n"][1], [0.0, 0.0, 0.0])
    assert result["pad_force_n"][0] == pytest.approx(0.2 * 9.81, abs=0.002)
