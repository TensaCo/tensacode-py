"""The controller's belief boundary must not read simulator world-pose labels."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
path = Path(__file__).resolve().parents[2] / "examples/robotics/transfer/sensors.py"
spec = spec_from_file_location("humanoid_sensor_example", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)


class EncoderOnly:
    def __init__(self, qpos, allowed):
        self.qpos = qpos
        self.allowed = set(allowed)
        self.reads = set()

    def __getitem__(self, key):
        indices = np.arange(len(self.qpos))[key]
        selected = set(np.atleast_1d(indices).tolist())
        assert selected <= self.allowed, (
            f"Forbidden pose read: {selected - self.allowed}"
        )
        self.reads |= selected
        return self.qpos[key]


@pytest.fixture
def belief(monkeypatch):
    monkeypatch.setattr(
        mujoco, "Renderer", lambda *a, **kw: SimpleNamespace(close=lambda: None)
    )
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="pelvis" pos="0 0 .79"><freejoint name="floating_base_joint"/>
        <geom type="sphere" size=".1"/>
        <body name="left_ankle_roll_link" pos="0 .1 -.6">
          <joint name="left" type="hinge"/><geom name="left_foot1" size=".05"/>
        </body>
        <body name="right_ankle_roll_link" pos="0 -.1 -.6">
          <joint name="right" type="hinge"/><geom name="right_foot1" size=".05"/>
        </body>
      </body>
      <body name="object" pos="1 -.25 .8"><freejoint name="object_free"/>
        <geom name="cube" size=".025"/>
      </body>
    </worldbody></mujoco>""")
    observer = module.SensorBelief(model)
    data = mujoco.MjData(model)
    data.qpos[0:3] = [99, -42, 7]  # Privileged world truth is intentionally wrong.
    data.qpos[-7:-4] = [-123, 456, 789]
    data.qpos[7:9] = [0.2, -0.3]
    guarded = EncoderOnly(data.qpos, [3, 4, 5, 6, 7, 8])
    proxy = SimpleNamespace(qpos=guarded, contact=[], time=0.0)
    return observer, proxy, guarded


def test_no_world_pose_reads_or_missing_vision_fallback(belief):
    observer, data, guarded = belief
    observer._camera_measurement = lambda _: (None, 0.0)
    result = observer.observe(data)
    assert result["object_position"] is None
    assert result["camera_valid"] is False
    np.testing.assert_allclose(result["root_position"], [0, 0, 0.79])
    assert guarded.reads == {3, 4, 5, 6, 7, 8}
    assert len(observer._encoders) == 2  # NumPy integer / MuJoCo enum regression.
    np.testing.assert_allclose(observer.kinematic_data.qpos[7:9], [0.2, -0.3])
    np.testing.assert_allclose(observer.kinematic_data.qpos[:3], [0, 0, 0.79])


def test_repeated_pixels_reduce_uncertainty_and_gate_outliers(belief):
    observer, data, _ = belief
    measurement = np.array([1.2, -0.2, 0.81])
    observer._camera_measurement = lambda _: (measurement.copy(), 1.0)
    first = observer.observe(data)
    first_variance = np.trace(first["object_covariance"])
    data.time = 0.1
    second = observer.observe(data)
    assert np.trace(second["object_covariance"]) < first_variance
    np.testing.assert_allclose(second["object_position"], measurement)
    observer._camera_measurement = lambda _: (np.array([4.0, 4.0, 4.0]), 1.0)
    data.time = 0.2
    result = observer.observe(data)
    assert result["camera_valid"] is False
    np.testing.assert_allclose(result["object_position"], measurement)
    assert observer.rejected_measurements == 1


def test_unseen_object_never_accepts_placeholder_state(belief):
    observer, data, _ = belief
    observer._camera_measurement = lambda _: (None, 0.0)
    for time in (0.0, 0.1, 1.0):
        data.time = time
        assert observer.observe(data)["object_position"] is None
    assert observer.accepted_measurements == 0


def test_offline_audit_separates_pickup_from_stale_lift_posterior():
    audit_path = path.with_name("audit.py")
    audit_spec = spec_from_file_location("humanoid_sensor_audit", audit_path)
    audit_module = module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit_module)
    base = {
        "base": [0, 0, 0.8],
        "object": [1, 0, 0.8],
        "sensor_state": {
            "root_position": [0.003, 0, 0.8],
            "object_position": [1, 0, 0.8],
        },
        "target_anchor": [1, 0, 0.809],
    }
    rows = [
        {**base, "stage": "walk"},
        {**base, "stage": "reach"},
        {**base, "stage": "lift", "object": [1, 0, 1.0]},
    ]
    report = audit_module.audit_records(rows)
    assert report["root_localization"]["max_m"] == pytest.approx(0.003)
    assert report["object_localization_before_grasp"]["max_m"] == 0
    assert report["object_localization_all_phases"]["max_m"] == pytest.approx(0.2)
    assert report["frozen_pickup_anchor_error_m"] == pytest.approx(0, abs=1e-9)
