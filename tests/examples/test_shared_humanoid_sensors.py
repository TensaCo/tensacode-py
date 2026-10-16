"""Shared physics never leaks other robot/object poses into a robot belief."""

import xml.etree.ElementTree as ET
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
path = (
    Path(__file__).resolve().parents[2]
    / "examples/robotics/warehouse/physics_sensors.py"
)
spec = spec_from_file_location("shared_physics_sensors", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)


class RestrictedArray:
    def __init__(self, values, allowed):
        self.values, self.allowed = values, set(allowed)
        self.reads = set()

    def __getitem__(self, index):
        indices = set(np.atleast_1d(np.arange(len(self.values))[index]).tolist())
        assert indices <= self.allowed, (
            f"Forbidden world-state access: {indices - self.allowed}"
        )
        self.reads |= indices
        return self.values[index]


def models():
    root = ET.fromstring("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <body name="pelvis" pos="0 0 .79"><freejoint name="floating_base_joint"/>
        <geom size=".1"/>
        <geom name="pad_a" size=".01" pos=".2 0 0"/>
        <geom name="pad_b" size=".01" pos=".3 0 0"/>
        <body name="left_ankle_roll_link" pos="0 .1 -.6">
          <joint name="left"/><geom name="left_foot1" size=".05"/>
        </body>
        <body name="right_ankle_roll_link" pos="0 -.1 -.6">
          <joint name="right"/><geom name="right_foot1" size=".05"/>
        </body>
      </body>
      <body name="object" pos="1 0 .8"><freejoint name="object_free"/><geom size=".025"/></body>
    </worldbody></mujoco>""")
    local = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for element in root.iter():
        if element.get("name") and element.get("name") != "floor":
            element.set("name", "r0/" + element.get("name"))
    shared = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return local, shared


def test_shared_world_pose_boundary_and_camera_data(monkeypatch):
    monkeypatch.setattr(
        mujoco, "Renderer", lambda *a, **kw: SimpleNamespace(close=lambda: None)
    )
    local, shared = models()
    belief = module.SharedSensorBelief(local, shared, "r0", 2.0)
    data = mujoco.MjData(shared)
    data.qpos[:3] = [123.0, 456.0, 789.0]
    data.qpos[7:9] = [0.2, -0.3]
    data.qvel[3:8] = [0.1, 0.2, 0.3, 0.4, 0.5]
    qpos = RestrictedArray(data.qpos, range(3, 9))
    qvel = RestrictedArray(data.qvel, range(3, 8))
    floor, foot = shared.geom("floor").id, shared.geom("r0/left_foot1").id
    proxy = SimpleNamespace(
        qpos=qpos,
        qvel=qvel,
        time=0.0,
        contact=[SimpleNamespace(geom1=floor, geom2=foot)],
    )
    camera_calls = []

    def camera(self, observed):
        assert observed is proxy
        camera_calls.append(observed)
        return None, 0.0

    monkeypatch.setattr(module.SensorBelief, "_camera_measurement", camera)
    result = belief.observe(proxy)
    assert result["object_position"] is None
    assert result["stance_contacts"] == [True, False]
    assert len(camera_calls) == 1
    np.testing.assert_allclose(result["root_position"], [0.0, 2.0, 0.79])
    np.testing.assert_allclose(belief.proprio.qpos[:3], [0.0, 2.0, 0.79])
    np.testing.assert_allclose(belief.proprio.qpos[7:9], [0.2, -0.3])
    np.testing.assert_allclose(belief.proprio.qvel[3:8], [0.1, 0.2, 0.3, 0.4, 0.5])
    assert qpos.reads == set(range(3, 9))
    assert qvel.reads == set(range(3, 8))
    assert result["pad_force_local_n"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_tactile_transducer_uses_shared_model_and_contact_index(monkeypatch):
    monkeypatch.setattr(
        mujoco, "Renderer", lambda *a, **kw: SimpleNamespace(close=lambda: None)
    )
    local, shared = models()
    belief = module.SharedSensorBelief(local, shared, "r0/", [0.0, 2.0, 0.0])
    belief._camera_measurement = lambda _: (None, 0.0)
    data = mujoco.MjData(shared)
    pad = shared.geom("r0/pad_a").id
    proxy = SimpleNamespace(
        qpos=data.qpos,
        qvel=data.qvel,
        time=0.0,
        contact=[SimpleNamespace(geom1=pad, geom2=0, frame=np.eye(3).ravel())],
        geom_xmat=np.tile(np.eye(3).ravel(), (shared.ngeom, 1)),
    )
    calls = []

    def force(model, observed, index, output):
        assert model is shared and observed is proxy
        calls.append(index)
        output[:3] = [2.0, 0.0, 1.0]

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    result = belief.observe(proxy)
    assert calls == [0]
    assert result["pad_force_local_n"] == [[-2.0, 0.0, -1.0], [0.0, 0.0, 0.0]]
