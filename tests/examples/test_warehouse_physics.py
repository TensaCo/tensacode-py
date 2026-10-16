"""Shared-world collision and independent actuation contracts for real G1 assets."""

import xml.etree.ElementTree as ET
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def warehouse():
    robotics = Path(__file__).resolve().parents[2] / "examples/robotics"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(robotics))
        scene = import_module("warehouse.physics_scene")
        if not (scene.ASSETS / "src/assets/robots/unitree_g1/xmls/g1.xml").is_file():
            pytest.skip("Pinned Unitree G1 simulation assets are not installed")
        xml, local_xml, offsets = scene.compose_scene(
            count=2, spacing=1.6, scenery=False
        )
        model = mujoco.MjModel.from_xml_string(xml)
        yield scene, model, xml, local_xml, offsets


@pytest.fixture
def world(warehouse):
    scene, model, *_ = warehouse
    return model, scene.initialize(model, 2)


def root_address(model, index):
    return model.jnt_qposadr[model.joint(f"r{index}/floating_base_joint").id]


def robot_geoms(model, index):
    pelvis = model.body(f"r{index}/pelvis").id
    return {
        geom
        for geom in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[geom]] == pelvis
    }


def touching(data, left, right):
    return any(
        (contact.geom1 in left and contact.geom2 in right)
        or (contact.geom2 in left and contact.geom1 in right)
        for contact in data.contact
    )


def test_namespaced_actuators_have_disjoint_joint_and_state_addresses(world):
    model, data = world
    assert (model.nq, model.nu) == (90, 62)
    joint_sets = []
    qpos_sets = []
    dof_sets = []
    for index in range(2):
        motors = [
            a
            for a in range(model.nu)
            if model.actuator(a).name.startswith(f"r{index}/")
        ]
        assert len(motors) == 31
        joints = {int(model.actuator_trnid[a, 0]) for a in motors}
        assert len(joints) == 31
        assert all(model.joint(j).name.startswith(f"r{index}/") for j in joints)
        joint_sets.append(joints)
        qpos_sets.append({int(model.jnt_qposadr[j]) for j in joints})
        dof_sets.append({int(model.jnt_dofadr[j]) for j in joints})
    assert joint_sets[0].isdisjoint(joint_sets[1])
    assert qpos_sets[0].isdisjoint(qpos_sets[1])
    assert dof_sets[0].isdisjoint(dof_sets[1])
    other_indices = sorted(qpos_sets[1])
    previous = data.qpos[other_indices].copy()
    first_q = min(qpos_sets[0])
    data.qpos[first_q] += 0.1
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.qpos[other_indices], previous)
    data.ctrl[0] = 1.0
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.qfrc_actuator[sorted(dof_sets[0])]) > 0
    np.testing.assert_array_equal(data.qfrc_actuator[sorted(dof_sets[1])], 0.0)


def test_overlapping_robots_generate_real_cross_robot_contacts(world):
    model, data = world
    first, second = root_address(model, 0), root_address(model, 1)
    data.qpos[second : second + 3] = data.qpos[first : first + 3] + [0.015, 0.015, 0.0]
    mujoco.mj_forward(model, data)
    assert touching(data, robot_geoms(model, 0), robot_geoms(model, 1))
    assert not touching(data, robot_geoms(model, 0), robot_geoms(model, 0))
    assert not touching(data, robot_geoms(model, 1), robot_geoms(model, 1))


@pytest.mark.parametrize("index", [0, 1])
def test_floor_contacts_survive_scene_composition(world, index):
    model, data = world
    data.qpos[root_address(model, index) + 2] -= 0.08
    mujoco.mj_forward(model, data)
    assert touching(data, robot_geoms(model, index), {model.geom("floor").id})


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("support", ["pedestal", "receiving_marker", "floor"])
def test_parcels_physically_contact_supports(world, index, support):
    model, data = world
    prefix = f"r{index}/"
    cube = model.geom(prefix + "cube").id
    joint = model.joint(prefix + "object_free").id
    adr = model.jnt_qposadr[joint]
    surface = model.geom("floor" if support == "floor" else prefix + support).id
    if support == "floor":
        position = np.array([2.0, index * 1.6, 0.024])
    else:
        position = model.geom_pos[surface].copy()
        position[2] += model.geom_size[surface, 2] + 0.024
    data.qpos[adr : adr + 3] = position
    mujoco.mj_forward(model, data)
    assert touching(data, {cube}, {surface})


def test_self_collision_masks_and_original_exclusions_are_preserved(warehouse):
    _, model, xml, local_xml, _ = warehouse
    for index in range(2):
        geoms = robot_geoms(model, index)
        physical = [geom for geom in geoms if model.geom_contype[geom]]
        assert physical
        for first in physical:
            for second in physical:
                assert not (
                    (model.geom_contype[first] & model.geom_conaffinity[second])
                    or (model.geom_contype[second] & model.geom_conaffinity[first])
                )
    original = ET.fromstring(local_xml).findall("contact/exclude")
    combined = ET.fromstring(xml).findall("contact/exclude")
    actual = {(entry.get("body1"), entry.get("body2")) for entry in combined}
    assert original
    for index in range(2):
        for entry in original:
            assert (
                f"r{index}/" + entry.get("body1"),
                f"r{index}/" + entry.get("body2"),
            ) in actual
    assert len(combined) == 2 * len(original)


def test_initial_lane_offsets_are_explicit_calibration_priors(warehouse, world):
    _, _, _, _, offsets = warehouse
    model, data = world
    np.testing.assert_allclose(offsets, [[0.0, 0.0, 0.0], [0.0, 1.6, 0.0]])
    for index, offset in enumerate(offsets):
        adr = root_address(model, index)
        np.testing.assert_allclose(
            data.qpos[adr : adr + 3], np.array(offset) + [0.0, 0.0, 0.79]
        )
        np.testing.assert_allclose(data.qpos[adr + 3 : adr + 7], [1.0, 0.0, 0.0, 0.0])
    assert model.opt.timestep == pytest.approx(0.002)
