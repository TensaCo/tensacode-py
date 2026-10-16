"""Multi-job workcells contain independent physical parcels and clear supports."""

from importlib import import_module
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def scene():
    robotics = Path(__file__).resolve().parents[2] / "examples/robotics"
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(robotics))
        module = import_module("warehouse.physics_jobs_scene")
        base = import_module("warehouse.physics_scene")
        if not (module.ASSETS / "src/assets/robots/unitree_g1/xmls/g1.xml").is_file():
            pytest.skip("Pinned Unitree G1 assets are unavailable")
        xml, local_xml, offsets = module.compose_jobs_scene(scenery=False)
        model = mujoco.MjModel.from_xml_string(xml)
        local = mujoco.MjModel.from_xml_string(local_xml)
        yield module, base, model, local, offsets


@pytest.fixture
def world(scene):
    _, base, model, _, _ = scene
    return model, base.initialize(model, 2)


def test_four_parcels_have_independent_free_joint_state(scene, world):
    _, _, model, local, _ = scene
    _, data = world
    assert model.nq == 104
    assert local.nq == 45
    addresses = []
    for index in range(2):
        for suffix in ("", "_b"):
            joint = model.joint(f"r{index}/object_free{suffix}").id
            assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE
            addresses.append(int(model.jnt_qposadr[joint]))
    assert len(set(addresses)) == 4
    blocks = [set(range(address, address + 7)) for address in addresses]
    assert all(left.isdisjoint(right) for left, right in combinations(blocks, 2))
    untouched = {
        address: data.qpos[address : address + 7].copy() for address in addresses[1:]
    }
    data.qpos[addresses[0]] += 0.1
    mujoco.mj_forward(model, data)
    for address, before in untouched.items():
        np.testing.assert_array_equal(data.qpos[address : address + 7], before)


def test_additional_jobs_leave_robot_actuator_namespaces_independent(world):
    model, data = world
    assert model.nu == 62
    joint_sets = []
    for index in range(2):
        actuators = [
            a
            for a in range(model.nu)
            if model.actuator(a).name.startswith(f"r{index}/")
        ]
        assert len(actuators) == 31
        joints = {int(model.actuator_trnid[a, 0]) for a in actuators}
        assert len(joints) == 31
        assert all(model.joint(joint).name.startswith(f"r{index}/") for joint in joints)
        assert all(
            model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE for joint in joints
        )
        joint_sets.append(joints)
    assert joint_sets[0].isdisjoint(joint_sets[1])
    data.ctrl[0] = 1.0
    mujoco.mj_forward(model, data)
    assert (
        np.linalg.norm(data.qfrc_actuator[[model.jnt_dofadr[j] for j in joint_sets[0]]])
        > 0
    )
    np.testing.assert_array_equal(
        data.qfrc_actuator[[model.jnt_dofadr[j] for j in joint_sets[1]]], 0.0
    )


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("suffix", ["", "_b"])
@pytest.mark.parametrize("surface_name", ["pedestal", "receiving_marker"])
def test_every_parcel_can_contact_its_real_pickup_and_receiving_surface(
    world, index, suffix, surface_name
):
    model, data = world
    prefix = f"r{index}/"
    cube = model.geom(prefix + "cube" + suffix).id
    surface = model.geom(prefix + surface_name + suffix).id
    adr = model.jnt_qposadr[model.joint(prefix + "object_free" + suffix).id]
    top = model.geom_pos[surface].copy()
    top[2] += model.geom_size[surface, 2] + model.geom_size[cube, 2] - 0.001
    data.qpos[adr : adr + 3] = top
    mujoco.mj_forward(model, data)
    assert any({int(c.geom1), int(c.geom2)} == {cube, surface} for c in data.contact)


def test_solid_support_volumes_do_not_overlap(world):
    model, _ = world
    names = [
        f"r{index}/{name}{suffix}"
        for index in range(2)
        for name in ("pedestal", "receiving_support")
        for suffix in ("", "_b")
    ]
    for left_name, right_name in combinations(names, 2):
        left, right = model.geom(left_name).id, model.geom(right_name).id
        # These fixtures are unrotated boxes; positive overlap on all axes
        # indicates intersecting physical support volumes, not mere contact.
        overlap = (
            model.geom_size[left]
            + model.geom_size[right]
            - np.abs(model.geom_pos[left] - model.geom_pos[right])
        )
        assert np.any(overlap <= 1e-9), (
            f"Intersecting supports {left_name}/{right_name}: {overlap}"
        )


def test_initial_job_positions_follow_lane_and_second_job_calibration(scene, world):
    module, _, model, _, offsets = scene
    _, data = world
    for index, offset in enumerate(offsets):
        root = model.jnt_qposadr[model.joint(f"r{index}/floating_base_joint").id]
        np.testing.assert_allclose(
            data.qpos[root : root + 3], np.array(offset) + [0, 0, 0.79]
        )
        for suffix, shift in (("", np.zeros(3)), ("_b", module.SECOND_JOB_OFFSET)):
            adr = model.jnt_qposadr[model.joint(f"r{index}/object_free{suffix}").id]
            np.testing.assert_allclose(
                data.qpos[adr : adr + 3],
                np.array(offset) + [1.25, -0.25, 0.808] + shift,
            )
            np.testing.assert_allclose(data.qpos[adr + 3 : adr + 7], [1, 0, 0, 0])
            cube = model.geom(f"r{index}/cube{suffix}").id
            pedestal = model.geom(f"r{index}/pedestal{suffix}").id
            np.testing.assert_allclose(
                data.geom_xpos[cube, :2], model.geom_pos[pedestal, :2]
            )
            assert (
                data.geom_xpos[cube, 2] - model.geom_size[cube, 2]
                >= model.geom_pos[pedestal, 2] + model.geom_size[pedestal, 2]
            )
