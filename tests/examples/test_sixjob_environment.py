"""Environment perturbations cover both jobs without changing controller inputs."""

import xml.etree.ElementTree as ET

import pytest

from examples.robotics.warehouse.physics_jobs_environment import perturb_both_jobs


def scene():
    return (
        "<mujoco><worldbody>"
        + "".join(
            f'<body name="r0/object{s}" pos="1 2 3"><geom name="r0/cube{s}" mass="0.08"/></body><geom name="r0/pedestal{s}" pos="1 2 0.5" size=".1 .1 .5"/><geom name="r0/receiving_marker{s}" pos="1 2 1"/><geom name="r0/receiving_support{s}" pos="1 2 .5" size=".1 .1 .5"/>'
            for s in ("", "_b")
        )
        + "</worldbody></mujoco>"
    )


def case(**changes):
    return (
        {
            "pickup_delta_xyz": [0, 0, 0],
            "receiver_delta_xyz": [0, 0, 0],
            "payload_mass_kg": 0.08,
        }
        | changes
    )


def test_nominal_preserves_exact_xml():
    xml = scene()
    assert perturb_both_jobs(xml, case(), 1) == xml


def test_both_jobs_change_mass_and_receiving_surface_together():
    root = ET.fromstring(
        perturb_both_jobs(
            scene(), case(payload_mass_kg=0.2, receiver_delta_xyz=[0, 0, 0.01]), 1
        )
    )
    for suffix in ("", "_b"):
        assert root.find(f".//geom[@name='r0/cube{suffix}']").get("mass") == "0.2"
        marker = root.find(f".//geom[@name='r0/receiving_marker{suffix}']")
        support = root.find(f".//geom[@name='r0/receiving_support{suffix}']")
        assert float(marker.get("pos").split()[2]) == pytest.approx(1.01)
        assert float(support.get("pos").split()[2]) - float(
            support.get("size").split()[2]
        ) == pytest.approx(0)


def test_outside_declared_geometry_is_rejected():
    with pytest.raises(ValueError, match="bounds"):
        perturb_both_jobs(scene(), case(pickup_delta_xyz=[0.03, 0, 0]), 1)
