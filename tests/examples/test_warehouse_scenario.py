"""Interventions change the environment without becoming controller targets."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "scenario",
    Path(__file__).parents[2] / "examples/robotics/warehouse/physics_scenario.py",
)
scenario = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scenario)


@pytest.mark.parametrize(
    "value",
    [
        {"typo": 1},
        {"cells": []},
        {"cells": [{"payload_mass_kg": -1}]},
        {"sensors": {"depth_noise_std_m": float("nan")}},
        {"sensors": {"camera_outage": [2, 1]}},
        {"load_compensation": "false"},
        {"seed": -1},
    ],
)
def test_reject_invalid_interventions(value):
    with pytest.raises((ValueError, TypeError)):
        scenario.normalize_scenario(value, 1)


def test_geometry_supports_remain_grounded_and_mass_is_physical():
    xml = """<mujoco><worldbody><geom name="r0/pedestal" pos="1 -.2 .39" size=".1 .1 .39"/><body name="r0/object" pos="1 -.2 .808"><geom name="r0/cube" mass=".08"/></body><geom name="r0/receiving_support" pos="1.2 -.48 .44" size=".1 .1 .44"/><geom name="r0/receiving_marker" pos="1.2 -.48 .882"/></worldbody></mujoco>"""
    cfg = scenario.normalize_scenario(
        {
            "cells": [
                {
                    "pickup_delta_xyz": [0.02, -0.01, 0.06],
                    "receiver_delta_xyz": [-0.02, 0.04, 0.08],
                    "payload_mass_kg": 0.2,
                }
            ]
        },
        1,
    )
    root = ET.fromstring(scenario.perturb_scene(xml, cfg))

    def item(tag, name):
        return root.find(f".//{tag}[@name='r0/{name}']")

    for name in ["pedestal", "receiving_support"]:
        p = np.fromstring(item("geom", name).get("pos"), sep=" ")
        size = np.fromstring(item("geom", name).get("size"), sep=" ")
        assert p[2] - size[2] == pytest.approx(0)
    assert np.fromstring(item("body", "object").get("pos"), sep=" ") == pytest.approx(
        [1.02, -0.21, 0.868]
    )
    assert np.fromstring(
        item("geom", "receiving_marker").get("pos"), sep=" "
    ) == pytest.approx([1.18, -0.44, 0.962])
    assert float(item("geom", "cube").get("mass")) == 0.2


def test_depth_noise_is_seeded_and_does_not_modify_rgb_or_source_frame():
    class Renderer:
        def __init__(self):
            self.depth = False
            self.frame = np.ones((20, 20))

        def enable_depth_rendering(self):
            self.depth = True

        def disable_depth_rendering(self):
            self.depth = False

        def render(self):
            return self.frame if self.depth else np.ones((20, 20, 3), dtype=np.uint8)

    a = scenario.DepthNoiseRenderer(Renderer(), 0.005, 12)
    b = scenario.DepthNoiseRenderer(Renderer(), 0.005, 12)
    assert np.array_equal(a.render(), b.render())
    a.enable_depth_rendering()
    b.enable_depth_rendering()
    first = a.render()
    assert np.array_equal(first, b.render())
    assert np.std(first) > 0.004
    assert np.all(a.renderer.frame == 1)
    assert not np.array_equal(first, a.render())
