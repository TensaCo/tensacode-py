"""Validated environment and sensor interventions for warehouse physics trials.

Scene perturbations are applied before simulation. Their values are never sent
as targets to controllers; controllers receive only the resulting sensor data.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

import numpy as np


def vector(value, name, bound):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) and abs(x) <= bound for x in result):
        raise ValueError(f"{name} exceeds its bounded trial range")
    return result


def normalize_scenario(value, robots):
    """Reject misspelled interventions rather than silently running nominal."""
    value = copy.deepcopy(value or {})
    allowed = {"id", "seed", "cells", "sensors", "load_compensation"}
    if set(value) - allowed:
        raise ValueError(f"Unknown scenario fields: {set(value) - allowed}")
    cells = value.get("cells", [{} for _ in range(robots)])
    if len(cells) != robots:
        raise ValueError("One explicit cell configuration is required per robot")
    normalized = []
    for cell in cells:
        if set(cell) - {"pickup_delta_xyz", "receiver_delta_xyz", "payload_mass_kg"}:
            raise ValueError("Unknown cell intervention")
        pickup = vector(
            cell.get("pickup_delta_xyz", [0, 0, 0]), "pickup_delta_xyz", 0.2
        )
        receiver = vector(
            cell.get("receiver_delta_xyz", [0, 0, 0]), "receiver_delta_xyz", 0.3
        )
        mass = float(cell.get("payload_mass_kg", 0.08))
        if not math.isfinite(mass) or not 0.02 <= mass <= 0.5:
            raise ValueError("payload_mass_kg must be between .02 and .5 kg")
        normalized.append(
            {
                "pickup_delta_xyz": pickup,
                "receiver_delta_xyz": receiver,
                "payload_mass_kg": mass,
            }
        )
    sensor = value.get("sensors", {})
    if set(sensor) - {
        "depth_noise_std_m",
        "calibration_bias_xyz",
        "camera_outage",
        "camera_azimuth_delta_deg",
        "camera_elevation_delta_deg",
    }:
        raise ValueError("Unknown sensor intervention")
    noise = float(sensor.get("depth_noise_std_m", 0))
    if not math.isfinite(noise) or not 0 <= noise <= 0.05:
        raise ValueError("Depth noise must be between 0 and .05 m")
    outage = sensor.get("camera_outage")
    if outage is not None:
        if (
            len(outage) != 2
            or not all(math.isfinite(float(x)) for x in outage)
            or not 0 <= outage[0] < outage[1]
        ):
            raise ValueError("camera_outage must be [start,end] seconds")
        outage = [float(x) for x in outage]
    sensor = {
        "depth_noise_std_m": noise,
        "calibration_bias_xyz": vector(
            sensor.get("calibration_bias_xyz", [0, 0, 0]), "calibration_bias_xyz", 0.1
        ),
        "camera_outage": outage,
        "camera_azimuth_delta_deg": float(sensor.get("camera_azimuth_delta_deg", 0)),
        "camera_elevation_delta_deg": float(
            sensor.get("camera_elevation_delta_deg", 0)
        ),
    }
    for key in ("camera_azimuth_delta_deg", "camera_elevation_delta_deg"):
        if not math.isfinite(sensor[key]) or abs(sensor[key]) > 20:
            raise ValueError("Camera perturbation must be within20 degrees")
    compensation = value.get("load_compensation", True)
    if not isinstance(compensation, bool):
        raise TypeError("load_compensation must be boolean")
    seed = value.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned32-bit integer")
    return {
        "id": str(value.get("id", "nominal")),
        "seed": seed,
        "cells": normalized,
        "sensors": sensor,
        "load_compensation": compensation,
    }


def perturb_scene(xml, scenario):
    """Move real initial geometry, preserving grounded support tops and inertia."""
    root = ET.fromstring(xml)
    for index, cell in enumerate(scenario["cells"]):
        prefix = f"r{index}/"

        def named(tag, name, prefix=prefix):
            result = root.find(f".//{tag}[@name='{prefix + name}']")
            if result is None:
                raise ValueError(f"Missing scene element {prefix + name}")
            return result

        def position(element, delta):
            pos = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
            element.set("pos", " ".join(map(str, pos + delta)))

        pickup = np.asarray(cell["pickup_delta_xyz"])
        position(named("body", "object"), pickup)
        pedestal = named("geom", "pedestal")
        position(pedestal, pickup * [1, 1, 0.5])
        size = np.fromstring(pedestal.get("size"), sep=" ")
        size[2] += pickup[2] / 2
        pedestal.set("size", " ".join(map(str, size)))
        named("geom", "cube").set("mass", str(cell["payload_mass_kg"]))
        receiver = np.asarray(cell["receiver_delta_xyz"])
        position(named("geom", "receiving_marker"), receiver)
        support = named("geom", "receiving_support")
        position(support, receiver * [1, 1, 0.5])
        size = np.fromstring(support.get("size"), sep=" ")
        size[2] += receiver[2] / 2
        support.set("size", " ".join(map(str, size)))
    return ET.tostring(root, encoding="unicode")


class DepthNoiseRenderer:
    """Transparent renderer wrapper that corrupts axial depth, never poses."""

    def __init__(self, renderer, std, seed):
        self.renderer, self.std = renderer, std
        self.rng = np.random.default_rng(seed)
        self.depth = False

    def __getattr__(self, name):
        return getattr(self.renderer, name)

    def enable_depth_rendering(self):
        self.depth = True
        self.renderer.enable_depth_rendering()

    def disable_depth_rendering(self):
        self.depth = False
        self.renderer.disable_depth_rendering()

    def render(self):
        frame = self.renderer.render()
        if self.depth and self.std:
            frame = frame.copy()
            valid = np.isfinite(frame) & (frame > 0)
            frame[valid] = np.maximum(
                0.001, frame[valid] + self.rng.normal(0, self.std, valid.sum())
            )
        return frame
