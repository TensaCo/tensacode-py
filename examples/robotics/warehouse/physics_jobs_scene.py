"""Two physical pickup/receiving pairs per calibrated humanoid workcell."""

from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1.model import ASSETS
from warehouse.physics_scene import compose_scene

SECOND_JOB_OFFSET = np.array([0.0, 1.0, 0.0])


def compose_jobs_scene(count=2, spacing=2.5, assets=ASSETS, scenery=True):
    xml, local_xml, offsets = compose_scene(count, spacing, assets, scenery)
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    for index in range(count):
        prefix = f"r{index}/"
        for name in ("object", "pedestal", "receiving_support", "receiving_marker"):
            template = next(item for item in world if item.get("name") == prefix + name)
            second = copy.deepcopy(template)
            for item in second.iter():
                if item.get("name"):
                    item.set("name", item.get("name") + "_b")
            position = np.fromstring(second.get("pos", "0 0 0"), sep=" ")
            second.set("pos", " ".join(map(str, position + SECOND_JOB_OFFSET)))
            if name == "object":
                second.find("geom").set("rgba", ".05 .75 .12 1")
            if name == "receiving_marker":
                second.set("rgba", ".03 .12 .85 1")
            world.append(second)
    return ET.tostring(root, encoding="unicode"), local_xml, offsets
