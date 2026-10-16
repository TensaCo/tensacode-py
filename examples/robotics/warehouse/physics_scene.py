"""One shared MuJoCo warehouse with independent, physically colliding G1s."""

from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1.model import ASSETS, configuration, scene


def cell_xml(assets=ASSETS):
    root = ET.fromstring(scene(assets))
    pelvis = root.find(".//body[@name='pelvis']")
    ET.SubElement(
        pelvis,
        "geom",
        name="localizer",
        type="sphere",
        pos=".12 0 .12",
        size=".025",
        rgba="0 .65 1 1",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="receiving_support",
        type="box",
        pos="1.20 -.48 .44",
        size=".10 .10 .44",
        rgba=".22 .27 .32 1",
        contype="2",
        conaffinity="1",
    )
    ET.SubElement(
        world,
        "geom",
        name="receiving_marker",
        type="box",
        pos="1.20 -.48 .882",
        size=".10 .10 .002",
        rgba=".7 .035 .85 1",
        friction="1.8 .02 .001",
        contype="2",
        conaffinity="1",
    )
    return root


def compose_scene(count=2, spacing=1.6, assets=ASSETS, scenery=True):
    if not 1 <= count <= 3:
        raise ValueError("Bounded warehouse supports one to three robots")
    base = cell_xml(assets)
    local_xml = ET.tostring(base, encoding="unicode")
    local_model = mujoco.MjModel.from_xml_string(local_xml)
    effective_types = {
        local_model.geom(i).name: int(local_model.geom_contype[i])
        for i in range(local_model.ngeom)
        if local_model.geom(i).name
    }
    for section in list(base):
        if section.tag in ("sensor", "keyframe"):
            base.remove(section)
    world = base.find("worldbody")
    templates = [
        copy.deepcopy(x)
        for x in world
        if x.tag not in ("light",) and x.get("name") != "floor"
    ]
    for child in list(world):
        if child.tag != "light" and child.get("name") != "floor":
            world.remove(child)
    actuators = base.find("actuator")
    motors = [copy.deepcopy(x) for x in actuators]
    actuators.clear()
    contact = base.find("contact")
    excludes = list(contact) if contact is not None else []
    if contact is not None:
        contact.clear()
    ET.SubElement(
        base.find("asset"),
        "material",
        name="warehouse_concrete",
        rgba=".36 .38 .40 1",
        specular=".05",
        shininess=".1",
    )
    ET.SubElement(
        base.find("asset"),
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".7 .75 .8",
        rgb2=".3 .35 .4",
        width="512",
        height="3072",
    )
    world.find("geom[@name='floor']").set("material", "warehouse_concrete")
    robot_bits = sum(1 << (i + 2) for i in range(count))
    world.find("geom[@name='floor']").set("conaffinity", str(robot_bits | 1))
    offsets = []
    for index in range(count):
        prefix = f"r{index}/"
        offset = np.array([0.0, spacing * index, 0.0])
        offsets.append(offset.tolist())
        elements = copy.deepcopy(templates)
        motor_copy = copy.deepcopy(motors)
        for excluded in copy.deepcopy(excludes):
            for key in ("body1", "body2"):
                if excluded.get(key):
                    excluded.set(key, prefix + excluded.get(key))
            contact.append(excluded)
        names = {
            e.get("name")
            for top in elements + motor_copy
            for e in top.iter()
            if e.get("name")
        }
        for top in elements + motor_copy:
            is_robot = top.tag == "body" and top.get("name") == "pelvis"
            for element in top.iter():
                original_name = element.get("name")
                for key, value in list(element.attrib.items()):
                    if (
                        key
                        in (
                            "name",
                            "joint",
                            "body",
                            "body1",
                            "body2",
                            "site",
                            "geom",
                            "target",
                        )
                        and value in names
                    ):
                        element.set(key, prefix + value)
                if element.tag == "geom":
                    # Resolve inherited robot collision masks explicitly. Visual
                    # geoms with contype=0 remain non-colliding.
                    typ = str(
                        effective_types.get(
                            original_name, int(element.get("contype", "0"))
                        )
                    )
                    if typ == "0":
                        continue
                    if is_robot and typ != "0":
                        bit = 1 << (index + 2)
                        element.set("contype", str(bit))
                        element.set("conaffinity", str(3 | (robot_bits ^ bit)))
                    elif typ == "3":
                        element.set("contype", "1")
                        element.set("conaffinity", str(3 | robot_bits))
                    elif typ == "2":
                        element.set("conaffinity", str(robot_bits | 1))
            if top.tag != "motor":
                position = np.fromstring(top.get("pos", "0 0 0"), sep=" ") + offset
                top.set("pos", " ".join(map(str, position)))
                world.append(top)
            else:
                actuators.append(top)
    if scenery:
        try:
            from warehouse.scenery import add_scenery
        except ImportError:
            pass
        else:
            add_scenery(world, [o[1] for o in offsets])
            for geom in world.findall("geom"):
                if (
                    geom.get("name", "").startswith("warehouse_")
                    and geom.get("contype", "1") != "0"
                ):
                    geom.set("contype", "2")
                    geom.set("conaffinity", str(robot_bits | 1))
    return ET.tostring(base, encoding="unicode"), local_xml, offsets


def initialize(model, count, assets=ASSETS):
    data = mujoco.MjData(model)
    default = np.array(configuration(assets)["default_joint_pos"])
    for i in range(count):
        joint = model.joint(f"r{i}/floating_base_joint").id
        adr = model.jnt_qposadr[joint]
        data.qpos[adr + 2] = 0.79
        # Actuator order is the calibrated 29 body joints followed by two fingers.
        for j in range(29):
            joint_id = model.actuator_trnid[i * 31 + j, 0]
            data.qpos[model.jnt_qposadr[joint_id]] = default[j]
    mujoco.mj_forward(model, data)
    return data
