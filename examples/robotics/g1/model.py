"""Shared G1 model, frozen locomotion, servo and estimated-state inverse kinematics."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

from tensorcode.algorithm import AlgorithmBuilder

ASSETS = Path("/tmp/tensorcode-humanoid-assets/unitree_rl_mjlab")


def configuration(assets=ASSETS):
    return yaml.safe_load(
        (
            assets / "deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml"
        ).read_text()
    )


def scene(assets=ASSETS, gripper=True, object_x=1.25, object_y=-0.25, mass=0.08):
    root = ET.parse(assets / "src/assets/robots/unitree_g1/xmls/g1.xml").getroot()
    root.find("compiler").set(
        "meshdir", str(assets / "src/assets/robots/unitree_g1/xmls/assets")
    )
    cfg = configuration(assets)
    ET.SubElement(
        root,
        "option",
        timestep=".002",
        integrator="implicitfast",
        cone="elliptic",
        iterations="50",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(
        visual, "headlight", diffuse=".7 .7 .7", ambient=".3 .3 .3", specular=".2 .2 .2"
    )
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1=".16 .20 .27",
        rgb2=".21 .26 .34",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_mat",
        texture="grid",
        texrepeat="6 6",
        texuniform="true",
    )
    # Keep training collision geometry; disable robot self-collisions, retain environment contacts.
    root.find(".//default[@class='collision']/geom").set("contype", "1")
    root.find(".//default[@class='collision']/geom").set("conaffinity", "2")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="1 -2 4", dir="0 .3 -1", directional="true")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="10 10 .1",
        material="floor_mat",
        friction="1 .005 .0001",
        contype="2",
        conaffinity="1",
    )
    actuators = ET.SubElement(root, "actuator")
    joints = [j for j in root.findall(".//joint") if j.get("name")]
    for i, j in enumerate(joints):
        n = j.get("name")
        j.set("armature", str(cfg["stiffness"][i] / (20 * math.pi) ** 2))
        limit = (
            139
            if ("hip_roll" in n or "knee" in n)
            else 88
            if ("hip_" in n or "waist_yaw" in n)
            else 50
            if ("waist" in n or "ankle" in n)
            else 5
            if ("wrist_pitch" in n or "wrist_yaw" in n)
            else 25
        )
        ET.SubElement(
            actuators,
            "motor",
            name=n,
            joint=n,
            ctrllimited="true",
            ctrlrange=f"{-limit} {limit}",
        )
    if gripper:
        wrist = root.find(".//body[@name='right_wrist_yaw_link']")
        for child in list(wrist):
            if (
                child.get("mesh") == "right_rubber_hand"
                or child.get("name") == "right_hand_collision"
            ):
                wrist.remove(child)
        hand = ET.SubElement(wrist, "body", name="tool", pos=".04 0 0")
        ET.SubElement(
            hand,
            "geom",
            name="tool_palm",
            type="box",
            size=".025 .065 .025",
            mass=".08",
            rgba=".22 .27 .32 1",
            contype="1",
            conaffinity="2",
        )
        ET.SubElement(
            hand, "site", name="grasp", pos=".085 0 0", size=".004", rgba="0 1 0 1"
        )
        for side, sgn in [("a", 1), ("b", -1)]:
            body = ET.SubElement(
                hand, "body", name=f"finger_{side}", pos=f".065 {sgn * 0.045} 0"
            )
            ET.SubElement(
                body,
                "joint",
                name=f"grip_{side}",
                type="slide",
                axis=f"0 {-sgn} 0",
                range="0 .042",
                damping="1",
                armature=".001",
            )
            ET.SubElement(
                body,
                "geom",
                name=f"pad_{side}",
                type="box",
                size=".045 .008 .026",
                mass=".035",
                friction="2 .02 .001",
                condim="4",
                rgba=".1 .75 .58 1",
                solref=".01 1",
                contype="1",
                conaffinity="2",
            )
            ET.SubElement(
                actuators,
                "motor",
                name=f"grip_{side}",
                joint=f"grip_{side}",
                ctrllimited="true",
                ctrlrange="-12 12",
            )
        ET.SubElement(
            world,
            "geom",
            name="pedestal",
            type="box",
            pos=f"{object_x} {object_y} .39",
            size=".10 .10 .39",
            rgba=".26 .32 .4 1",
            contype="2",
            conaffinity="1",
        )
        cube = ET.SubElement(
            world, "body", name="object", pos=f"{object_x} {object_y} .808"
        )
        ET.SubElement(cube, "freejoint", name="object_free")
        ET.SubElement(
            cube,
            "geom",
            name="cube",
            type="box",
            size=".025 .025 .025",
            mass=str(mass),
            friction="1.8 .02 .001",
            condim="4",
            rgba="1 .55 .12 1",
            solref=".01 1",
            contype="3",
            conaffinity="3",
        )
    return ET.tostring(root, encoding="unicode")


def observation(d, a, cmd, default):
    qw, qx, qy, qz = d.qpos[3:7]
    grav = [
        2 * (-qz * qx + qw * qy),
        -2 * (qz * qy + qw * qx),
        1 - 2 * (qw * qw + qz * qz),
    ]
    phase = 2 * math.pi * d.time / 0.6
    gait = [math.sin(phase), math.cos(phase)] if np.linalg.norm(cmd) >= 0.1 else [0, 0]
    return np.concatenate(
        [d.qvel[3:6], grav, cmd, gait, d.qpos[7:36] - default, d.qvel[6:35], a]
    ).astype(np.float32)


def compile_walker(assets=ASSETS):
    """Lower the pinned supplied MLP, including its normalization, into tensor IR."""
    import onnx
    from onnx import numpy_helper

    source = onnx.load(
        assets / "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
    )
    if [n.op_type for n in source.graph.node] != [
        "Sub",
        "Div",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
        "Elu",
        "Gemm",
    ]:
        raise ValueError("Unexpected pretrained architecture")
    w = {n.name: numpy_helper.to_array(n) for n in source.graph.initializer}
    b = AlgorithmBuilder(name="unitree_g1_pretrained_velocity", dtype="float32")
    x = b.input("observation", (98,)).reshape(1, 98)
    x = (x - b.constant(w["obs_normalizer._mean"])) / b.constant(w["onnx::Div_24"])
    for layer in [0, 2, 4, 6]:
        x = x @ b.parameter(
            f"weight_{layer}", w[f"mlp.{layer}.weight"].T
        ) + b.parameter(f"bias_{layer}", w[f"mlp.{layer}.bias"])
        if layer != 6:
            positive = x >= 0
            # Eager branch evaluation: mask positive values before exponentiation.
            negative = b.where(positive, 0, x).exp() - 1
            x = b.where(positive, x, negative)
    return b.build({"action": x.reshape(29)})


def compile_servo(limits):
    b = AlgorithmBuilder(name="humanoid_joint_and_gripper_servo", dtype="float64")
    q = b.input("position", (31,))
    v = b.input("velocity", (31,))
    goal = b.input("target", (31,))
    kp = b.input("stiffness", (31,))
    kd = b.input("damping", (31,))
    ff = b.input("feedforward", (31,))
    return b.build(
        {
            "torque": ((goal - q) * kp - v * kd + ff).clip(
                -np.asarray(limits), np.asarray(limits)
            )
        }
    )


def arm_ik(m, d, desired, seed=None, desired_rot=None):
    ik = mujoco.MjData(m)
    ik.qpos[:] = d.qpos
    ids = np.arange(22, 29)
    qa = 7 + ids
    va = 6 + ids
    if seed is not None:
        ik.qpos[qa] = seed
    site = m.site("grasp").id
    if desired_rot is None:
        desired_rot = np.eye(3)
    for _ in range(20):
        mujoco.mj_forward(m, ik)
        pos = ik.site_xpos[site]
        rot = ik.site_xmat[site].reshape(3, 3)
        orient = 0.5 * sum(
            (np.cross(rot[:, i], desired_rot[:, i]) for i in range(3)),
            start=np.zeros(3),
        )
        err = np.r_[desired - pos, 0.25 * orient]
        jp = np.zeros((3, m.nv))
        jr = np.zeros_like(jp)
        mujoco.mj_jacSite(m, ik, jp, jr, site)
        jac = np.vstack([jp[:, va], 0.25 * jr[:, va]])
        delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.0001 * np.eye(6), err)
        ik.qpos[qa] += np.clip(delta, -0.12, 0.12)
        for a in range(7):
            ji = m.joint(
                [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ][a]
            ).id
            ik.qpos[qa[a]] = np.clip(
                ik.qpos[qa[a]], m.jnt_range[ji, 0] + 0.02, m.jnt_range[ji, 1] - 0.02
            )
        if np.linalg.norm(err) < 0.001:
            break
    return ik.qpos[qa].copy()
