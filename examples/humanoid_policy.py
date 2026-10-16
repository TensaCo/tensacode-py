"""Full-body G1 locomotion and grasping demonstration, using published Unitree weights."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
import yaml
from scipy.spatial.transform import Rotation

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


def main(args):
    xml = scene(
        args.assets, not args.no_gripper, args.object_x, args.object_y, args.mass
    )
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    cfg = configuration(args.assets)
    default = np.array(cfg["default_joint_pos"])
    kp = np.array(cfg["stiffness"])
    kd = np.array(cfg["damping"])
    scale = np.array(cfg["actions"]["JointPositionAction"]["scale"])
    d.qpos[7:36] = default
    d.qpos[2] = 0.79
    mujoco.mj_forward(m, d)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(
            args.assets
            / "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
        ),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    torch.set_num_threads(2)
    machine = None
    alg = None
    if args.substrate != "native":
        alg = compile_walker(args.assets)
        machine = alg.compile(args.substrate)
    servo = None
    if machine is not None and m.nu == 31:
        servo = compile_servo(m.actuator_ctrlrange[:, 1]).compile(args.substrate)
    action = np.zeros(29)
    target = default.copy()
    minimum = 10
    poses = []
    records = []
    stage = "walk"
    stage_start = 0.0
    arm_start = None
    arm_seed = None
    max_lift = 0.0
    contact_frames = 0
    site = m.site("grasp").id if not args.no_gripper else None
    obj = m.body("object").id if not args.no_gripper else None
    goal = np.array([args.object_x - 0.25, args.object_y + 0.25])
    grip_target = 0.0
    held_seconds = 0.0
    walk_distance = 0.0
    foot_counts = [0, 0]
    previous_feet = [False, False]
    object_goal = np.array([args.object_x, args.object_y, 0.814])
    for step in range(int(args.seconds / 0.002)):
        if step % 10 == 0:
            quat = d.qpos[3:7]
            yaw = math.atan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
            )
            delta = goal - d.qpos[:2]
            cmd = (
                np.r_[
                    np.clip(
                        np.array(
                            [
                                [math.cos(yaw), math.sin(yaw)],
                                [-math.sin(yaw), math.cos(yaw)],
                            ]
                        )
                        @ delta
                        * 1.5,
                        -args.speed,
                        args.speed,
                    ),
                    np.clip(-1.5 * yaw, -0.4, 0.4),
                ]
                if args.task
                else np.array([args.speed, 0, 0])
            )
            obs = observation(d, action, cmd, default)
            if args.task and stage in ["reach", "close", "lift", "hold", "release"]:
                obs[33:40] = 0
                obs[62:69] = 0
            if machine is None:
                action = sess.run(None, {"obs": obs[None]})[0][0]
            else:
                with torch.no_grad():
                    value = machine({"observation": obs})["action"]
                action = value.numpy() if isinstance(value, torch.Tensor) else value
            target = default + scale * action
            if args.task:
                if stage == "walk" and np.linalg.norm(delta) < 0.055:
                    stage = "settle"
                    stage_start = d.time
                    walk_distance = float(np.linalg.norm(d.qpos[:2]))
                if stage == "settle" and d.time - stage_start > 1.5:
                    stage = "reach"
                    stage_start = d.time
                    arm_start = d.site_xpos[site].copy()
                    arm_rot = Rotation.from_matrix(
                        d.site_xmat[site].reshape(3, 3)
                    ).as_rotvec()
                    arm_seed = d.qpos[29:36].copy()
                elapsed = d.time - stage_start
                if stage in ["reach", "close", "lift", "hold", "release"]:
                    if stage == "reach":
                        alpha = np.clip(elapsed / 6, 0, 1)
                        alpha = alpha * alpha * (3 - 2 * alpha)
                        if elapsed < 3:
                            blend = np.clip(elapsed / 3, 0, 1)
                            blend = blend * blend * (3 - 2 * blend)
                            desired = (1 - blend) * arm_start + blend * (
                                object_goal + [-0.18, 0, 0.126]
                            )
                        elif elapsed < 6:
                            blend = (elapsed - 3) / 3
                            blend = blend * blend * (3 - 2 * blend)
                            desired = (1 - blend) * (
                                object_goal + [-0.18, 0, 0.126]
                            ) + blend * (object_goal + [0, 0, 0.056])
                        else:
                            blend = np.clip((elapsed - 6) / 2, 0, 1)
                            blend = blend * blend * (3 - 2 * blend)
                            desired = (1 - blend) * (
                                object_goal + [0, 0, 0.056]
                            ) + blend * object_goal.copy()
                        if elapsed > 9:
                            stage = "close"
                            stage_start = d.time
                    else:
                        desired = object_goal.copy()
                        if stage == "close" and elapsed > 1.5:
                            stage = "lift"
                            stage_start = d.time
                            elapsed = 0.0
                        if stage in ["lift", "hold", "release"]:
                            alpha = min(elapsed / 3, 1) if stage == "lift" else 1
                            desired += np.array([-0.04, 0, 0.16]) * alpha
                            if stage == "lift" and elapsed > 3:
                                stage = "hold"
                                stage_start = d.time
                                elapsed = 0.0
                            if stage == "hold" and elapsed > 2:
                                stage = "release"
                                stage_start = d.time
                                elapsed = 0.0
                    desired_rot = (
                        Rotation.from_rotvec((1 - alpha) * arm_rot).as_matrix()
                        if stage == "reach"
                        else np.eye(3)
                    )
                    candidate = arm_ik(m, d, desired, arm_seed, desired_rot)
                    arm_seed += np.clip(candidate - arm_seed, -0.015, 0.015)
                    target[22:29] = arm_seed
                    grip_target = (
                        0.028
                        if stage in ["close", "lift", "hold"] and not args.open_gripper
                        else 0
                    )
                if stage == "release" and d.time - stage_start > 2:
                    break
            if step % 2500 == 0:
                print(
                    json.dumps({"progress_seconds": d.time, "stage": stage}), flush=True
                )
            poses.append(d.qpos.copy())
            contact_names = []
            pad_forces = {"pad_a": 0.0, "pad_b": 0.0}
            feet = [False, False]
            if obj is not None:
                max_lift = max(max_lift, d.xpos[obj, 2] - 0.805)
                for ci, c in enumerate(d.contact):
                    names = [m.geom(int(c.geom1)).name, m.geom(int(c.geom2)).name]
                    if "cube" in names:
                        contact_names.extend(names)
                        force = np.zeros(6)
                        mujoco.mj_contactForce(m, d, ci, force)
                        for pad in pad_forces:
                            if pad in names:
                                pad_forces[pad] += float(np.linalg.norm(force[:3]))
                    if "floor" in names:
                        for side, idx in [("left", 0), ("right", 1)]:
                            if any(side + "_foot" in n for n in names):
                                feet[idx] = True
                if "pad_a" in contact_names and "pad_b" in contact_names:
                    contact_frames += 1
                    if d.xpos[obj, 2] > 0.905:
                        held_seconds += 0.02
            for idx in range(2):
                if feet[idx] and not previous_feet[idx] and stage == "walk":
                    foot_counts[idx] += 1
            previous_feet = feet
            records.append(
                {
                    "t": d.time,
                    "stage": stage,
                    "base": d.qpos[:3].tolist(),
                    "hand": d.site_xpos[site].tolist() if site is not None else [],
                    "object": d.xpos[obj].tolist() if obj is not None else [],
                    "contacts": contact_names,
                    "pad_force_n": pad_forces,
                    "feet_contact": feet,
                    "observation": obs.tolist(),
                    "action": action.tolist(),
                }
            )
        d.ctrl[:29] = kp * (target - d.qpos[7:36]) - kd * d.qvel[6:35]
        if args.task and stage in ["reach", "close", "lift", "hold", "release"]:
            d.ctrl[22:29] = (
                np.array([30] * 4 + [12] * 3) * (target[22:29] - d.qpos[29:36])
                - np.array([2] * 4 + [0.8] * 3) * d.qvel[28:35]
                + d.qfrc_bias[28:35]
            )
        if m.nu > 29:
            d.ctrl[29:] = 300 * (grip_target - d.qpos[36:38]) - 3 * d.qvel[35:37]
        if servo is not None:
            gains = np.r_[kp, 300.0, 300.0]
            damp = np.r_[kd, 3.0, 3.0]
            feedforward = np.zeros(31)
            if args.task and stage in ["reach", "close", "lift", "hold", "release"]:
                gains[22:29] = [30] * 4 + [12] * 3
                damp[22:29] = [2] * 4 + [0.8] * 3
                feedforward[22:29] = d.qfrc_bias[28:35]
            with torch.no_grad():
                torque = servo(
                    {
                        "position": d.qpos[7:38].copy(),
                        "velocity": d.qvel[6:37].copy(),
                        "target": np.r_[target, grip_target, grip_target],
                        "stiffness": gains,
                        "damping": damp,
                        "feedforward": feedforward,
                    }
                )["torque"]
            d.ctrl[:] = torque.numpy() if isinstance(torque, torch.Tensor) else torque
        mujoco.mj_step(m, d)
        minimum = min(minimum, d.qpos[2])
        if d.qpos[2] < 0.4:
            break
    result = {
        "substrate": args.substrate,
        "seconds": d.time,
        "position": d.qpos[:3].tolist(),
        "min_height": minimum,
        "steps": len(poses),
        "stage": stage,
        "max_lift": max_lift,
        "bilateral_contact_seconds": contact_frames * 0.02,
        "held_above_10cm_seconds": held_seconds,
        "walking_distance_m": walk_distance,
        "foot_touchdowns": foot_counts,
        "open_gripper_control": args.open_gripper,
        "object_mass_kg": args.mass,
        "object_initial_xy": [args.object_x, args.object_y],
        "success": bool(
            minimum > 0.65
            and walk_distance > 0.7
            and held_seconds >= 1
            and stage == "release"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scene.xml").write_text(xml)
    (args.output / "records.json").write_text(json.dumps(records))
    np.savez_compressed(args.output / "probe.npz", qpos=np.array(poses))
    (args.output / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, default=ASSETS)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--speed", type=float, default=0.3)
    p.add_argument("--no-gripper", action="store_true")
    p.add_argument("--task", action="store_true")
    p.add_argument("--open-gripper", action="store_true")
    p.add_argument("--substrate", default="native")
    p.add_argument("--object-x", type=float, default=1.25)
    p.add_argument("--object-y", type=float, default=-0.25)
    p.add_argument("--mass", type=float, default=0.08)
    p.add_argument("--output", type=Path, default=Path("/tmp/tensorcode-g1-mjlab"))
    main(p.parse_args())
