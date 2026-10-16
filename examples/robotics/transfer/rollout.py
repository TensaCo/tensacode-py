"""Full-body G1 locomotion and grasping demonstration, using published Unitree weights."""

from __future__ import annotations

import hashlib
import json
import math

# Shared physical model, locomotion policy, servo and IK implementation.
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
from handling import PickPlace
from planning import GuardedManipulation
from scipy.spatial.transform import Rotation
from sensors import SensorBelief

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1.model import (
    ASSETS,
    arm_ik,
    compile_servo,
    compile_walker,
    configuration,
    observation,
    scene,
)

__all__ = ["ASSETS", "main"]


def main(args):
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("planning.py"),
        Path(__file__).with_name("handling.py"),
        Path(__file__).with_name("sensors.py"),
        Path(__file__).with_name("experiment.py"),
        Path(__file__).with_name("box_qp.py"),
        Path(__file__).parents[1] / "g1/model.py",
    ]
    source_hashes = {
        str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in source_paths
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(Path(__file__).parents[1])))
    xml = scene(
        args.assets, not args.no_gripper, args.object_x, args.object_y, args.mass
    )
    tree = ET.fromstring(xml)
    pelvis = tree.find(".//body[@name='pelvis']")
    if pelvis is None:
        pelvis = tree.find("worldbody/body")
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
    shift = getattr(args, "height_shift", 0.0)
    if shift:
        for element in [
            tree.find(".//geom[@name='pedestal']"),
            tree.find(".//body[@name='object']"),
        ]:
            position = [float(v) for v in element.get("pos").split()]
            position[2] += shift
            element.set("pos", " ".join(map(str, position)))
    handling = getattr(args, "handling", False)
    if handling:
        # Scene parameters configure the environment only; the controller never
        # receives these coordinates. It locates the magenta top via RGB-D.
        dx, dy, dz = args.destination
        world = tree.find("worldbody")
        ET.SubElement(
            world,
            "geom",
            name="receiving_support",
            type="box",
            pos=f"{dx} {dy} {dz / 2}",
            size=f".10 .10 {dz / 2}",
            rgba=".22 .27 .32 1",
            contype="2",
            conaffinity="1",
        )
        ET.SubElement(
            world,
            "geom",
            name="receiving_marker",
            type="box",
            pos=f"{dx} {dy} {dz + 0.002}",
            size=".10 .10 .002",
            rgba=".7 .035 .85 1",
            friction="1.8 .02 .001",
            contype="2",
            conaffinity="1",
        )
    xml = ET.tostring(tree, encoding="unicode")
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
    observer = SensorBelief(
        m,
        calibration_bias=getattr(args, "camera_bias", [0.0, 0.0, 0.0]),
        camera_outage=getattr(args, "camera_outage", None),
    )
    sensed = None
    estimated_root = np.array([0.0, 0.0, 0.79])
    sensing_lost = False
    grip_target = 0.0
    held_seconds = 0.0
    walk_distance = 0.0
    foot_counts = [0, 0]
    previous_feet = [False, False]
    object_goal = None
    context_policy = getattr(args, "context_policy", None)
    manipulation_start = None
    planner = None
    controller_state = None
    placement_streak = 0.0
    placement_longest = 0.0
    receiving_surface = None
    object_speed = 0.0
    hand_object_distance = 0.0
    receiving_contact = False
    if handling:
        receiving_id = m.geom("receiving_marker").id
        object_velocity = int(m.jnt_dofadr[m.joint("object_free").id])
    for step in range(int(args.seconds / 0.002)):
        if step % 10 == 0:
            sensed = observer.observe(d)
            estimated_root = np.asarray(sensed["root_position"])
            control_time = d.time
            if sensed["object_position"] is None:
                sensing_lost = True
                break
            if stage in ["walk", "settle"]:
                object_goal = np.asarray(sensed["object_position"]) + [0.0, 0.0, 0.009]
            goal = object_goal[:2] + [-0.25, 0.25]
            quat = d.qpos[3:7]
            yaw = math.atan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
            )
            delta = goal - estimated_root[:2]
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
                    stage_start = control_time
                    walk_distance = float(np.linalg.norm(d.qpos[:2]))
                if stage == "settle" and control_time - stage_start > 1.5:
                    stage = "reach"
                    stage_start = control_time
                    manipulation_start = control_time
                    arm_start = observer.kinematic_data.site_xpos[site].copy()
                    arm_rot = Rotation.from_matrix(
                        observer.kinematic_data.site_xmat[site].reshape(3, 3)
                    ).as_rotvec()
                    arm_seed = d.qpos[29:36].copy()
                    if getattr(args, "planned", False):
                        planner = (PickPlace if handling else GuardedManipulation)(
                            m,
                            observer.kinematic_data,
                            object_goal - [0, 0, 0.009],
                            d.time,
                            backend=getattr(args, "ik_backend", "scipy"),
                            **(
                                {
                                    "load_compensation": getattr(
                                        args, "load_compensation", True
                                    )
                                }
                                if handling
                                else {}
                            ),
                        )
                elapsed = control_time - stage_start
                if stage in ["reach", "close", "lift", "hold", "release"]:
                    if planner is not None:
                        (
                            arm_seed,
                            desired,
                            desired_rot,
                            grip_target,
                            controller_state,
                        ) = planner.step(observer.kinematic_data, sensed, d.time)
                        target[22:29] = arm_seed
                        stage = {
                            "approach": "reach",
                            "align": "reach",
                            "descend": "reach",
                            "retract": "reach",
                            "failed": "reach",
                            "locate_destination": "hold",
                            "transport": "hold",
                            "lower": "hold",
                            "open": "release",
                            "withdraw": "release",
                            "complete": "release",
                        }.get(planner.phase, planner.phase)
                        if planner.done and not handling:
                            break
                    else:
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
                                stage_start = control_time
                        else:
                            desired = object_goal.copy()
                            if stage == "close" and elapsed > 1.5:
                                stage = "lift"
                                stage_start = control_time
                                elapsed = 0.0
                            if stage in ["lift", "hold", "release"]:
                                alpha = min(elapsed / 3, 1) if stage == "lift" else 1
                                desired += np.array([-0.04, 0, 0.16]) * alpha
                                if stage == "lift" and elapsed > 3:
                                    stage = "hold"
                                    stage_start = control_time
                                    elapsed = 0.0
                                if stage == "hold" and elapsed > 2:
                                    stage = "release"
                                    stage_start = control_time
                                    elapsed = 0.0
                        desired_rot = (
                            Rotation.from_rotvec((1 - alpha) * arm_rot).as_matrix()
                            if stage == "reach"
                            else np.eye(3)
                        )
                        if context_policy is not None:
                            desired, desired_rot, contextual_grip = context_policy(
                                control_time - manipulation_start,
                                object_goal,
                                observer.kinematic_data,
                            )
                        candidate = arm_ik(
                            m, observer.kinematic_data, desired, arm_seed, desired_rot
                        )
                        arm_seed += np.clip(candidate - arm_seed, -0.015, 0.015)
                        target[22:29] = arm_seed
                        grip_target = (
                            0.028
                            if stage in ["close", "lift", "hold"]
                            and not args.open_gripper
                            else 0
                        )
                        if context_policy is not None:
                            grip_target = contextual_grip
                if (
                    planner is None
                    and stage == "release"
                    and control_time - stage_start > 2
                ):
                    break
            if step % 2500 == 0:
                print(
                    json.dumps(
                        {
                            "progress_seconds": d.time,
                            "stage": stage,
                            "controller": controller_state,
                        }
                    ),
                    flush=True,
                )
            poses.append(d.qpos.copy())
            contact_names = []
            pad_forces = {"pad_a": 0.0, "pad_b": 0.0}
            feet = [False, False]
            if obj is not None:
                max_lift = max(max_lift, d.xpos[obj, 2] - (0.805 + shift))
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
                    if d.xpos[obj, 2] > 0.905 + shift:
                        held_seconds += 0.02
            if handling:
                # All world-label reads in this block are evaluation only.
                receiving_surface = d.geom_xpos[receiving_id].copy()
                receiving_surface[2] += m.geom_size[receiving_id, 2]
                object_speed = float(
                    np.linalg.norm(d.qvel[object_velocity : object_velocity + 3])
                )
                hand_object_distance = float(
                    np.linalg.norm(d.site_xpos[site] - d.xpos[obj])
                )
                receiving_contact = "receiving_marker" in contact_names
                on_target = (
                    np.max(np.abs(d.xpos[obj, :2] - receiving_surface[:2])) <= 0.065
                    and abs(d.xpos[obj, 2] - receiving_surface[2] - 0.025) <= 0.01
                )
                pads_touch = any(p in contact_names for p in ("pad_a", "pad_b"))
                valid_placement = (
                    planner is not None
                    and planner.phase in ("withdraw", "complete")
                    and receiving_contact
                    and on_target
                    and object_speed < 0.03
                    and not pads_touch
                    and grip_target <= 1e-5
                    and hand_object_distance > 0.10
                )
                placement_streak = placement_streak + 0.02 if valid_placement else 0.0
                placement_longest = max(placement_longest, placement_streak)
            for idx in range(2):
                if feet[idx] and not previous_feet[idx] and stage == "walk":
                    foot_counts[idx] += 1
            previous_feet = feet
            records.append(
                {
                    "t": d.time,
                    "control_time": control_time,
                    "stage": stage,
                    "base": d.qpos[:3].tolist(),
                    "hand": d.site_xpos[site].tolist() if site is not None else [],
                    "object": d.xpos[obj].tolist() if obj is not None else [],
                    "contacts": contact_names,
                    "pad_force_n": pad_forces,
                    "feet_contact": feet,
                    "observation": obs.tolist(),
                    "action": action.tolist(),
                    "desired_hand": desired.tolist()
                    if stage in ["reach", "close", "lift", "hold", "release"]
                    else [],
                    "desired_rotation": desired_rot.reshape(-1).tolist()
                    if stage in ["reach", "close", "lift", "hold", "release"]
                    else [],
                    "grip_command": grip_target,
                    "target_anchor": object_goal.tolist(),
                    "sensor_state": sensed,
                    "controller_state": controller_state,
                    **(
                        {
                            "receiving_surface_xyz": receiving_surface.tolist(),
                            "object_speed_m_s": object_speed,
                            "hand_object_distance_m": hand_object_distance,
                            "receiving_contact": receiving_contact,
                        }
                        if handling
                        else {}
                    ),
                }
            )
        if handling and planner is not None and planner.done:
            break
        d.ctrl[:29] = kp * (target - d.qpos[7:36]) - kd * d.qvel[6:35]
        if args.task and stage in ["reach", "close", "lift", "hold", "release"]:
            d.ctrl[22:29] = (
                np.array([30] * 4 + [12] * 3) * (target[22:29] - d.qpos[29:36])
                - np.array([2] * 4 + [0.8] * 3) * d.qvel[28:35]
                + observer.kinematic_data.qfrc_bias[28:35]
                + (planner.arm_feedforward if handling and planner is not None else 0.0)
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
                feedforward[22:29] = observer.kinematic_data.qfrc_bias[28:35] + (
                    planner.arm_feedforward if handling and planner is not None else 0.0
                )
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
        "planned": getattr(args, "planned", False),
        "source_sha256": source_hashes,
        "controller_failed": planner.failed if planner is not None else False,
        "controller_inputs": "RGB-D color/depth, IMU orientation/angular rate, joint encoders, foot contacts; stance odometry"
        + ("; own-pad tactile forces" if planner is not None else ""),
        "sensing_lost": sensing_lost,
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
    if handling:
        result.update(
            task="visual_pick_transport_place",
            load_compensation=getattr(args, "load_compensation", True),
            receiving_surface_xyz=None
            if receiving_surface is None
            else receiving_surface.tolist(),
            placement_settled_seconds=placement_longest,
            placement_xy_error_m=None
            if receiving_surface is None
            else float(
                np.linalg.norm(
                    np.asarray(records[-1]["object"][:2]) - receiving_surface[:2]
                )
            ),
            hand_clearance_m=hand_object_distance,
            object_speed_m_s=object_speed,
            controller_phase=planner.phase if planner else None,
            placement_thresholds={
                "tray_half_extent_m": 0.10,
                "cube_half_extent_m": 0.025,
                "xy_center_tolerance_m": 0.065,
                "height_tolerance_m": 0.01,
                "max_object_speed_m_s": 0.03,
                "min_hand_distance_m": 0.10,
                "min_settled_seconds": 1.0,
            },
            success=bool(
                minimum > 0.65
                and walk_distance > 0.7
                and held_seconds >= 1
                and placement_longest >= 1
                and planner is not None
                and planner.phase == "complete"
                and not planner.failed
            ),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    if hasattr(observer, "save_evidence"):
        observer.save_evidence(args.output)
    if hasattr(observer, "close"):
        observer.close()
    (args.output / "scene.xml").write_text(xml)
    (args.output / "records.json").write_text(json.dumps(records))
    if planner is not None:
        (args.output / "events.json").write_text(json.dumps(planner.events, indent=2))
        if planner.ik.compiled_solver is not None:
            (args.output / "controller-algorithm.json").write_text(
                planner.ik.compiled_solver.algorithm.to_json()
            )
    np.savez_compressed(args.output / "probe.npz", qpos=np.array(poses))
    (args.output / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
