"""Execute grounded task assignments with concurrent robots in one MuJoCo world.

Controller state comes from shared rendered RGB-D and ideal local proprioception;
world object/root positions below are accessed only by the separate scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROBOTICS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTICS))
sys.path.insert(0, str(ROBOTICS / "transfer"))
from g1.model import ASSETS, configuration, observation
from handling import PickPlace
from warehouse.coordination import Coordinator, RobotState
from warehouse.language import (
    GroundingError,
    ground_instruction,
    instruction_from_dict,
    parse_instruction,
)
from warehouse.physics_recording import policy_action, policy_observation, save_episode
from warehouse.physics_scenario import normalize_scenario, perturb_scene
from warehouse.physics_scene import compose_scene, initialize
from warehouse.physics_sensors import SharedSensorBelief
from warehouse.representation import Entity, SceneGraph


class RobotController:
    def __init__(
        self,
        index,
        local,
        shared,
        offset,
        session,
        cfg,
        *,
        scenario=None,
        grasp_recovery=False,
    ):
        self.index, self.id, self.model = index, f"r{index}", local
        self.grasp_recovery = grasp_recovery
        self.scenario = scenario or normalize_scenario({}, 1)
        self.observer = SharedSensorBelief(
            local,
            shared,
            self.id + "/",
            offset,
            sensor_interventions=self.scenario["sensors"],
            noise_seed=self.scenario["seed"] + index,
        )
        self.command = np.zeros(3)
        self.session = session
        self.default = np.array(cfg["default_joint_pos"])
        self.kp, self.kd = np.array(cfg["stiffness"]), np.array(cfg["damping"])
        self.scale = np.array(cfg["actions"]["JointPositionAction"]["scale"])
        self.action = np.zeros(29)
        self.target = self.default.copy()
        self.arm = None
        self.grip = 0.0
        self.phase = "waiting"
        self.phase_start = 0.0
        self.assignment = None
        self.state = {}
        self.sensed = None
        self.object_goal = None
        self.finished = False
        mapping = sorted(self.observer._joint_map)
        self.encoder_q = np.array([row[1] for row in mapping])
        self.encoder_v = np.array([row[3] for row in mapping])

    def observe(self, data):
        self.sensed = self.observer.observe(data)
        return self.sensed

    def assign(self, assignment, time):
        if (
            assignment.object_id != f"parcel_{self.index}"
            or assignment.destination_id != f"receiver_{self.index}"
        ):
            raise ValueError(
                "Current calibrated controller supports its own workcell only"
            )
        self.assignment = assignment
        self.phase, self.phase_start = "walk", time

    def control(self, time):
        sensed, proprio = self.sensed, self.observer.proprio
        command = np.zeros(3)
        if self.assignment is not None:
            if self.phase in ("walk", "settle"):
                self.object_goal = np.array(sensed["object_position"]) + [0, 0, 0.009]
            if self.object_goal is not None:
                delta = (
                    self.object_goal[:2]
                    + [-0.25, 0.25]
                    - np.array(sensed["root_position"][:2])
                )
                quat = proprio.qpos[3:7]
                yaw = math.atan2(
                    2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                    1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
                )
                command = np.r_[
                    np.clip(
                        np.array(
                            [
                                [math.cos(yaw), math.sin(yaw)],
                                [-math.sin(yaw), math.cos(yaw)],
                            ]
                        )
                        @ delta
                        * 1.5,
                        -0.35,
                        0.35,
                    ),
                    np.clip(-1.5 * yaw, -0.4, 0.4),
                ]
                if self.phase == "walk" and np.linalg.norm(delta) < 0.055:
                    self.phase, self.phase_start = "settle", time
                if self.phase == "settle" and time - self.phase_start > 1.5:
                    arm_type = PickPlace
                    if self.grasp_recovery:
                        from warehouse.physics_recovery import TactilePickPlace

                        arm_type = TactilePickPlace
                    self.arm = arm_type(
                        self.model,
                        self.observer.kinematic_data,
                        self.object_goal - [0, 0, 0.009],
                        time,
                        load_compensation=self.scenario["load_compensation"],
                    )
                    self.phase = "approach"
        self.command = command.copy()
        obs = observation(proprio, self.action, command, self.default)
        if self.arm is not None:
            obs[33:40] = 0
            obs[62:69] = 0
        self.action = self.session.run(None, {"obs": obs[None]})[0][0]
        self.target = self.default + self.scale * self.action
        if self.arm is not None:
            joints, _desired, _rotation, self.grip, self.state = self.arm.step(
                self.observer.kinematic_data, sensed, time
            )
            self.target[22:29] = joints
            self.phase = self.arm.phase
            self.finished = self.arm.done

    def servo(self, shared_data):
        # Ideal own scalar joint encoders are sampled at the 500Hz servo rate.
        q = shared_data.qpos[self.encoder_q]
        v = shared_data.qvel[self.encoder_v]
        torque = self.kp * (self.target - q[:29]) - self.kd * v[:29]
        if self.arm is not None:
            torque[22:29] = (
                np.array([30] * 4 + [12] * 3) * (self.target[22:29] - q[22:29])
                - np.array([2] * 4 + [0.8] * 3) * v[22:29]
                + self.observer.kinematic_data.qfrc_bias[28:35]
                + self.arm.arm_feedforward
            )
        return np.r_[torque, 300 * (self.grip - q[29:31]) - 3 * v[29:31]]


def score_robot(model, data, index, controller, previous):
    """Evaluation-only access to true object/support poses and contact identity."""
    prefix = f"r{index}/"
    obj = model.body(prefix + "object").id
    marker = model.geom(prefix + "receiving_marker").id
    cube = model.geom(prefix + "cube").id
    pads = {model.geom(prefix + name).id for name in ("pad_a", "pad_b")}
    contacts = []
    for contact in data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube in pair:
            contacts.extend(model.geom(g).name for g in pair - {cube})
    surface = data.geom_xpos[marker].copy()
    surface[2] += model.geom_size[marker, 2]
    position = data.xpos[obj].copy()
    site = model.site(prefix + "grasp").id
    hand_distance = float(np.linalg.norm(data.site_xpos[site] - position))
    adr = model.jnt_dofadr[model.joint(prefix + "object_free").id]
    speed = float(np.linalg.norm(data.qvel[adr : adr + 3]))
    valid = (
        prefix + "receiving_marker" in contacts
        and max(abs(position[:2] - surface[:2])) <= 0.065
        and abs(position[2] - surface[2] - 0.025) <= 0.01
        and speed < 0.03
        and not any(model.geom(p).name in contacts for p in pads)
        and controller.grip <= 1e-5
        and hand_distance > 0.1
    )
    streak = previous.get("streak", 0) + 0.02 if valid else 0.0
    longest = max(previous.get("settled_seconds", 0), streak)
    rootadr = model.jnt_qposadr[model.joint(prefix + "floating_base_joint").id]
    minimum = min(
        previous.get("minimum_root_height", 10), float(data.qpos[rootadr + 2])
    )
    return {
        "object_position": position.tolist(),
        "receiving_surface_xyz": surface.tolist(),
        "contacts": contacts,
        "object_speed_m_s": speed,
        "hand_clearance_m": hand_distance,
        "placement_xy_error_m": float(np.linalg.norm(position[:2] - surface[:2])),
        "streak": streak,
        "settled_seconds": longest,
        "minimum_root_height": minimum,
        "physical_success": bool(valid and longest >= 1 and minimum > 0.65),
        "controller_declared_completion": bool(
            getattr(controller, "predicted_done", controller.phase == "complete")
        ),
        "success": bool(valid and longest >= 1 and minimum > 0.65),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    scenario = normalize_scenario(
        json.loads(args.scenario_json.read_text()) if args.scenario_json else {},
        args.robots,
    )
    (args.output / "scenario.json").write_text(json.dumps(scenario, indent=2))
    if args.grasp_recovery:
        from warehouse.physics_recovery import RECOVERY_PROTOCOL

        (args.output / "recovery-protocol.json").write_text(
            json.dumps(RECOVERY_PROTOCOL, indent=2)
        )
    xml, local_xml, offsets = compose_scene(args.robots, args.spacing, args.assets)
    xml = perturb_scene(xml, scenario)

    def camera_configuration(source):
        root = ET.fromstring(source)
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        quality = visual.find("quality")
        if quality is None:
            quality = ET.SubElement(visual, "quality")
        quality.set("offsamples", str(args.offscreen_samples))
        return ET.tostring(root, encoding="unicode")

    xml, local_xml = camera_configuration(xml), camera_configuration(local_xml)
    (args.output / "scene.xml").write_text(xml)
    source_files = (
        list(Path(__file__).parent.rglob("*.py"))
        + [
            ROBOTICS / "transfer" / n
            for n in ("handling.py", "planning.py", "sensors.py", "box_qp.py")
        ]
        + [ROBOTICS / "g1/model.py"]
    )
    hashes = {
        str(p.relative_to(ROBOTICS)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    with zipfile.ZipFile(
        args.output / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in source_files:
            archive.write(p, str(p.relative_to(ROBOTICS)))
    model = mujoco.MjModel.from_xml_string(xml)
    data = initialize(model, args.robots, args.assets)
    initial_qpos_hash = hashlib.sha256(data.qpos.tobytes()).hexdigest()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(
            args.assets
            / "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
        ),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    controller_type = RobotController
    controller_kwargs = {"scenario": scenario, "grasp_recovery": args.grasp_recovery}
    if args.learned_policy:
        from warehouse.learned_policy.adapter import LearnedController

        controller_type = LearnedController
        if args.learned_policy_kind == "diffusion":
            from warehouse.diffusion_policy.adapter import DiffusionController

            controller_type = DiffusionController
        if args.learned_policy_kind == "dagger":
            from warehouse.learned_policy.dagger import DaggerController

            controller_type = DaggerController
        if args.learned_policy_kind == "geometry":
            from warehouse.learned_policy.geometry_adapter import GeometryController

            controller_type = GeometryController
        controller_kwargs["policy_directory"] = args.learned_policy
    controllers = [
        controller_type(
            i,
            mujoco.MjModel.from_xml_string(local_xml),
            model,
            offsets[i],
            session,
            configuration(args.assets),
            **controller_kwargs,
        )
        for i in range(args.robots)
    ]
    from OpenGL import GL

    gl_info = {
        name: (GL.glGetString(token) or b"unavailable").decode()
        for name, token in (
            ("vendor", GL.GL_VENDOR),
            ("renderer", GL.GL_RENDERER),
            ("version", GL.GL_VERSION),
        )
    }
    policy_hashes = {}
    if args.learned_policy:
        import shutil

        policy_files = ["policy.json", "weights.pt"]
        if args.learned_policy_kind == "geometry":
            policy_files.append("kinematics.json")
        for name in policy_files:
            source = args.learned_policy / name
            policy_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            (args.output / "learned-policy").mkdir(exist_ok=True)
            shutil.copyfile(source, args.output / "learned-policy" / name)
    instruction = args.instruction or " while ".join(
        f"move parcel_{i} to receiver_{i}" for i in range(args.robots)
    )
    tree = (
        instruction_from_dict(json.loads(args.instruction_json.read_text()))
        if args.instruction_json
        else parse_instruction(instruction)
    )
    scene = SceneGraph()
    coordinator = Coordinator(substrate=args.dispatch_substrate)
    tasks = None
    records = []
    poses = []
    events = []
    scores = [{} for _ in controllers]
    lastphase = ["waiting"] * args.robots
    finished = set()
    policy_steps = [[] for _ in controllers]
    control_latencies = [[] for _ in controllers]
    arm_slew_rad = getattr(args, "arm_slew_rad", None)
    if arm_slew_rad is not None:
        from warehouse.physics_actuation import ArmSlewLimiter

        arm_limiters = [ArmSlewLimiter(arm_slew_rad) for _ in controllers]
    else:
        arm_limiters = [None for _ in controllers]
    arm_slew_steps = [None for _ in controllers]
    termination_reason = "horizon"
    for step in range(int(args.seconds / 0.002)):
        if step % 10 == 0:
            now = float(data.time)
            for i, robot in enumerate(controllers):
                sensed = robot.observe(data)
                for kind, key, agekey, color in [
                    ("parcel", "object_position", "object_measurement_age_s", "orange"),
                    (
                        "receiver",
                        "destination_position",
                        "destination_measurement_age_s",
                        "magenta",
                    ),
                ]:
                    if sensed[key] is not None and sensed[agekey] is not None:
                        scene.upsert(
                            Entity(
                                f"{kind}_{i}",
                                "parcel" if kind == "parcel" else "station",
                                {"color": color, "lane": str(i)},
                                tuple(sensed[key]),
                                now - sensed[agekey],
                                f"{robot.id}/shared_rgbd",
                                0.95,
                            )
                        )
            if tasks is None:
                try:
                    tasks = ground_instruction(tree, scene, now)
                except GroundingError:
                    pass
                else:
                    events.append(
                        {
                            "t": now,
                            "type": "grounded",
                            "tasks": [asdict(t) for t in tasks],
                        }
                    )
            if tasks is not None:
                assignments = coordinator.dispatch(
                    tasks,
                    [
                        RobotState(
                            r.id,
                            tuple(r.sensed["root_position"]),
                            available=r.assignment is None,
                            allowed_object_ids=(f"parcel_{r.index}",),
                            allowed_destination_ids=(f"receiver_{r.index}",),
                        )
                        for r in controllers
                    ],
                    scene,
                    now,
                )
                for assignment in assignments:
                    robot = controllers[int(assignment.robot_id[1:])]
                    robot.assign(assignment, now)
                    events.append(dict(t=now, type="assigned", **asdict(assignment)))
            for i, robot in enumerate(controllers):
                before = policy_observation(robot) if args.bc_record else None
                control_started = time.perf_counter_ns()
                robot.control(now)
                control_latencies[i].append(
                    (time.perf_counter_ns() - control_started) / 1e6
                )
                if arm_limiters[i] is not None:
                    arm_slew_steps[i] = arm_limiters[i].apply(robot)
                if args.bc_record:
                    policy_steps[i].append(
                        {
                            "t": now,
                            "observation": before,
                            "action": policy_action(robot),
                        }
                    )
                scores[i] = score_robot(model, data, i, robot, scores[i])
                if robot.phase != lastphase[i]:
                    events.append(
                        {
                            "t": now,
                            "type": "phase",
                            "robot": robot.id,
                            "phase": robot.phase,
                        }
                    )
                    lastphase[i] = robot.phase
                if robot.finished and robot.assignment.task_id not in finished:
                    task_id = robot.assignment.task_id
                    finished.add(task_id)
                    coordinator.finish(
                        task_id,
                        robot.phase == "complete" and not robot.arm.failed,
                        f"records.json@{now:.2f}/{robot.id}",
                    )
                    events.append(
                        {
                            "t": now,
                            "type": "task_finished",
                            "robot": robot.id,
                            "success": scores[i]["success"],
                            "controller_success": robot.phase == "complete"
                            and not robot.arm.failed,
                            "evaluation_success": scores[i]["physical_success"],
                            "task_id": task_id,
                        }
                    )
            rows = {
                r.id: {
                    "phase": r.phase,
                    "sensor_state": r.sensed,
                    "controller_state": r.state,
                    "grip_command": r.grip,
                    "control_latency_ms": control_latencies[i][-1],
                    "arm_slew": arm_slew_steps[i],
                    "task_id": r.assignment.task_id if r.assignment else None,
                    "evaluation": scores[i],
                }
                for i, r in enumerate(controllers)
            }
            records.append({"t": now, "robots": rows})
            poses.append(data.qpos.copy())
            if step % 2500 == 0:
                print(
                    json.dumps({"t": now, "phases": [r.phase for r in controllers]}),
                    flush=True,
                )
            if (
                not args.learned_policy
                and not args.full_horizon
                and tasks is not None
                and len(finished) == len(tasks)
            ):
                termination_reason = "controller_finished"
                break
            if any(s["minimum_root_height"] < 0.4 for s in scores):
                termination_reason = "fall_harness_stop"
                break
        for i, robot in enumerate(controllers):
            data.ctrl[i * 31 : (i + 1) * 31] = robot.servo(data)
        mujoco.mj_step(model, data)
    np.savez_compressed(args.output / "probe.npz", qpos=np.asarray(poses))
    (args.output / "records.json").write_text(json.dumps(records))
    (args.output / "events.json").write_text(json.dumps(events, indent=2))
    (args.output / "scene-graph.json").write_text(json.dumps(scene.to_dict(), indent=2))
    physical_success = bool(tasks and all(s["physical_success"] for s in scores))
    report = {
        "schema": "tensorcode.warehouse.physics.v4",
        "arm_slew_rad": arm_slew_rad,
        "arm_slew": {
            r.id: limiter.summary if limiter is not None else None
            for r, limiter in zip(controllers, arm_limiters)
        },
        "control_latency": {
            "scope": "robot.control including own controller, frozen walker and IK or learned high-level runtime; excludes rendering, physics and recorder; empirical, not worst-case execution guarantee",
            "unit": "ms",
            "robots": {
                r.id: {
                    "calls": len(control_latencies[i]),
                    "p50": float(np.percentile(control_latencies[i], 50)),
                    "p95": float(np.percentile(control_latencies[i], 95)),
                    "max": float(max(control_latencies[i])),
                }
                for i, r in enumerate(controllers)
            },
        },
        "grasp_recovery": bool(args.grasp_recovery),
        "physical_success": physical_success,
        "full_horizon": bool(args.full_horizon or args.learned_policy),
        "requested_duration_s": args.seconds,
        "termination_reason": termination_reason,
        "ever_physical_success": {
            r.id: any(
                row["robots"][r.id]["evaluation"]["physical_success"] for row in records
            )
            for r in controllers
        },
        "controller_declared_completion": {
            r.id: bool(getattr(r, "predicted_done", r.phase == "complete"))
            for r in controllers
        },
        "scenario": scenario,
        "render_backend": os.environ.get("MUJOCO_GL", "default"),
        "opengl": gl_info,
        "initial_qpos_sha256": initial_qpos_hash,
        "learned_policy_sha256": policy_hashes,
        "mujoco_version": mujoco.__version__,
        "offscreen_samples": int(model.vis.quality.offsamples),
        "policy": ("learned_" + args.learned_policy_kind)
        if args.learned_policy
        else "authored",
        "learned_policy": str(args.learned_policy) if args.learned_policy else None,
        "success": bool(
            tasks
            and (bool(args.learned_policy) or len(coordinator.completed) == len(tasks))
            and all(
                scores[i]["success"]
                for i, robot in enumerate(controllers)
                if robot.assignment is not None
            )
        ),
        "robots": args.robots,
        "seconds": float(data.time),
        "instruction": instruction,
        "instruction_tree": tree.to_dict(),
        "tasks": [] if tasks is None else [asdict(t) for t in tasks],
        "outcomes": coordinator.outcomes,
        "evaluation": {f"r{i}": s for i, s in enumerate(scores)},
        "source_sha256": hashes,
        "shared_physics": True,
        "dispatch_substrate": args.dispatch_substrate,
        "scope": "Calibrated independent workcells with grounded dispatch; no physical handoff or learned visual semantics",
        "sensor_contract": "Shared rendered RGB-D; ideal own joint encoders/velocities and IMU attitude/angular rate; contact/vision root odometry; initial root calibration; no live root translation or object pose in controller",
    }
    (args.output / "probe.json").write_text(json.dumps(report, indent=2))
    for i, robot in enumerate(controllers):
        if args.bc_record:
            save_episode(
                args.output / robot.id / "bc-episode.json.gz",
                policy_steps[i],
                robot_id=robot.id,
                scenario=scenario,
                physical_success=scores[i]["physical_success"],
            )
        if args.learned_policy and args.learned_policy_kind == "dagger":
            robot.save_dagger(args.output / robot.id / "dagger-episode.json.gz")
        robot.observer.save_evidence(args.output / robot.id)
        robot.observer.close()
    print(
        json.dumps({"success": report["success"], "seconds": report["seconds"]}),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=1.6)
    parser.add_argument("--seconds", type=float, default=110)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--scenario-json", type=Path)
    parser.add_argument("--bc-record", action="store_true")
    parser.add_argument("--full-horizon", action="store_true")
    parser.add_argument("--grasp-recovery", action="store_true")
    parser.add_argument("--offscreen-samples", type=int, choices=[0, 4], default=4)
    parser.add_argument("--arm-slew-rad", type=float)
    parser.add_argument("--learned-policy", type=Path)
    parser.add_argument(
        "--learned-policy-kind", choices=["bc", "diffusion", "dagger", "geometry"], default="bc"
    )
    parser.add_argument("--instruction")
    parser.add_argument("--instruction-json", type=Path)
    parser.add_argument(
        "--dispatch-substrate", choices=["alg0", "alg1", "alg2"], default="alg0"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
