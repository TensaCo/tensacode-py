"""Execute repeated pickup jobs with concurrent humanoids in one physical world."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROBOTICS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOTICS))
sys.path.insert(0, str(ROBOTICS / "transfer"))
from g1.model import ASSETS, configuration
from warehouse.coordination import Coordinator, RobotState
from warehouse.jobs import Capability, JobScheduler
from warehouse.language import (
    GroundingError,
    ground_instruction,
    instruction_from_dict,
    parse_instruction,
)
from warehouse.physics_jobs_controller import (
    ARM_SERVO_PROTOCOL,
    TRANSITION_PROTOCOL,
    RepeatedJobController,
)
from warehouse.physics_jobs_scene import compose_jobs_scene
from warehouse.physics_scene import initialize
from warehouse.representation import SceneGraph


def score_job(model, data, index, job, controller, previous):
    """Separate evaluator only; never passed into scheduling or motor targets."""
    prefix = f"r{index}/"
    suffix = "" if job == "a" else "_b"
    obj = model.body(prefix + "object" + suffix).id
    cube = model.geom(prefix + "cube" + suffix).id
    marker = model.geom(prefix + "receiving_marker" + suffix).id
    contacts = []
    for contact in data.contact:
        pair = {int(contact.geom1), int(contact.geom2)}
        if cube in pair:
            contacts.extend(model.geom(g).name for g in pair - {cube})
    surface = data.geom_xpos[marker].copy()
    surface[2] += model.geom_size[marker, 2]
    position = data.xpos[obj].copy()
    hand = data.site_xpos[model.site(prefix + "grasp").id]
    distance = float(np.linalg.norm(hand - position))
    adr = model.jnt_dofadr[model.joint(prefix + "object_free" + suffix).id]
    speed = float(np.linalg.norm(data.qvel[adr : adr + 3]))
    root = model.jnt_qposadr[model.joint(prefix + "floating_base_joint").id]
    minimum = min(previous.get("minimum_root_height", 10), float(data.qpos[root + 2]))
    valid = (
        prefix + "receiving_marker" + suffix in contacts
        and max(abs(position[:2] - surface[:2])) <= 0.065
        and abs(position[2] - surface[2] - 0.025) <= 0.01
        and speed < 0.03
        and not any(prefix + pad in contacts for pad in ("pad_a", "pad_b"))
        and controller.grip <= 1e-5
        and distance > 0.1
    )
    streak = previous.get("streak", 0) + 0.02 if valid else 0.0
    return {
        "object_position": position.tolist(),
        "receiving_surface_xyz": surface.tolist(),
        "contacts": contacts,
        "object_speed_m_s": speed,
        "hand_clearance_m": distance,
        "placement_xy_error_m": float(np.linalg.norm(position[:2] - surface[:2])),
        "streak": streak,
        "settled_seconds": max(previous.get("settled_seconds", 0), streak),
        "minimum_root_height": minimum,
        "physical_success": bool(valid and streak >= 1 and minimum > 0.65),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    xml, local_xml, offsets = compose_jobs_scene(args.robots, args.spacing, args.assets)
    (args.output / "scene.xml").write_text(xml)
    files = (
        list(Path(__file__).parent.rglob("*.py"))
        + [
            ROBOTICS / "transfer" / name
            for name in ("handling.py", "planning.py", "sensors.py", "box_qp.py")
        ]
        + [ROBOTICS / "g1/model.py"]
    )
    hashes = {
        str(p.relative_to(ROBOTICS)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    with zipfile.ZipFile(
        args.output / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in files:
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
    controllers = [
        RepeatedJobController(
            i,
            mujoco.MjModel.from_xml_string(local_xml),
            model,
            offsets[i],
            session,
            configuration(args.assets),
            grasp_recovery=True,
        )
        for i in range(args.robots)
    ]
    instruction = args.instruction or " then ".join(
        " while ".join(
            f"move parcel_{i}_{job} to receiver_{i}_{job}" for i in range(args.robots)
        )
        for job in ("a", "b")
    )
    tree = (
        instruction_from_dict(json.loads(args.instruction_json.read_text()))
        if args.instruction_json
        else parse_instruction(instruction)
    )
    scene = SceneGraph()
    coordinator = Coordinator(substrate=args.dispatch_substrate)
    scheduler = None
    tasks = None
    finished = set()
    records = []
    poses = []
    events = []
    scores = [{job: {} for job in ("a", "b")} for _ in controllers]
    lastphase = ["waiting"] * args.robots
    termination = "horizon"
    waiting_captured = False
    for step in range(int(args.seconds / 0.002)):
        if step % 10 == 0:
            now = float(data.time)
            for robot in controllers:
                robot.observe(data)
                for entity in robot.scene_entities(now):
                    scene.upsert(entity)
            if tasks is None:
                try:
                    tasks = ground_instruction(tree, scene, now)
                except GroundingError:
                    pass
                else:
                    allowed = {
                        (f"parcel_{i}_{job}", f"receiver_{i}_{job}")
                        for i in range(args.robots)
                        for job in ("a", "b")
                    }
                    if any(
                        (task.object_id, task.destination_id) not in allowed
                        for task in tasks
                    ):
                        raise ValueError(
                            "Mission exceeds calibrated own-workcell exact-pair capabilities"
                        )
                    (args.output / "initial-scene-graph.json").write_text(
                        json.dumps(scene.to_dict())
                    )
                    scheduler = JobScheduler(
                        tasks,
                        tuple(
                            Capability(
                                f"r{i}",
                                tuple(f"parcel_{i}_{job}" for job in ("a", "b")),
                                tuple(f"receiver_{i}_{job}" for job in ("a", "b")),
                                allowed_pairs=tuple(
                                    (f"parcel_{i}_{job}", f"receiver_{i}_{job}")
                                    for job in ("a", "b")
                                ),
                            )
                            for i in range(args.robots)
                        ),
                        coordinator,
                    )
                    events.append(
                        {
                            "t": now,
                            "type": "grounded",
                            "tasks": [asdict(t) for t in tasks],
                        }
                    )
            if tasks is not None:
                assignments = scheduler.dispatch(
                    [
                        RobotState(
                            r.id,
                            tuple(r.sensed["root_position"]),
                            available=r.assignment is None,
                            allowed_object_ids=tuple(
                                f"parcel_{r.index}_{job}" for job in ("a", "b")
                            ),
                            allowed_destination_ids=tuple(
                                f"receiver_{r.index}_{job}" for job in ("a", "b")
                            ),
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
            if scheduler is not None and not waiting_captured:
                for robot in controllers:
                    if (
                        robot.assignment is None
                        and len(robot.completed_jobs) == 1
                        and len(coordinator.completed) < args.robots
                    ):
                        (args.output / "scheduler-waiting.json").write_text(
                            json.dumps(scheduler.snapshot())
                        )
                        events.append(
                            {
                                "t": now,
                                "type": "scheduler_wait",
                                "robot": robot.id,
                                "receipt": "scheduler-waiting.json",
                                "explanation": scheduler.explain(
                                    f"why is {robot.id} waiting"
                                ),
                            }
                        )
                        waiting_captured = True
                        break
            for i, robot in enumerate(controllers):
                robot.control(now)
                for job in ("a", "b"):
                    scores[i][job] = score_job(
                        model, data, i, job, robot, scores[i][job]
                    )
                if robot.phase != lastphase[i]:
                    events.append(
                        {
                            "t": now,
                            "type": "phase",
                            "robot": robot.id,
                            "job": robot.current_job,
                            "phase": robot.phase,
                        }
                    )
                    lastphase[i] = robot.phase
                if (
                    robot.finished
                    and robot.assignment
                    and robot.assignment.task_id not in finished
                ):
                    task = robot.assignment.task_id
                    finished.add(task)
                    controller_success = (
                        robot.phase == "complete" and not robot.arm.failed
                    )
                    scheduler.finish(
                        task,
                        controller_success,
                        f"records.json@{now:.2f}/{robot.id}",
                        now,
                    )
                    events.append(
                        {
                            "t": now,
                            "type": "task_finished",
                            "robot": robot.id,
                            "job": robot.current_job,
                            "task_id": task,
                            "controller_success": controller_success,
                            "evaluation_success": scores[i][robot.current_job][
                                "physical_success"
                            ],
                        }
                    )
                    if controller_success:
                        robot.release_assignment()
            rows = {
                r.id: {
                    "phase": r.phase,
                    "current_job": r.current_job,
                    "completed_jobs": list(r.completed_jobs),
                    "sensor_state": r.sensed,
                    "job_sensors": r.job_sensors,
                    "controller_state": r.state,
                    "grip_command": r.grip,
                    "task_id": r.assignment.task_id if r.assignment else None,
                    "evaluation": scores[i][r.current_job],
                    "job_evaluation": scores[i],
                }
                for i, r in enumerate(controllers)
            }
            # Deep-copy scores because each job's dictionary is replaced next tick.
            records.append({"t": now, "robots": json.loads(json.dumps(rows))})
            poses.append(data.qpos.copy())
            if step % 2500 == 0:
                print(
                    json.dumps(
                        {
                            "t": now,
                            "phases": [r.phase for r in controllers],
                            "jobs": [r.current_job for r in controllers],
                        }
                    ),
                    flush=True,
                )
            if tasks and len(finished) == len(tasks):
                termination = "controller_finished"
                break
            if any(
                s["minimum_root_height"] < 0.4
                for group in scores
                for s in group.values()
            ):
                termination = "fall_harness_stop"
                break
        for i, robot in enumerate(controllers):
            data.ctrl[i * 31 : (i + 1) * 31] = robot.servo(data)
        mujoco.mj_step(model, data)
    physical = all(s["physical_success"] for group in scores for s in group.values())
    report = {
        "schema": "tensorcode.warehouse.physics.jobs.v1",
        "robots": args.robots,
        "jobs_per_robot": 2,
        "arm_servo_protocol": ARM_SERVO_PROTOCOL,
        "transition_protocol": TRANSITION_PROTOCOL,
        "initial_qpos_sha256": initial_qpos_hash,
        "shared_physics": True,
        "success": bool(
            tasks and len(coordinator.completed) == len(tasks) and physical
        ),
        "physical_success": physical,
        "seconds": float(data.time),
        "requested_duration_s": args.seconds,
        "termination_reason": termination,
        "instruction": instruction,
        "instruction_tree": tree.to_dict(),
        "tasks": [] if tasks is None else [asdict(t) for t in tasks],
        "outcomes": coordinator.outcomes,
        "evaluation": {f"r{i}": s for i, s in enumerate(scores)},
        "source_sha256": hashes,
        "dispatch_substrate": args.dispatch_substrate,
        "scope": "Two known-color parcel/receiver pairs per calibrated workcell; real arm reset and lateral walk; no handoff, object teleport, privileged pose or evaluator-driven task unlock",
    }
    np.savez_compressed(args.output / "probe.npz", qpos=np.asarray(poses))
    for name, value in (
        ("probe.json", report),
        ("records.json", records),
        ("events.json", events),
        ("scene-graph.json", scene.to_dict()),
    ):
        (args.output / name).write_text(json.dumps(value))
    if scheduler is not None:
        (args.output / "scheduler-final.json").write_text(
            json.dumps(scheduler.snapshot())
        )
    for robot in controllers:
        for job, observer in robot.observers.items():
            observer.save_evidence(args.output / robot.id / job)
            observer.close()
    print(
        json.dumps({"success": report["success"], "seconds": report["seconds"]}),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=2.5)
    parser.add_argument("--seconds", type=float, default=160)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--instruction")
    parser.add_argument("--instruction-json", type=Path)
    parser.add_argument(
        "--dispatch-substrate", choices=["alg0", "alg1", "alg2"], default="alg0"
    )
    run(parser.parse_args())
