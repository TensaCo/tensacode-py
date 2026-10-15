"""20-joint MuJoCo policy: ALG.0 -> ALG.1 -> ALG.2, learning, export, and rollouts.

Run from the Python checkout:
  MUJOCO_GL=egl OMP_NUM_THREADS=2 .venv/bin/python examples/robotics_policy.py --output /tmp/tensorcode-robot

The robot is a fixed-base four-limb articulated mechanism, not a locomotion result.
The baseline is authored impedance control with model-based gravity compensation.
The residual learns from a supplied synthetic actuator-calibration dataset; it is
not reinforcement learning or discovery of an unknown real robot's dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tensorcode.algorithm import AlgorithmBuilder, compare, convert

N = 20
DT = 0.005


def robot_xml():
    bodies = []
    motors = []
    colors = [".2 .55 .85 1", ".85 .45 .2 1", ".3 .75 .45 1", ".7 .35 .8 1"]
    for limb in range(4):
        angle = limb * math.pi / 2
        dx, dy = math.cos(angle), math.sin(angle)
        text = f'<body name="limb{limb}" pos="{0.22 * dx} {0.22 * dy} .75">'
        for j in range(5):
            i = limb * 5 + j
            if j:
                text += f'<body pos="{0.13 * dx} {0.13 * dy} 0">'
            text += f'<joint name="j{i}" type="hinge" axis="{-dy} {dx} 0" range="-70 70" damping=".08" armature=".015"/>'
            text += f'<geom type="capsule" fromto="0 0 0 {0.13 * dx} {0.13 * dy} 0" size=".025" mass=".15" rgba="{colors[limb]}"/>'
            motors.append(
                f'<motor joint="j{i}" gear="1" ctrllimited="true" ctrlrange="-8 8"/>'
            )
        text += "</body>" * 5
        bodies.append(text)
    return (
        """<mujoco model="TensorCode 20 joint policy">
  <compiler angle="degree"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="800" offheight="600"/><quality shadowsize="2048"/></visual>
  <default><geom friction=".8 .1 .1" contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="0 -2 4" dir="0 .3 -1"/>
    <geom type="plane" size="3 3 .1" rgba=".15 .18 .23 1"/>
    <geom type="box" pos="0 0 .37" size=".12 .12 .37" rgba=".35 .4 .5 1"/>
    <geom type="box" pos="0 0 .75" size=".23 .23 .06" rgba=".15 .25 .35 1"/>
    """
        + "".join(bodies)
        + """</worldbody><actuator>"""
        + "".join(motors)
        + """</actuator></mujoco>"""
    )


def residual_algorithm():
    b = AlgorithmBuilder(name="actuator_correction")
    v = b.input("velocity", (N,))
    viscous = b.parameter("viscous", np.zeros(N))
    coulomb = b.parameter("coulomb", np.zeros(N))
    bias = b.parameter("bias", np.zeros(N))
    return b.build({"correction": viscous * v + coulomb * (5 * v).tanh() + bias})


def policy_algorithm(residual=None):
    b = AlgorithmBuilder(name="joint_impedance_policy")
    q = b.input("position", (N,))
    v = b.input("velocity", (N,))
    qd = b.input("target", (N,))
    vd = b.input("target_velocity", (N,))
    ff = b.input("feedforward", (N,))
    torque = 30 * (qd - q) + 2.5 * (vd - v) + ff
    if residual is not None:
        torque = (
            torque
            + b.include(residual, {"velocity": v}, prefix="actuator")["correction"]
        )
    return b.build({"torque": torque.clip(-8, 8)})


def coefficients():
    # Authored synthetic actuator disturbance, hidden from the deployed residual.
    return (
        0.15 + 0.1 * np.arange(N) / N,
        0.18 + 0.05 * np.arange(N) / N,
        np.linspace(-0.45, 0.45, N),
    )


def friction(velocity, scale=1.0):
    viscous, coulomb, bias = coefficients()
    return scale * (viscous * velocity + coulomb * np.tanh(5 * velocity) + bias)


def fit_residual(steps=500):
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    a = residual_algorithm()
    model = a.compile("alg2")
    train = rng.uniform(-2, 2, size=(128, N))
    test = np.random.default_rng(1701).uniform(-2.5, 2.5, size=(128, N))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    start = time.perf_counter()
    for i in range(steps):
        x = train[i % len(train)]
        target = torch.tensor(friction(x), dtype=torch.float64)
        optimizer.zero_grad()
        prediction = model({"velocity": x})["correction"]
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        errors = [
            np.mean((model({"velocity": v})["correction"].numpy() - friction(v)) ** 2)
            for v in test
        ]
    return model.artifact().freeze(), {
        "steps": steps,
        "training_cases": 128,
        "held_out_cases": 128,
        "held_out_mse": float(np.mean(errors)),
        "zero_residual_mse": float(np.mean(friction(test) ** 2)),
        "seconds": time.perf_counter() - start,
        "training_substrate": "alg2",
        "supervision": "authored synthetic actuator calibration",
    }


def target(t, phase=0.0):
    offsets = np.arange(N) * 0.31 + phase
    omega = 0.75 + np.arange(N) % 5 * 0.06
    return 0.22 * np.sin(omega * t + offsets), 0.22 * omega * np.cos(
        omega * t + offsets
    )


def rollout(
    xml,
    policy,
    *,
    target_substrate,
    seconds=6.0,
    phase=0.0,
    disturbance_scale=1.0,
    record=False,
):
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = target(0, phase)[0]
    data.qvel[:] = target(0, phase)[1]
    mujoco.mj_forward(model, data)
    executable = policy.compile(target_substrate)
    errors = []
    controls = []
    latencies = []
    trajectory = []
    frames = []
    renderer = None
    if record:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.azimuth = 135
        camera.elevation = -28
        camera.distance = 2.8
        camera.lookat[:] = [0, 0, 0.6]
    try:
        with torch.no_grad():
            for step in range(round(seconds / DT)):
                qd, vd = target(data.time, phase)
                inputs = {
                    "position": data.qpos.copy(),
                    "velocity": data.qvel.copy(),
                    "target": qd,
                    "target_velocity": vd,
                    "feedforward": data.qfrc_bias.copy(),
                }
                begin = time.perf_counter()
                out = executable(inputs)["torque"]
                latencies.append(time.perf_counter() - begin)
                tau = (
                    out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else out
                )
                if not np.isfinite(tau).all() or np.max(np.abs(tau)) > 8 + 1e-8:
                    raise RuntimeError("controller torque contract violated")
                data.ctrl[:] = tau
                data.qfrc_applied[:] = -friction(data.qvel, disturbance_scale)
                errors.append((data.qpos - qd).copy())
                controls.append(tau.copy())
                if step % 4 == 0:
                    trajectory.append(
                        {
                            "t": float(data.time),
                            "position": data.qpos.tolist(),
                            "target": qd.tolist(),
                            "torque": tau.tolist(),
                        }
                    )
                if renderer is not None and step % 8 == 0:
                    renderer.update_scene(data, camera)
                    frames.append(renderer.render().copy())
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qpos)) > 1.3:
                    raise RuntimeError("simulation state contract violated")
    finally:
        if renderer is not None:
            renderer.close()
    e = np.asarray(errors)
    u = np.asarray(controls)
    lat = np.asarray(latencies)
    return (
        {
            "substrate": target_substrate,
            "seconds": seconds,
            "phase": phase,
            "disturbance_scale": disturbance_scale,
            "joint_rmse_rad": float(np.sqrt(np.mean(e**2))),
            "worst_joint_error_rad": float(np.max(np.abs(e))),
            "max_torque_nm": float(np.max(np.abs(u))),
            "mean_compute_ms": float(lat.mean() * 1000),
            "p99_compute_ms": float(np.quantile(lat, 0.99) * 1000),
            "max_compute_ms": float(lat.max() * 1000),
            "hard_realtime_claim": False,
        },
        trajectory,
        frames,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.train_steps < 1 or not 0 < args.seconds <= 60:
        parser.error("invalid run budget")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    residual, training = fit_residual(args.train_steps)
    policy = policy_algorithm(residual)
    baseline = policy_algorithm()
    xml = robot_xml()
    (args.output / "robot.xml").write_text(xml)
    (args.output / "policy.json").write_text(policy.to_json())
    (args.output / "policy.py").write_text(policy.to_python("policy"))
    (args.output / "residual.json").write_text(residual.to_json())
    (args.output / "attention-checkpoint.json").write_text(
        json.dumps(policy.compile("alg2").checkpoint())
    )
    rng = np.random.default_rng(101)
    cases = []
    for _ in range(32):
        cases.append(
            {
                "position": rng.uniform(-0.5, 0.5, N),
                "velocity": rng.uniform(-1, 1, N),
                "target": rng.uniform(-0.3, 0.3, N),
                "target_velocity": rng.uniform(-0.5, 0.5, N),
                "feedforward": rng.uniform(-2, 2, N),
            }
        )
    roundtrips = {
        target_name: compare(policy, policy.compile(target_name), cases)
        for target_name in ("alg0", "alg1", "alg2")
    }
    roundtrips["alg2_to_alg0"] = compare(
        policy, convert(policy.compile("alg2"), "alg0"), cases
    )
    if not all(r["passed"] for r in roundtrips.values()):
        raise RuntimeError("conversion failed")
    reports = []
    selected = None
    render_error = None
    for label, p in [("baseline", baseline), ("learned", policy)]:
        targets = ("alg0",) if label == "baseline" else ("alg0", "alg1", "alg2")
        for substrate in targets:
            report, trajectory, frames = rollout(
                xml, p, target_substrate=substrate, seconds=args.seconds, phase=0.7
            )
            report["policy"] = label
            reports.append(report)
            print(json.dumps(report), flush=True)
            if label == "learned" and substrate == "alg2":
                selected = trajectory
    # Held-out trajectory phase and changed actuator disturbance, same frozen controller.
    for label, p in [("baseline", baseline), ("learned", policy)]:
        report, _, _ = rollout(
            xml,
            p,
            target_substrate="alg2" if label == "learned" else "alg0",
            seconds=args.seconds,
            phase=2.1,
            disturbance_scale=1.25,
        )
        report["policy"] = label
        report["evaluation"] = "shifted_disturbance"
        reports.append(report)
    (args.output / "trajectory.json").write_text(json.dumps(selected))
    if not args.no_render:
        try:
            _, _, frames = rollout(
                xml,
                policy,
                target_substrate="alg2",
                seconds=min(args.seconds, 6),
                phase=0.7,
                record=True,
            )
            import imageio.v3 as iio

            iio.imwrite(
                args.output / "robot.gif", np.stack(frames), duration=40, loop=0
            )
            iio.imwrite(args.output / "robot.png", frames[len(frames) // 2])
        except Exception as e:  # noqa: BLE001 - optional renderer errors are recorded, never hidden
            render_error = f"{type(e).__name__}: {e}"
    import tensorcode.algorithm as algorithm_package

    source_paths = [
        Path(__file__),
        *Path(algorithm_package.__file__).parent.glob("*.py"),
    ]
    provenance = {
        "source_sha256": {
            str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in source_paths
        },
        "versions": {
            k: importlib.metadata.version(k) for k in ("torch", "numpy", "mujoco")
        },
        "training_seed": 17,
        "calibration_test_seed": 1701,
        "conversion_seed": 101,
    }
    report = {
        "provenance": provenance,
        "robot": "fixed-base four-limb mechanism, 20 actuated hinge joints",
        "physics": "MuJoCo",
        "task": "joint trajectory tracking",
        "training": training,
        "conversion": roundtrips,
        "rollouts": reports,
        "render_error": render_error,
        "supplied": [
            "robot morphology",
            "impedance policy",
            "inverse-dynamics bias forces",
            "synthetic friction calibration labels",
        ],
        "learned": ["60 actuator correction coefficients"],
        "not_claimed": [
            "locomotion",
            "hardware readiness",
            "unknown dynamics discovery",
            "arbitrary pretrained transformer extraction",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "training": training,
                "output": str(args.output),
                "render_error": render_error,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
