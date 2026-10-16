"""One recorded action trajectory as runtime context for a frozen humanoid policy.

This is authored attention-based trajectory retargeting with an RGB-D estimated pickup anchor,
not learned vision-language ICL. Unitree's pretrained RL walker remains frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rollout import ASSETS
from rollout import main as rollout
from scipy.spatial.transform import Rotation

from tensorcode.algorithm import AlgorithmBuilder


def build_context_algorithm(n):
    b = AlgorithmBuilder(name="humanoid_demo_context")
    phase = b.input("phase")
    times = b.input("times", (n,))
    values = b.input("values", (n, 7))
    anchor = b.input("anchor", (3,))
    weights = (-600.0 * ((times - phase) * (times - phase))).softmax()
    selected = (weights.reshape(1, n) @ values).reshape(7)
    return b.build(
        {
            "hand": selected.slice(0, 3) + anchor,
            "rotation": selected.slice(3, 6),
            "grip": selected.slice(6, 7),
        }
    )


def extract_context(reference):
    records = json.loads((reference / "records.json").read_text())
    records = [r for r in records if r.get("desired_hand")]
    if not records:
        raise ValueError("Expert did not reach manipulation; no demonstration exported")
    origin = np.array(records[0]["target_anchor"])
    start = records[0]["t"]
    chosen = records[::5]
    if chosen[-1] is not records[-1]:
        chosen.append(records[-1])
    values = np.array(
        [
            np.r_[
                np.array(r["desired_hand"]) - origin,
                Rotation.from_matrix(
                    np.array(r["desired_rotation"]).reshape(3, 3)
                ).as_rotvec(),
                r["grip_command"],
            ]
            for r in chosen
        ]
    )
    return {
        "times": np.array([r["t"] - start for r in chosen]).tolist(),
        "values": values.tolist(),
        "origin": origin.tolist(),
        "source_sha256": hashlib.sha256(
            (reference / "records.json").read_bytes()
        ).hexdigest(),
    }


class Context:
    def __init__(self, context, mode, substrate="alg2"):
        self.context = context
        self.mode = mode
        self.algorithm = build_context_algorithm(len(context["times"]))
        self.machine = self.algorithm.compile(substrate)

    def __call__(self, phase, anchor, data):
        actual_anchor = (
            np.asarray(self.context["origin"]) if self.mode == "fixed" else anchor
        )
        with torch.no_grad():
            out = self.machine(
                {
                    "phase": phase,
                    "times": self.context["times"],
                    "values": self.context["values"],
                    "anchor": actual_anchor,
                }
            )
        out = {
            k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in out.items()
        }
        target = out["hand"]
        return (
            target,
            Rotation.from_rotvec(out["rotation"]).as_matrix(),
            float(out["grip"][0]),
        )


def run(args):
    context = None
    if args.mode not in ("expert", "planned", "handling"):
        context = json.loads(args.context.read_text())
    callback = (
        None if context is None else Context(context, args.mode, args.context_substrate)
    )
    a = SimpleNamespace(
        assets=args.assets,
        no_gripper=False,
        object_x=args.x,
        object_y=args.y,
        mass=args.mass,
        seconds=100.0
        if args.mode == "handling"
        else (65.0 if args.mode == "planned" else 32.0),
        handling=args.mode == "handling",
        destination=args.destination,
        camera_outage=args.camera_outage,
        load_compensation=not args.disable_load_compensation,
        planned=args.mode in ("planned", "handling"),
        speed=0.3,
        task=True,
        open_gripper=False,
        substrate="native",
        output=args.output,
        context_policy=callback,
        camera_bias=args.camera_bias,
        height_shift=args.height_shift,
        ik_backend=args.ik_backend,
    )
    rollout(a)
    if args.mode == "expert":
        (args.output / "context.json").write_text(
            json.dumps(extract_context(args.output), indent=2)
        )
    elif callback is not None:
        (args.output / "context-policy.json").write_text(callback.algorithm.to_json())
    result = json.loads((args.output / "probe.json").read_text())
    result.update(
        mode=args.mode,
        ik_backend=args.ik_backend if args.mode in ("planned", "handling") else None,
        context_substrate=args.context_substrate,
        demonstrations=0 if context is None else 1,
        gradient_updates=0,
        weights="Frozen pinned Unitree RL walker",
        camera_frame_shift=args.camera_bias,
        height_shift=args.height_shift,
        source_context_sha256=None
        if context is None
        else hashlib.sha256(args.context.read_bytes()).hexdigest(),
    )
    (args.output / "probe.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=["expert", "fixed", "context", "planned", "handling"],
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=ASSETS)
    p.add_argument("--context", type=Path)
    p.add_argument(
        "--context-substrate", default="alg2", choices=["alg0", "alg1", "alg2"]
    )
    p.add_argument("--x", type=float, default=1.25)
    p.add_argument("--y", type=float, default=-0.25)
    p.add_argument("--camera-bias", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--height-shift", type=float, default=0.0)
    p.add_argument(
        "--ik-backend", choices=["scipy", "alg0", "alg1", "alg2"], default="scipy"
    )
    p.add_argument("--destination", type=float, nargs=3, default=[1.20, -0.48, 0.78])
    p.add_argument("--camera-outage", type=float, nargs=2)
    p.add_argument(
        "--mass",
        type=float,
        default=0.08,
        help="Environment payload mass; never supplied to the controller",
    )
    p.add_argument("--disable-load-compensation", action="store_true")
    run(p.parse_args())
