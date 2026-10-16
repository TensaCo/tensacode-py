"""Frozen native SIMPLE comparison; execute only inside an admitted GPU lease.

Imports existing external stacks read-only. One arm/process, all conditions.
No weight training, simulator truth inputs, or outcome-driven condition selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def transform_rgb(image, condition: str):
    """Fixed image-coordinate sensor corruption, independent of object state."""
    import numpy as np

    if condition not in {"nominal", "paraphrase", "occlusion"}:
        raise ValueError(f"Unknown condition: {condition}")
    result = np.asarray(image).copy()
    if condition == "occlusion":
        height, width = result.shape[:2]
        dh, dw = int(height * 0.15), int(width * 0.15)
        y, x = (height - dh) // 2, (width - dw) // 2
        result[y : y + dh, x : x + dw] = 127
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=["psi01", "bc", "released", "psi01-no-logits"], required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = Path(__file__).with_name("native_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    ckpt = (
        args.run_dir
        / "checkpoints"
        / f"ckpt_{protocol['step'][args.arm]}"
        / "model.safetensors"
    )
    checkpoint_hash = digest(ckpt)
    if checkpoint_hash != protocol["checkpoint_sha256"][args.arm]:
        raise SystemExit("Checkpoint mismatch; refusing inference")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, args.output / protocol_path.name)
    shutil.copy2(__file__, args.output / "native_eval.py")

    import numpy as np
    import psi01.evaluate as E
    import rrp.policies.psi0 as P
    from psi01 import patch
    from rrp.envs.simple import SimpleEnv
    from rrp.harness.eval.evaluate import _json_default, evaluate, task_hooks

    # Check the exact code of the structural loader and evaluator, not only git HEAD.
    expected_code = {
        "evaluate.py": "d4d95650ee91399880e935e9621265c98d062a2c2a17c0aae8ca4a0366631298",
        "patch.py": "4ee7b1a01da5bc275ee2a00b3c34af4ea4b70d670db16d67906a41e802e34043",
        "fields.py": "f099d8ba45c8176f9f9f84e4a581129688acd36d68d7c925e2ca32db0fa059df",
    }
    code_dir = Path(E.__file__).parent
    for name, expected in expected_code.items():
        if digest(code_dir / name) != expected:
            raise SystemExit(f"External Psi-0.1 code mismatch: {name}")
    metadata = {
        "arm": args.arm,
        "checkpoint_sha256": checkpoint_hash,
        "protocol_sha256": digest(protocol_path),
        "runner_sha256": digest(Path(__file__)),
        "external_code_sha256": expected_code,
        "task": protocol["task"],
        "action_tracker": protocol["action_tracker"],
        "fresh_evaluation": True,
        "ablation": protocol["ablation"] if args.arm == "psi01-no-logits" else None,
        "observation_boundary": "RGB + ideal proprioception + task instruction; privileged SIMPLE judge only",
    }
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    task = protocol["task"]
    source = f"{'upstream' if args.arm == 'released' else 'learned'}:{args.arm}@{protocol['step'][args.arm]}:{checkpoint_hash[:16]}"
    P.psi_runtime()
    if args.arm == "released":
        patch.install_compat()
        policy = P.make_direct(
            task=f"simple/{task}",
            weights="released",
            seed=protocol["policy_seed"],
            nfe=protocol["nfe"],
        )
        policy.run = args.run_dir
    else:
        E.check_run(args.run_dir, protocol["step"][args.arm], task)
        policy = E.build_policy(
            args.run_dir,
            protocol["step"][args.arm],
            task,
            source,
            checkpoint_hash[:16],
            {},
        )
    if args.arm == "psi01-no-logits":
        original_build = P.build_server

        def ablated_server(*a, **kw):
            server = original_build(*a, **kw)
            structural = server.model.action_header.psi01
            assert structural.specs and structural.n_morph > 0
            structural.specs = ()
            return server

        P.build_server = ablated_server

    original_infer = P.server_infer
    started = time.monotonic()
    for condition in protocol["conditions"]:
        cid = condition["id"]
        condition_dir = args.output / cid
        condition_dir.mkdir()
        calls = []

        def observed_infer(
            server, image, state32, instruction, reset, _cid=cid, _calls=calls
        ):
            assert server.num_inference_steps == protocol["nfe"]
            assert server.Ta == protocol["exec_horizon"]
            observed = transform_rgb(image, _cid)
            _calls.append(
                {
                    "instruction": str(instruction),
                    "reset": bool(reset),
                    "rgb_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                    "state_sha256": hashlib.sha256(
                        np.asarray(state32).tobytes()
                    ).hexdigest(),
                }
            )
            return original_infer(server, observed, state32, instruction, reset)

        P.server_infer = observed_infer
        old_reset, old_step = SimpleEnv.reset, SimpleEnv.step
        cap = E.FrameCapture()
        captured_reset = SimpleEnv.reset
        initial_receipts = {}

        def audited_reset(
            env, seed=None, _reset=captured_reset, _receipts=initial_receipts
        ):
            observation = _reset(env, seed)
            truth = env.truth()  # Audit-only sink; never supplied to policy.
            receipt = {
                "source": "privileged:initial-reset-audit-only",
                "config": int(seed or 0),
                "level": protocol["scene_level"],
                "joint_qpos": env._obs["joint_qpos"],
                "pelvis": truth["pelvis"],
                "objects": truth["objects"],
                "palm": truth["palm"],
                "target_name": truth["target_name"],
                "scope": "Observable joint state plus privileged root/object/palm poses; not full simulator state",
            }
            encoded = json.dumps(receipt, sort_keys=True, default=_json_default)
            _receipts[int(seed or 0)] = {
                "receipt": json.loads(encoded),
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            }
            return observation

        SimpleEnv.reset = audited_reset
        # Same transformed camera shown to policy; preserve original capture cadence.
        cap._img = lambda env, _cid=cid: transform_rgb(env._obs["image"], _cid)
        env_kw = {
            "level": protocol["scene_level"],
            "render_profile": protocol["render"],
        }
        if condition["instruction"] is not None:
            env_kw["instruction"] = condition["instruction"]
        try:
            episodes = evaluate(
                policy,
                "simple",
                f"simple/{task}",
                "g1_simple",
                protocol["configs"],
                batch=protocol["batch"],
                hooks=task_hooks(f"simple/{task}", "simple"),
                env_kw=env_kw,
            )
        finally:
            SimpleEnv.reset, SimpleEnv.step = old_reset, old_step
            P.server_infer = original_infer
        rows = []
        for episode in episodes:
            row = episode.row()
            assert row["provenance"]["env"]["render"] == protocol["render"]
            assert row["provenance"]["env"]["level"] == protocol["scene_level"]
            rows.append(
                {
                    "arm": args.arm,
                    "condition": cid,
                    "config": row["seed"],
                    "repeat": 0,
                    "level": protocol["scene_level"],
                    "task": task,
                    "success": bool(row["success_privileged"]),
                    "outcome": row["outcome"],
                    "steps": row["steps"],
                    "wall_s": row["wall_s"],
                    "source": source,
                    "checkpoint_sha256": checkpoint_hash,
                    "protocol_sha256": metadata["protocol_sha256"],
                    "initial_observed_rgb_sha256": hashlib.sha256(
                        cap.frames[(row["seed"], 0)][0][1].tobytes()
                    ).hexdigest(),
                    "initial_reset_audit": initial_receipts[row["seed"]],
                    "native_row": row,
                }
            )
        assert {r["config"] for r in rows} == set(protocol["configs"]) and len(
            rows
        ) == 4
        (condition_dir / "rows.jsonl").write_text(
            "".join(json.dumps(r, default=_json_default) + "\n" for r in rows)
        )
        (condition_dir / "policy_inputs.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in calls)
        )
        E.write_videos(
            cap,
            rows,
            len(rows),
            condition_dir / "video",
            f"{source}; condition={cid}",
            task,
            protocol["scene_level"],
        )
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "condition": cid,
                    "successes": sum(r["success"] for r in rows),
                    "attempted": len(rows),
                    "elapsed_s": time.monotonic() - started,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
