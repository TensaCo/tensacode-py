"""Run the official pi0 ALOHA simulator policy, retaining unscreened evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import gym_aloha  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from openpi import transforms
from openpi.models import pi0_config, tokenizer
from openpi.policies import aloha_policy, policy
from openpi.shared import normalize
from openpi_client import image_tools


def build_policy(checkpoint: Path, device: str):
    """Same transform chain as pi0_aloha_sim; no training dataset dependency."""
    import safetensors.torch
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    config = pi0_config.Pi0Config(pytorch_compile_mode=None)
    model = PI0Pytorch(config)
    safetensors.torch.load_model(model, str(checkpoint / "model.safetensors"))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    stats = normalize.load(checkpoint / "assets/lerobot/aloha_sim_transfer_cube_human")
    return policy.Policy(
        model,
        transforms=[
            aloha_policy.AlohaInputs(),
            transforms.Normalize(stats, use_quantiles=False),
            transforms.InjectDefaultPrompt("Transfer cube"),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(
                tokenizer.PaligemmaTokenizer(config.max_token_len)
            ),
            transforms.PadStatesAndActions(config.action_dim),
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=False),
            aloha_policy.AlohaOutputs(),
        ],
        is_pytorch=True,
        pytorch_device=device,
    )


def observation(raw):
    rgb = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(raw["pixels"]["top"], 224, 224)
    )
    return {"state": raw["agent_pos"], "images": {"cam_high": rgb.transpose(2, 0, 1)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.monotonic()
    controller = build_policy(args.checkpoint, args.device)
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    raw, _ = env.reset(seed=args.seed)
    obs = observation(raw)
    np.savez_compressed(
        args.output / "initial-observation.npz",
        state=obs["state"],
        rgb=obs["images"]["cam_high"],
    )
    rows, predictions, latencies = [], [], []
    max_reward, terminated, truncated = 0.0, False, False
    with imageio.get_writer(
        args.output / "episode.mp4", fps=50, codec="libx264"
    ) as video:
        for step in range(args.max_steps):
            video.append_data(raw["pixels"]["top"])
            if step % 10 == 0:
                before = time.monotonic()
                with torch.inference_mode():
                    actions = np.asarray(controller.infer(observation(raw))["actions"])
                latencies.append(time.monotonic() - before)
                if actions.shape != (50, 14) or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid action chunk {actions.shape}")
                predictions.append(actions)
            action = actions[step % 10]
            prior_state = np.asarray(raw["agent_pos"]).tolist()
            raw, reward, terminated, truncated, _ = env.step(action)
            max_reward = max(max_reward, float(reward))
            physics = env.unwrapped._env.physics
            contacts = [
                [
                    physics.model.id2name(int(contact.geom1), "geom"),
                    physics.model.id2name(int(contact.geom2), "geom"),
                ]
                for contact in physics.data.contact[: physics.data.ncon]
            ]
            rows.append(
                {
                    "step": step,
                    "sensor_state": prior_state,
                    "action": action.tolist(),
                    "evaluation_reward": float(reward),
                    "evaluation_qpos": env.unwrapped._env.physics.data.qpos.copy().tolist(),
                    "evaluation_contacts": contacts,
                }
            )
            if step % 10 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "reward": float(reward),
                            "inference_seconds": latencies[-1],
                        }
                    ),
                    flush=True,
                )
            if terminated or truncated:
                break
        video.append_data(raw["pixels"]["top"])
    env.close()
    np.savez_compressed(args.output / "predictions.npz", actions=np.stack(predictions))
    (args.output / "trace.json").write_text(json.dumps(rows))
    result = {
        "policy": "pi0_aloha_sim",
        "embodiment": "ALOHA",
        "seed": args.seed,
        "openpi_revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "task": "gym_aloha/AlohaTransferCube-v0",
        "status": "completed",
        "steps": len(rows),
        "max_reward": max_reward,
        "success": max_reward >= 4,
        "success_criterion": "Official gym-aloha reward4: cube contacts the left gripper finger and does not contact table. No sustained-hold or right-hand-release requirement.",
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "wall_seconds": time.monotonic() - started,
        "inference_seconds": latencies,
        "humanoid_control": False,
        "matched_tensorcode_comparison": False,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_versions": {
            name: version(name)
            for name in (
                "torch",
                "transformers",
                "jax",
                "flax",
                "gym-aloha",
                "gymnasium",
                "mujoco",
                "dm-control",
                "numpy",
            )
        },
    }
    if args.device.startswith("cuda"):
        device = torch.cuda.get_device_properties(args.device)
        result["gpu"] = {
            "name": device.name,
            "visible_multiprocessors": device.multi_processor_count,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
