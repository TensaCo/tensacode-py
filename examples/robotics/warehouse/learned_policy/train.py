"""Train recurrent BC on declared teacher episodes; report offline error honestly."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .contract import ACTION_NAMES, FEATURE_NAMES, SCHEMA, input_statistics
from .data import load_manifest, sha256
from .model import SequencePolicy


def loss(prediction, target, mask=None):
    continuous = F.mse_loss(
        prediction[..., :-2], target[..., :-2], reduction="none"
    ).mean(-1)
    binary = F.binary_cross_entropy_with_logits(
        prediction[..., -2:], target[..., -2:], reduction="none"
    ).mean(-1)
    per_frame = continuous + binary
    if mask is None:
        return per_frame.mean()
    return (per_frame * mask).sum() / mask.sum().clamp_min(1)


def batch_episodes(episodes, mean, scale, ymean, yscale, device):
    length = max(len(row[0]) for row in episodes)
    tx = np.zeros((len(episodes), length, len(FEATURE_NAMES)), dtype=np.float32)
    ty = np.zeros((len(episodes), length, len(ACTION_NAMES)), dtype=np.float32)
    mask = np.zeros((len(episodes), length), dtype=np.float32)
    for i, (ex, ey, *_) in enumerate(episodes):
        tx[i, : len(ex)] = (ex - mean) / scale
        ty[i, : len(ey)] = (ey - ymean) / yscale
        mask[i, : len(ex)] = 1
    return tuple(torch.as_tensor(value, device=device) for value in (tx, ty, mask))


@torch.inference_mode()
def evaluate(model, episodes, mean, scale, ymean, yscale, device):
    totals, errors, flags = [], [], []
    for x, y, *_ in episodes:
        tx = torch.as_tensor((x - mean) / scale, device=device)[None]
        ty = torch.as_tensor((y - ymean) / yscale, device=device)[None]
        p, _ = model(tx)
        totals.append(float(loss(p, ty)))
        raw = p[0].cpu().numpy() * yscale + ymean
        errors.append(np.mean(np.abs(raw[:, :-2] - y[:, :-2]), axis=0))
        flags.append(np.mean((p[0, :, -2:].cpu().numpy() >= 0) == y[:, -2:], axis=0))
    return {
        "normalized_loss": float(np.mean(totals)),
        "continuous_mae": np.mean(errors, axis=0).tolist(),
        "flag_accuracy": np.mean(flags, axis=0).tolist(),
        "episodes": len(episodes),
        "scope": "teacher_trajectory_prediction_not_closed_loop_success",
    }


def train(args):
    if (args.output / "weights.pt").exists():
        raise ValueError("Refusing to overwrite an existing trained policy")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest, splits = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(Path(__file__).parent.glob("*.py")) + sorted(
        Path(__file__).parent.glob("*.json")
    )
    source_hashes = {p.name: sha256(p) for p in source_paths}
    with zipfile.ZipFile(
        args.output / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for source in source_paths:
            archive.write(source, source.name)
    # All normalization fitted on train only. Test bytes are verified but neither
    # scored nor used to choose a checkpoint in this training command.
    x = np.concatenate([row[0] for row in splits["train"]])
    y = np.concatenate([row[1] for row in splits["train"]])
    mean, scale = input_statistics(x)
    ymean, yscale = y.mean(0), np.maximum(y.std(0), 0.01)
    ymean[-2:], yscale[-2:] = 0, 1
    device = torch.device(args.device)
    model = SequencePolicy(args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history, best, best_state = [], float("inf"), None
    started = time.monotonic()
    for epoch in range(args.epochs):
        order = list(range(len(splits["train"])))
        random.shuffle(order)
        model.train()
        training_losses = []
        for start in range(0, len(order), args.batch_size):
            episodes = [
                splits["train"][i] for i in order[start : start + args.batch_size]
            ]
            tx, ty, mask = batch_episodes(episodes, mean, scale, ymean, yscale, device)
            state = None
            for begin in range(0, tx.shape[1], args.chunk):
                optimizer.zero_grad(set_to_none=True)
                prediction, state = model(tx[:, begin : begin + args.chunk], state)
                objective = loss(
                    prediction,
                    ty[:, begin : begin + args.chunk],
                    mask[:, begin : begin + args.chunk],
                )
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                state = state.detach()
                training_losses.append(float(objective.detach()))
        model.eval()
        validation = evaluate(
            model, splits["validation"], mean, scale, ymean, yscale, device
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(training_losses)),
            "validation": validation,
        }
        history.append(row)
        if validation["normalized_loss"] < best:
            best, best_state = (
                validation["normalized_loss"],
                copy.deepcopy(model.state_dict()),
            )
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps(row), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    successful_validation = [
        episode
        for episode in splits["validation"]
        if manifest.get("evaluation_only_teacher_outcomes", {})
        .get(f"{episode[3]['physical_run_id']}/{episode[3].get('robot_id', '')}", {})
        .get("strict_current_settle_success")
    ]
    successful_validation_report = (
        evaluate(model, successful_validation, mean, scale, ymean, yscale, device)
        if successful_validation
        else None
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in best_state.items()}, args.output / "weights.pt")
    config = {
        "schema": SCHEMA,
        "hidden": args.hidden,
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "input_mean": mean.tolist(),
        "input_scale": scale.tolist(),
        "action_mean": ymean.tolist(),
        "action_scale": yscale.tolist(),
        "architecture": "linear-tanh-GRU-linear",
        "shared_controller": "frozen Unitree walker and calibrated servo",
        "teacher_phase_input": False,
        "ground_truth_pose_input": False,
    }
    (args.output / "policy.json").write_text(json.dumps(config, indent=2))
    report = {
        "schema": SCHEMA,
        "manifest": manifest,
        "source_hashes": source_hashes,
        "source_archive_sha256": sha256(args.output / "source.zip"),
        "manifest_sha256": sha256(args.manifest),
        "weights_sha256": sha256(args.output / "weights.pt"),
        "config_sha256": sha256(args.output / "policy.json"),
        "hyperparameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "elapsed_seconds": time.monotonic() - started,
        "selected_epoch": min(
            history, key=lambda row: row["validation"]["normalized_loss"]
        )["epoch"],
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
        },
        "history": history,
        "selection": "minimum declared validation normalized loss",
        "validation_successful_teacher_subset_not_used_for_selection": successful_validation_report,
        "test_evaluated": False,
        "closed_loop_evaluated": False,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (args.output / "training.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
