"""Train the frozen compact DDPM protocol; never score final conditions here."""

from __future__ import annotations

import argparse
import copy
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..learned_policy.contract import ACTION_NAMES, FEATURE_NAMES
from ..learned_policy.data import sha256
from .data import Windows, load_training_manifest, normalization
from .model import SCHEMA, Denoiser, Diffusion


@torch.inference_mode()
def evaluate(model, diffusion, loader, device, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0.0
    for batch in loader:
        x, y, mask = (value.to(device) for value in batch)
        objective = diffusion.loss(model, x, y, mask, generator)
        weight = float(mask.sum())
        total += float(objective) * weight
        count += weight
    return total / count


def train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite nonempty training output")
    if sha256(args.manifest) != args.manifest_sha256:
        raise ValueError("Immutable manifest SHA256 mismatch")
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    cfg = protocol["training"]
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    manifest, splits = load_training_manifest(args.manifest)
    statistics = normalization(splits["train"])
    architecture = {
        k: protocol["architecture"][k]
        for k in ("history", "horizon", "hidden", "blocks")
    }
    train_windows = Windows(
        splits["train"],
        statistics,
        architecture["history"],
        architecture["horizon"],
        cfg["train_window_stride"],
    )
    validation_windows = Windows(
        splits["validation"],
        statistics,
        architecture["history"],
        architecture["horizon"],
        cfg["validation_window_stride"],
    )
    train_loader = DataLoader(
        train_windows,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    val_loader = DataLoader(
        validation_windows, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )
    device = torch.device(args.device)
    model = Denoiser(**architecture).to(device)
    diffusion = Diffusion(protocol["diffusion"]["steps"], device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(Path(__file__).parent.glob("*.py")) + [protocol_path]
    shared = Path(__file__).parent.parent / "learned_policy"
    source_paths += [shared / name for name in ("contract.py", "data.py", "adapter.py")]
    ablation_protocol = shared / "ablation-protocol.json"
    if manifest.get("curation"):
        source_paths += [ablation_protocol, shared / "ablation.py"]
    with zipfile.ZipFile(
        args.output / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for source in source_paths:
            archive.write(source, str(source.relative_to(shared.parent)))
    binding = {
        "protocol_sha256": sha256(protocol_path),
        "manifest_sha256": args.manifest_sha256,
        "manifest": manifest,
        "final_data_opened": False,
        "data_curation": manifest.get("curation"),
        "ablation_protocol_sha256": sha256(ablation_protocol)
        if manifest.get("curation")
        else None,
    }
    (args.output / "binding.json").write_text(json.dumps(binding, indent=2))
    started = time.monotonic()
    history, best, best_state = [], float("inf"), None
    for epoch in range(cfg["epochs"]):
        model.train()
        total, count = 0.0, 0.0
        for batch in train_loader:
            x, y, mask = (value.to(device) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            objective = diffusion.loss(model, x, y, mask)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            weight = float(mask.sum())
            total += float(objective.detach()) * weight
            count += weight
        validation = evaluate(model, diffusion, val_loader, device, cfg["seed"] + 100)
        row = {
            "epoch": epoch + 1,
            "train_epsilon_mse": total / count,
            "validation_epsilon_mse": validation,
        }
        history.append(row)
        if validation < best:
            best, best_state = validation, copy.deepcopy(model.state_dict())
        print(json.dumps(row), flush=True)
    torch.save({k: v.cpu() for k, v in best_state.items()}, args.output / "weights.pt")
    # Optional diagnostic AFTER selection, using the independently declared
    # teacher outcome. It never filters the checkpoint-selection validation set.
    outcomes = manifest.get("evaluation_only_teacher_outcomes", {})
    successful_validation = [
        episode
        for episode, entry in zip(splits["validation"], manifest["validation"])
        if outcomes.get(
            f"{entry['physical_run_id']}/{entry.get('robot_id', '')}", {}
        ).get("strict_current_settle_success", False)
    ]
    successful_validation_report = None
    if successful_validation:
        model.load_state_dict(best_state)
        diagnostic = Windows(
            successful_validation,
            statistics,
            architecture["history"],
            architecture["horizon"],
            cfg["validation_window_stride"],
        )
        diagnostic_loader = DataLoader(
            diagnostic, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
        )
        successful_validation_report = {
            "episodes": len(successful_validation),
            "epsilon_mse": evaluate(
                model, diffusion, diagnostic_loader, device, cfg["seed"] + 100
            ),
            "used_for_selection": False,
        }
    mean, scale, action_mean, action_scale = statistics
    config = {
        "schema": SCHEMA,
        "architecture": architecture,
        "diffusion_steps": diffusion.steps,
        "execute": protocol["architecture"]["execute"],
        "seed": cfg["seed"],
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "input_mean": mean.tolist(),
        "input_scale": scale.tolist(),
        "action_mean": action_mean.tolist(),
        "action_scale": action_scale.tolist(),
        "protocol_sha256": binding["protocol_sha256"],
        "manifest_sha256": args.manifest_sha256,
    }
    (args.output / "policy.json").write_text(json.dumps(config, indent=2))
    report = {
        **binding,
        "history": history,
        "selected_epoch": min(history, key=lambda r: r["validation_epsilon_mse"])[
            "epoch"
        ],
        "elapsed_seconds": time.monotonic() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
        "weights_sha256": sha256(args.output / "weights.pt"),
        "config_sha256": sha256(args.output / "policy.json"),
        "source_archive_sha256": sha256(args.output / "source.zip"),
        "protocol": protocol,
        "test_evaluated": False,
        "closed_loop_evaluated": False,
        "validation_successful_teacher_subset_not_used_for_selection": successful_validation_report,
        "runtime": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "threads": args.threads,
        },
        "metric_scope": "teacher-window denoising objective; not closed-loop task success",
    }
    (args.output / "training.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    train(parser.parse_args())
