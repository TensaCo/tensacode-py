"""Conditional assisted aggregation fit; never changes the frozen final candidates."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from .dagger2_sampling import balanced_episode_batches
from .data import sha256
from .geometry_data import load_prepared
from .geometry_model import GeometrySequencePolicy
from .geometry_train import batch_episodes, evaluate, loss

PROTOCOL_SHA = "fd7512e59dd622d11d40e87529037b2e453f4cbdf8692468866c83ecc72904d9"


def validate_sources(protocol, original, corrective):
    """Whole-run TRAIN membership and unchanged full validation are mandatory."""
    if (
        len(original["train"]) != 70
        or len(original["validation"]) != 18
        or original["test"]
    ):
        raise ValueError("Original 70/18/0 episode split required")
    if corrective["validation"] or corrective["test"] or len(corrective["train"]) != 12:
        raise ValueError("Exactly twelve corrective TRAIN episodes required")
    expected = {
        (case, f"r{i}")
        for case in protocol["collection"]["case_sha256"]
        for i in range(2)
    }
    actual = [(e["physical_run_id"], e["robot_id"]) for e in corrective["train"]]
    if len(set(actual)) != 12 or set(actual) != expected:
        raise ValueError(
            "Corrective episode membership differs from six frozen TRAIN cases"
        )
    if corrective["dagger2"]["protocol_sha256"] != PROTOCOL_SHA:
        raise ValueError("Corrective manifest protocol mismatch")
    provenance = corrective["dagger2"]
    for key in ("qualification_report_sha256", "collection_source_archive_sha256"):
        digest = provenance.get(key, "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Missing corrective source or qualification binding")
    if provenance.get("case_sha256") != protocol["collection"]["case_sha256"]:
        raise ValueError("Corrective case provenance differs from protocol")
    if (
        provenance.get("collection_weights_sha256")
        != protocol["fit"]["initial_weights_sha256"]
    ):
        raise ValueError("Corrective collection policy differs from frozen learner")
    if not corrective["dagger2"]["qualification_gate_passed"]:
        raise ValueError("Qualification gate has not passed")


def train(args):
    protocol_path = Path(__file__).with_name("dagger2-finetune-proposal.json")
    if sha256(protocol_path) != PROTOCOL_SHA:
        raise ValueError("Conditional protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if sha256(args.manifest) != protocol["dataset"]["original_manifest_sha256"]:
        raise ValueError("Original manifest differs from frozen data")
    if sha256(args.initial / "weights.pt") != protocol["fit"]["initial_weights_sha256"]:
        raise ValueError("Initial geometry checkpoint changed")
    qualification = json.loads(
        Path(__file__).with_name("dagger2-protocol.json").read_text()
    )
    for name, key in [
        ("policy.json", "initial_policy_sha256"),
        ("kinematics.json", "kinematics_sha256"),
    ]:
        if sha256(args.initial / name) != qualification[key]:
            raise ValueError("Initial geometry configuration changed")
    if args.output.exists():
        raise ValueError("Use a new output directory; never overwrite an experiment")
    original, splits = load_prepared(
        args.manifest, args.initial / "kinematics.json", args.prepared
    )
    corrective, new = load_prepared(
        args.corrective_manifest,
        args.initial / "kinematics.json",
        args.corrective_prepared,
    )
    validate_sources(protocol, original, corrective)
    # Revalidate raw bytes too; prepared caches cannot replace evidence binding.
    for manifest_path, manifest in [
        (args.manifest, original),
        (args.corrective_manifest, corrective),
    ]:
        for split in ("train", "validation"):
            for entry in manifest[split]:
                if sha256(manifest_path.parent / entry["path"]) != entry["sha256"]:
                    raise ValueError("Raw episode changed after preparation")
    fit = protocol["fit"]
    torch.set_num_threads(args.threads)
    random.seed(fit["seed"])
    np.random.seed(fit["seed"])
    torch.manual_seed(fit["seed"])
    config = json.loads((args.initial / "policy.json").read_text())
    mean, scale, ym, ys = [
        np.asarray(config[key], dtype=np.float32)
        for key in ("input_mean", "input_scale", "action_mean", "action_scale")
    ]
    device = torch.device(args.device)
    model = GeometrySequencePolicy(128).to(device)
    model.load_state_dict(
        torch.load(args.initial / "weights.pt", map_location=device, weights_only=True)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=fit["learning_rate"], weight_decay=0.01
    )
    best, best_state, history = float("inf"), None, []
    started = time.monotonic()
    sources = {"original": splits["train"], "corrective": new["train"]}
    for epoch in range(fit["epochs"]):
        model.train()
        objectives = []
        for batch in balanced_episode_batches(70, 12, seed=fit["seed"], epoch=epoch):
            episodes = [sources[source][index] for source, index in batch]
            x, y, mask = batch_episodes(episodes, mean, scale, ym, ys, device)
            state = None
            for begin in range(0, x.shape[1], fit["chunk"]):
                end = begin + fit["chunk"]
                optimizer.zero_grad(set_to_none=True)
                prediction, state = model(x[:, begin:end], state)
                objective = loss(prediction, y[:, begin:end], mask[:, begin:end])
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                state = state.detach()
                objectives.append(float(objective.detach()))
        model.eval()
        validation = evaluate(model, splits["validation"], mean, scale, ym, ys, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(objectives)),
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
    args.output.mkdir(parents=True)
    torch.save({k: v.cpu() for k, v in best_state.items()}, args.output / "weights.pt")
    for name in ("policy.json", "kinematics.json"):
        shutil.copyfile(args.initial / name, args.output / name)
    source_paths = sorted(Path(__file__).parent.glob("*.py")) + sorted(
        Path(__file__).parent.glob("*.json")
    )
    with zipfile.ZipFile(
        args.output / "source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in source_paths:
            archive.write(path, path.name)
    report = {
        "experiment": "assisted dataset aggregation; separate development experiment",
        "protocol_sha256": PROTOCOL_SHA,
        "initial_weights_sha256": sha256(args.initial / "weights.pt"),
        "original_manifest": original,
        "corrective_manifest": corrective,
        "original_manifest_sha256": sha256(args.manifest),
        "corrective_manifest_sha256": sha256(args.corrective_manifest),
        "original_prepared_index_sha256": sha256(args.prepared / "index.json"),
        "corrective_prepared_index_sha256": sha256(
            args.corrective_prepared / "index.json"
        ),
        "source_hashes": {p.name: sha256(p) for p in source_paths},
        "source_archive_sha256": sha256(args.output / "source.zip"),
        "weights_sha256": sha256(args.output / "weights.pt"),
        "config_sha256": sha256(args.output / "policy.json"),
        "kinematics_sha256": sha256(args.output / "kinematics.json"),
        "history": history,
        "selected_epoch": min(
            history, key=lambda row: row["validation"]["normalized_loss"]
        )["epoch"],
        "fit": fit,
        "elapsed_seconds": time.monotonic() - started,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "closed_loop_evaluated": False,
        "final_comparison_candidate": False,
    }
    (args.output / "training.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "prepared",
        "corrective-manifest",
        "corrective-prepared",
        "initial",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    train(parser.parse_args())
