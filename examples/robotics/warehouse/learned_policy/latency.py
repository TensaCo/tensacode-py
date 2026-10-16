"""Measured single-thread CPU policy latency on validation observations only."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_episode, sha256


def summary(values):
    return {
        "samples": len(values),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "mean_ms": float(np.mean(values)),
    }


def measure(manifest_path, directory, kind, steps=512, warmup=64):
    if steps < 200 or warmup < 1:
        raise ValueError("Use at least 200 measured steps and a nonempty warmup")
    torch.set_num_threads(1)
    if kind == "bc":
        from .model import PolicyRuntime
    elif kind == "diffusion":
        from ..diffusion_policy.model import PolicyRuntime
    else:
        raise ValueError("Unknown policy kind")
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["validation"][0]
    episode_path = (manifest_path.parent / entry["path"]).resolve()
    if sha256(episode_path) != entry["sha256"]:
        raise ValueError("Validation episode hash differs from manifest")
    _, _, episode = load_episode(episode_path)
    if len(episode["steps"]) < steps + warmup:
        raise ValueError("Validation episode is too short for declared measurement")
    runtime = PolicyRuntime(directory)
    durations, planning, cached = [], [], []
    for index, row in enumerate(episode["steps"][: warmup + steps]):
        is_plan = kind == "diffusion" and not runtime.pending
        before = time.perf_counter_ns()
        runtime.step(row["observation"])
        elapsed_ms = (time.perf_counter_ns() - before) / 1_000_000
        if index >= warmup:
            durations.append(elapsed_ms)
            if kind == "diffusion":
                (planning if is_plan else cached).append(elapsed_ms)
    return {
        "policy_kind": kind,
        "device": "CPU",
        "torch_threads": 1,
        "batch_size": 1,
        "warmup_steps": warmup,
        "all_steps": summary(durations),
        "planning_steps": summary(planning) if planning else None,
        "cached_action_steps": summary(cached) if cached else None,
        "validation_physical_run": entry["physical_run_id"],
        "validation_robot": entry["robot_id"],
        "manifest_sha256": sha256(manifest_path),
        "episode_sha256": entry["sha256"],
        "weights_sha256": sha256(directory / "weights.pt"),
        "config_sha256": sha256(directory / "policy.json"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "load_average": list(os.getloadavg()),
        "scope": "measured feature extraction, policy inference and command decoding; excludes camera rendering, sensing, frozen walking policy, servos and simulation",
        "limitation": "observed CPU timing under recorded host load; not a hard real-time or worst-case execution-time guarantee",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--kind", choices=("bc", "diffusion"), required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            measure(args.manifest, args.policy, args.kind, args.steps, args.warmup),
            indent=2,
        )
        + "\n"
    )
