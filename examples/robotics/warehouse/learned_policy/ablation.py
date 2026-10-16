"""Explicit success-only TRAIN curation; keep original validation and failures."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from .data import sha256


def curate(manifest_path, output):
    original = json.loads(manifest_path.read_text())
    result = copy.deepcopy(original)
    result["train"] = []
    excluded = []
    for entry in original["train"]:
        key = f"{entry['physical_run_id']}/{entry['robot_id']}"
        outcome = original["evaluation_only_teacher_outcomes"][key]
        if outcome["strict_current_settle_success"]:
            result["train"].append(copy.deepcopy(entry))
        else:
            excluded.append(
                {
                    **entry,
                    "reason": "teacher failed independent strict current settling criterion",
                    "outcome": outcome,
                }
            )
    if not result["train"]:
        raise ValueError("No successful teacher trajectories in TRAIN")
    # Keep all episodes in their original physical-run partitions. Only TRAIN
    # is curated; full validation remains the sole checkpoint-selection metric.
    for split in ("train", "validation", "test"):
        for entry in result[split]:
            entry["path"] = os.path.relpath(
                (manifest_path.parent / entry["path"]).resolve(), output.parent
            )
    result["curation"] = {
        "name": "strict-success-only-train",
        "primary_manifest_sha256": sha256(manifest_path),
        "ablation_protocol_sha256": sha256(
            Path(__file__).with_name("ablation-protocol.json")
        ),
        "validation_unchanged": True,
        "selection": "minimum loss on original full validation split",
        "excluded_train_episodes": excluded,
        "train_episodes_before": len(original["train"]),
        "train_episodes_after": len(result["train"]),
        "ground_truth_usage": "offline demonstration curation only; no evaluator field enters policy observations",
    }
    result["selection"] = (
        "preregistered successful TRAIN demonstration ablation; original validation unchanged"
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curate(args.manifest, args.output)
