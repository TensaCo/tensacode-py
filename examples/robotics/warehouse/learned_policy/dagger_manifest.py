"""Bind TRAIN-only shadow labels; preserve the complete original validation set."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .data import load_episode, load_manifest, sha256


def aggregate(original, round_roots, output):
    original, output = Path(original), Path(output)
    manifest = json.loads(original.read_text())
    protocol_path = Path(__file__).with_name("dagger-protocol.json")
    protocol = json.loads(protocol_path.read_text())
    if sha256(original) != protocol["original_manifest_sha256"]:
        raise ValueError("Original dataset differs from preregistration")
    if not 1 <= len(round_roots) <= 3:
        raise ValueError("Expected one to three complete rounds")
    result = copy.deepcopy(manifest)
    for split in ("train", "validation", "test"):
        for entry in result[split]:
            entry["path"] = str((original.parent / entry["path"]).resolve())
    expected_scenarios = {}
    for entry in manifest["train"]:
        case = entry["physical_run_id"]
        if case in protocol["training_conditions"] and case not in expected_scenarios:
            _, _, document = load_episode(original.parent / entry["path"])
            expected_scenarios[case] = document["scenario"]
    source_fingerprints = []
    for round_number, root in enumerate(round_roots, 1):
        round_hashes = set()
        for case in protocol["training_conditions"]:
            directory = Path(root) / case
            probe = json.loads((directory / "probe.json").read_text())
            if probe["scenario"] != expected_scenarios[case]:
                raise ValueError(
                    "Collection condition differs from original TRAIN scenario"
                )
            if not probe["full_horizon"] or probe["requested_duration_s"] != 85:
                raise ValueError("Collection horizon differs from protocol")
            if probe["policy"] != "learned_dagger" or not probe["grasp_recovery"]:
                raise ValueError("Not the preregistered shadow expert collector")
            round_hashes.add(json.dumps(probe["source_sha256"], sort_keys=True))
            for robot in ("r0", "r1"):
                path = directory / robot / "dagger-episode.json.gz"
                x, _, document = load_episode(path)
                if (
                    document["label_provenance"]
                    != "sensor_only_shadow_expert_on_actual_visited_state"
                ):
                    raise ValueError("Unidentified supervision provenance")
                if document["dagger"]["seed"] != protocol["seed"]:
                    raise ValueError("Assistance seed differs from preregistration")
                if (
                    document["learner_weights_sha256"]
                    != probe["learned_policy_sha256"]["weights.pt"]
                ):
                    raise ValueError("Collector learner weights mismatch")
                if (
                    round_number == 1
                    and document["learner_weights_sha256"]
                    != protocol["initial_policy_weights_sha256"]
                ):
                    raise ValueError("Initial learner differs from preregistration")
                if document["dagger"]["round"] != round_number:
                    raise ValueError("DAgger round mismatch")
                if (
                    document["dagger"]["expert_probability"]
                    != protocol["expert_probability_by_round"][round_number - 1]
                ):
                    raise ValueError("Unexpected teacher assistance")
                if document["scenario"] != probe["scenario"]:
                    raise ValueError("Scenario provenance mismatch")
                if any(
                    row["executed_source"] not in ("learner", "expert")
                    for row in document["steps"]
                ):
                    raise ValueError("Missing executed action attribution")
                if round_number == 3 and any(
                    row["executed_source"] != "learner" for row in document["steps"]
                ):
                    raise ValueError("Final collection round must be pure learner")
                result["train"].append(
                    {
                        "path": str(path.resolve()),
                        "sha256": sha256(path),
                        "physical_run_id": f"dagger-round{round_number}-{case}",
                        "robot_id": robot,
                        "frames": len(x),
                        "condition_family": case,
                        "label_kind": "shadow_expert",
                    }
                )
        if len(round_hashes) != 1:
            raise ValueError("Source changed within a collection round")
        source_fingerprints.append(json.loads(next(iter(round_hashes))))
    result["dagger"] = {
        "protocol_sha256": sha256(protocol_path),
        "original_manifest_sha256": sha256(original),
        "rounds": len(round_roots),
        "source_fingerprints": source_fingerprints,
        "failed_runs_retained": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    load_manifest(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--round-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.original, args.round_roots, args.output)
