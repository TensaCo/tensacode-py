"""Freeze supplemental data conditions before observing collection outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def generate(output, nominal):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("Refusing to overwrite frozen protocol")
    base = json.loads(nominal.read_text())
    rng = np.random.default_rng(2026101601)
    cases = []
    for split, count in (("train", 32), ("validation", 8)):
        for index in range(count):
            case_id = f"bc-{split}-{index:02d}"
            scenario = json.loads(json.dumps(base))
            scenario["id"] = case_id
            scenario["seed"] = 2026101700 + len(cases)
            for cell in scenario["cells"]:
                # Moderate iid scene variation, no held-out physical regimes.
                cell["pickup_delta_xyz"] = [
                    *rng.uniform(-0.025, 0.025, 2).round(6).tolist(),
                    0.0,
                ]
                cell["receiver_delta_xyz"] = [
                    *rng.uniform(-0.025, 0.025, 2).round(6).tolist(),
                    0.0,
                ]
                cell["payload_mass_kg"] = 0.08
            raw = json.dumps(scenario, indent=2) + "\n"
            (output / f"{case_id}.json").write_text(raw)
            cases.append(
                {
                    "id": case_id,
                    "split": split,
                    "scenario": scenario,
                    "scenario_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                }
            )
    protocol = {
        "schema": "tensorcode.warehouse.bc.collection.v1",
        "seed": 2026101601,
        "robots": 2,
        "seconds": 85,
        "cases": cases,
        "split_unit": "whole physical run",
        "training_runs": 32,
        "validation_runs": 8,
        "collector": "--bc-record",
        "retention": "retain all failures; never filter by final outcome",
        "final_conditions": "learned_policy/protocol.json (unchanged)",
        "freeze": "before collection or model training",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output, args.nominal)
