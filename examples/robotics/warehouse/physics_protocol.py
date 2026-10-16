"""Declare the bounded v4 robustness cases before any outcome is inspected."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from physics_scenario import normalize_scenario


def cases():
    result = []

    def add(
        name,
        *,
        cell=None,
        sensors=None,
        compensation=True,
        split="declared_check",
        pair=None,
    ):
        result.append(
            {
                "id": name,
                "split": split,
                "pair": pair,
                "scenario": normalize_scenario(
                    {
                        "id": name,
                        "seed": 20261016,
                        "cells": [cell or {}, cell or {}],
                        "sensors": sensors or {},
                        "load_compensation": compensation,
                    },
                    2,
                ),
            }
        )

    for mass in (0.08, 0.2, 0.3):
        label = f"payload-{round(mass * 1000)}g"
        for compensation in (True, False):
            add(
                label + ("-compensated" if compensation else "-uncompensated"),
                cell={"payload_mass_kg": mass},
                compensation=compensation,
                split="development" if mass == 0.08 else "declared_check",
                pair=label,
            )
    add(
        "teacher-pickup-x10mm",
        cell={"pickup_delta_xyz": [0.01, 0, 0]},
        split="bc_train",
    )
    add(
        "teacher-pickup-y10mm",
        cell={"pickup_delta_xyz": [0, 0.01, 0]},
        split="bc_train",
    )
    add(
        "teacher-validation-offset",
        cell={"pickup_delta_xyz": [0.005, -0.005, 0]},
        split="bc_validation",
    )
    add("pickup-shift", cell={"pickup_delta_xyz": [0.02, 0.015, 0]})
    add("pickup-raised", cell={"pickup_delta_xyz": [-0.01, -0.01, 0.04]})
    add("receiver-shift", cell={"receiver_delta_xyz": [-0.02, 0.035, 0]})
    add("receiver-raised", cell={"receiver_delta_xyz": [0, 0, 0.04]})
    add("depth-noise-2mm", sensors={"depth_noise_std_m": 0.002})
    add("depth-noise-8mm", sensors={"depth_noise_std_m": 0.008})
    add("calibration-bias-5mm", sensors={"calibration_bias_xyz": [0.005, -0.005, 0]})
    add("calibration-bias-20mm", sensors={"calibration_bias_xyz": [0.02, -0.02, 0]})
    add("camera-outage-brief", sensors={"camera_outage": [15, 16]})
    add("camera-outage-long", sensors={"camera_outage": [15, 80]})
    add(
        "camera-view-shift",
        sensors={"camera_azimuth_delta_deg": 8, "camera_elevation_delta_deg": 5},
    )
    add("receiver-unreachable", cell={"receiver_delta_xyz": [0.2, -0.15, 0]})
    return result


def write_protocol(output):
    output.mkdir(parents=True, exist_ok=True)
    entries = cases()
    for entry in entries:
        raw = json.dumps(entry["scenario"], indent=2) + "\n"
        (output / (entry["id"] + ".json")).write_text(raw)
        entry["scenario_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    protocol = {
        "schema": "tensorcode.warehouse.robustness.v1",
        "robots": 2,
        "seconds": 85,
        "cases": entries,
        "scope": "Known geometry/color workcells; fixed authored controller, real environmental and pixel interventions; all successes and failures retained",
        "comparison": "Only compensated/uncompensated cases of equal mass isolate the load-feedforward switch. Other cases are sensitivity checks, not baseline superiority claims.",
        "bc_splits": {
            "train": [
                "payload-80g-compensated",
                "teacher-pickup-x10mm",
                "teacher-pickup-y10mm",
            ],
            "validation": ["teacher-validation-offset"],
            "final": [
                "payload-200g-compensated",
                "payload-300g-compensated",
                "receiver-raised",
                "camera-outage-brief",
                "camera-outage-long",
            ],
        },
        "split_policy": "Cases declared before execution. Nominal80g pair is development; other cases are new fixed-controller checks. Any tuning invalidates held-out status for observed cases.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "cases": len(write_protocol(args.output)["cases"]),
                "output": str(args.output),
            }
        )
    )
