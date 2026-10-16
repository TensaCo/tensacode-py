"""Bind the frozen collection plan to complete, unfiltered physical episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ..validate_v4 import validate
from .data import load_episode, sha256


def build(protocol, supplemental_protocol, roots, output):
    fixed = json.loads(protocol.read_text())
    supplemental = json.loads(supplemental_protocol.read_text())
    if (
        sha256(supplemental_protocol)
        != fixed["supplemental_collection"]["protocol_sha256"]
    ):
        raise ValueError("Supplemental protocol differs from the frozen declaration")
    expected = [
        (split, case["case_id"], case["scenario"])
        for split in ("train", "validation")
        for case in fixed[split]
    ]
    expected += [
        (case["split"], case["id"], case["scenario"]) for case in supplemental["cases"]
    ]
    result = {
        "train": [],
        "validation": [],
        "test": [],
        "protocol_sha256": sha256(protocol),
        "supplemental_protocol_sha256": sha256(supplemental_protocol),
        "selection": "all frozen collection episodes, including failures",
        "evaluation_only_teacher_outcomes": {},
    }
    teacher_sources = None
    action_paths = [
        "warehouse/physics.py",
        "warehouse/physics_scene.py",
        "warehouse/physics_sensors.py",
        "warehouse/physics_recording.py",
        "warehouse/physics_recovery.py",
        "warehouse/physics_scenario.py",
        "warehouse/scenery.py",
        "warehouse/coordination.py",
        "warehouse/language.py",
        "warehouse/representation.py",
        "transfer/handling.py",
        "transfer/planning.py",
        "transfer/sensors.py",
        "transfer/box_qp.py",
        "g1/model.py",
    ]
    for split, case_id, scenario in expected:
        matches = [
            root / case_id for root in roots if (root / case_id / "probe.json").exists()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one completed run for {case_id}; got {len(matches)}"
            )
        directory = matches[0]
        probe = json.loads((directory / "probe.json").read_text())
        if (
            not probe.get("grasp_recovery")
            or not probe.get("full_horizon")
            or probe.get("requested_duration_s") != 85
        ):
            raise ValueError(
                "Corrected expert collection requires recovery and the full 85-second protocol"
            )
        source_hashes = {name: probe["source_sha256"][name] for name in action_paths}
        if teacher_sources is None:
            teacher_sources = source_hashes
        elif teacher_sources != source_hashes:
            raise ValueError(
                "Teacher action source changed between physical runs; declare a new collection revision"
            )
        validation_path = directory / "validation-v4.json"
        validation = (
            json.loads(validation_path.read_text())
            if validation_path.exists()
            else validate(directory)
        )
        if not validation.get("validated") or validation["records_sha256"] != sha256(
            directory / "records.json"
        ):
            raise ValueError(
                f"Stale or invalid independent evidence audit for {case_id}"
            )
        paths = sorted(directory.glob("r*/bc-episode.json.gz"))
        if len(paths) != supplemental["robots"]:
            raise ValueError(f"Missing robot episodes for {case_id}")
        for path in paths:
            x, _, document = load_episode(path)
            if document["scenario"] != scenario:
                raise ValueError(
                    f"Recorded scenario differs from frozen case {case_id}"
                )
            result[split].append(
                {
                    "path": os.path.relpath(path, output.parent),
                    "sha256": sha256(path),
                    "physical_run_id": case_id,
                    "robot_id": document["robot_id"],
                    "frames": len(x),
                }
            )
            # Reporting only; never used to include/exclude training data.
            result["evaluation_only_teacher_outcomes"][
                f"{case_id}/{document['robot_id']}"
            ] = {
                **document["evaluation"],
                "strict_current_settle_success": validation["robots"][
                    document["robot_id"]
                ]["strict_current_settle_success"],
            }
    result["teacher_action_source_sha256"] = teacher_sources
    result["teacher_action_fingerprint"] = hashlib.sha256(
        json.dumps(teacher_sources, sort_keys=True).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("protocol.json")
    )
    parser.add_argument("--supplemental-protocol", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.protocol, args.supplemental_protocol, args.runs, args.output)
