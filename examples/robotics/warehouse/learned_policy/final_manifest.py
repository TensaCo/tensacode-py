"""Freeze every candidate and condition before any final comparison rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import sha256


def freeze(policies, cases, source_archive, output):
    protocol_path = Path(__file__).with_name("final-comparison-protocol.json")
    protocol = json.loads(protocol_path.read_text())
    expected = {
        row["id"]: row for row in protocol["policies"] if row["kind"] != "authored"
    }
    if set(policies) != set(expected):
        raise ValueError(
            "Bind every preregistered candidate, with no winner-only selection"
        )
    bound = []
    for name, specification in expected.items():
        directory = Path(policies[name]).resolve()
        names = ["weights.pt", "policy.json"]
        if specification["kind"] == "geometry":
            names.append("kinematics.json")
        files = {
            file: {"path": str(directory / file), "sha256": sha256(directory / file)}
            for file in names
        }
        training = json.loads((directory / "training.json").read_text())
        if (
            training["weights_sha256"] != files["weights.pt"]["sha256"]
            or training["config_sha256"] != files["policy.json"]["sha256"]
        ):
            raise ValueError(
                f"Candidate no longer matches selected training artifact: {name}"
            )
        bound.append(
            {
                "id": name,
                "kind": specification["kind"],
                "directory": str(directory),
                "files": files,
                "training_report_sha256": sha256(directory / "training.json"),
            }
        )
    conditions = {
        name: {
            "path": str((Path(cases) / f"{name}.json").resolve()),
            "sha256": sha256(Path(cases) / f"{name}.json"),
        }
        for name in protocol["cases"]
    }
    manifest = {
        "protocol_sha256": sha256(protocol_path),
        "protocol": protocol,
        "candidates": bound,
        "conditions": conditions,
        "source_archive": {
            "path": str(Path(source_archive).resolve()),
            "sha256": sha256(source_archive),
        },
        "result_selection": "none; all candidates and physical cases preregistered",
        "expected_physical_runs": 40,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        help="ID=POLICY_DIRECTORY, all seven learned candidates required",
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policies = dict(value.split("=", 1) for value in args.policy)
    freeze(policies, args.cases, args.source_archive, args.output)
