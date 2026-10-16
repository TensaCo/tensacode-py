"""Audit every declared policy/case cell, including missing and invalid evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .learned_policy.data import sha256
from .learned_policy.slew_audit import audit as audit_slew
from .validate_v4 import require


def check_contract(probe, policy, protocol, candidate, scenario, sources):
    """Check execution against the predeclared contract, not just other runs."""
    expected_kind = (
        "authored" if policy["kind"] == "authored" else "learned_" + policy["kind"]
    )
    require(probe["policy"] == expected_kind, "Wrong policy family")
    expected_files = (
        {}
        if candidate is None
        else {name: item["sha256"] for name, item in candidate["files"].items()}
    )
    require(probe["learned_policy_sha256"] == expected_files, "Wrong frozen candidate")
    require(probe["scenario"] == scenario, "Wrong frozen scenario")
    require(probe["robots"] == protocol["robots_per_run"], "Wrong robot count")
    require(probe["full_horizon"] is True, "Full horizon not enabled")
    require(
        probe["requested_duration_s"] == protocol["physical_seconds"], "Wrong horizon"
    )
    require(probe["offscreen_samples"] == 4, "Wrong camera sampling")
    require(probe["grasp_recovery"] is True, "Common recovery configuration differs")
    require(
        probe["arm_slew_rad"] == protocol["arm_slew_rad_per_20ms"],
        "Wrong actuator bound",
    )
    require(bool(probe["source_sha256"]), "Missing recorded source bindings")
    for name, digest in probe["source_sha256"].items():
        require(
            sources.get("examples/robotics/" + name) == digest,
            "Execution source differs: " + name,
        )


def audit(manifest_path, runs, source_archive=None, cases=None):
    manifest_path, runs = Path(manifest_path), Path(runs)
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest["protocol"]
    candidates = {row["id"]: row for row in manifest["candidates"]}
    require(
        set(candidates)
        == {p["id"] for p in protocol["policies"] if p["kind"] != "authored"},
        "Candidate registry omits a declared family",
    )
    archive = Path(source_archive or manifest["source_archive"]["path"])
    require(
        sha256(archive) == manifest["source_archive"]["sha256"],
        "Common source archive changed",
    )
    with zipfile.ZipFile(archive) as source:
        sources = {
            name: hashlib.sha256(source.read(name)).hexdigest()
            for name in source.namelist()
        }
        protocol_bytes = source.read(
            "examples/robotics/warehouse/learned_policy/final-comparison-protocol.json"
        )
    require(
        hashlib.sha256(protocol_bytes).hexdigest() == manifest["protocol_sha256"],
        "Frozen protocol hash mismatch",
    )
    require(
        json.loads(protocol_bytes) == protocol,
        "Embedded protocol differs from frozen source",
    )
    require(
        len({p["id"] for p in protocol["policies"]}) == len(protocol["policies"])
        and len(set(protocol["cases"])) == len(protocol["cases"])
        and set(manifest["conditions"]) == set(protocol["cases"])
        and len(protocol["policies"]) * len(protocol["cases"])
        == manifest["expected_physical_runs"]
        > 0,
        "Invalid declared evaluation matrix",
    )
    scenarios = {}
    for name in protocol["cases"]:
        item = manifest["conditions"][name]
        path = Path(cases) / (name + ".json") if cases else Path(item["path"])
        require(sha256(path) == item["sha256"], "Frozen case changed: " + name)
        scenarios[name] = json.loads(path.read_text())
    rows, environments, initial_states = [], {}, {}
    for policy in protocol["policies"]:
        for case in protocol["cases"]:
            directory = runs / policy["id"] / case
            row = {"policy": policy["id"], "case": case, "path": str(directory)}
            if not (directory / "probe.json").is_file():
                rows.append(
                    {**row, "status": "incomplete" if directory.exists() else "unrun"}
                )
                continue
            try:
                probe = json.loads((directory / "probe.json").read_text())
                check_contract(
                    probe,
                    policy,
                    protocol,
                    candidates.get(policy["id"]),
                    scenarios[case],
                    sources,
                )
                for name, digest in probe["learned_policy_sha256"].items():
                    require(
                        sha256(directory / "learned-policy" / name) == digest,
                        "Copied policy artifact changed",
                    )
                # Includes a fresh independent physical audit and first-encoder /
                # every-command actuator projection verification.
                checked = audit_slew(directory)
                metrics = {}
                for robot in checked["robots"]:
                    score = probe["evaluation"][robot["robot"]]
                    metrics[robot["robot"]] = {
                        **robot,
                        "placement_xy_error_m": score["placement_xy_error_m"],
                        "minimum_root_height": score["minimum_root_height"],
                    }
                environments[policy["id"] + "/" + case] = {
                    key: probe[key]
                    for key in (
                        "opengl",
                        "render_backend",
                        "mujoco_version",
                        "source_sha256",
                    )
                }
                initial_states.setdefault(case, {})[policy["id"]] = probe[
                    "initial_qpos_sha256"
                ]
                rows.append(
                    {
                        **row,
                        "status": "audited",
                        "probe_sha256": sha256(directory / "probe.json"),
                        "recorded_seconds": probe["seconds"],
                        "termination_reason": probe["termination_reason"],
                        "robots": metrics,
                        "episode_success": all(
                            r["strict_success"] for r in metrics.values()
                        ),
                    }
                )
            except (
                ValueError,
                KeyError,
                TypeError,
                OSError,
                zipfile.BadZipFile,
            ) as exc:
                rows.append({**row, "status": "invalid", "error": str(exc)})
    # Cross-run matching is checked separately, so disagreement cannot silently
    # turn into a comparison using only the first successful subset.
    matching_errors = []
    values = list(environments.values())
    if any(value != values[0] for value in values[1:]):
        matching_errors.append(
            "Reported backend/runtime or source set differs across runs"
        )
    for case, states in initial_states.items():
        if len(set(states.values())) > 1:
            matching_errors.append("Initial state differs across policies: " + case)
    summary = {}
    for policy in protocol["policies"]:
        group = [row for row in rows if row["policy"] == policy["id"]]
        valid = [row for row in group if row["status"] == "audited"]
        summary[policy["id"]] = {
            "expected_physical_cases": len(protocol["cases"]),
            "audited_physical_cases": len(valid),
            "successful_physical_cases": sum(row["episode_success"] for row in valid),
            "observed_robot_placements": sum(len(row["robots"]) for row in valid),
            "successful_robot_placements": sum(
                r["strict_success"] for row in valid for r in row["robots"].values()
            ),
            "unrun": sum(row["status"] == "unrun" for row in group),
            "incomplete": sum(row["status"] == "incomplete" for row in group),
            "invalid": sum(row["status"] == "invalid" for row in group),
        }
    return {
        "manifest_sha256": sha256(manifest_path),
        "expected_runs": len(rows),
        "complete_and_matched": all(row["status"] == "audited" for row in rows)
        and not matching_errors,
        "matching_errors": matching_errors,
        "summary": summary,
        "runs": rows,
        "scope": "Training-excluded conditions previously exercised by the original authored baseline; fresh frozen-candidate evaluations. Co-simulated robot placements are correlated. Missing/invalid runs are not physical failures or successes. Reports bind recorded evidence, not reconstructed dynamics or machine attestation.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.runs, args.source_archive, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete_and_matched": result["complete_and_matched"],
                "summary": result["summary"],
            }
        )
    )
