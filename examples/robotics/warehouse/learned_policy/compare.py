"""Compare matched fresh physical rollouts; never substitute offline BC loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..validate_v4 import validate
from .data import sha256


def compare(authored_root, learned_roots, case_ids):
    """Compare any named learned policies against one matched authored expert."""
    if not learned_roots or "authored" in learned_roots:
        raise ValueError("Provide named learned policy roots distinct from authored")
    roots = {"authored": authored_root, **learned_roots}
    rows = []
    policy_hashes = {}
    for case in case_ids:
        paths = {kind: root / case / "probe.json" for kind, root in roots.items()}
        probes = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
        teacher = probes["authored"]
        for kind, candidate in probes.items():
            if kind != "authored":
                artifact_hashes = candidate["learned_policy_sha256"]
                if kind in policy_hashes and policy_hashes[kind] != artifact_hashes:
                    raise ValueError(
                        f"Policy artifact changed across final cases: {kind}"
                    )
                policy_hashes[kind] = artifact_hashes
            if teacher.get("arm_slew_rad") != candidate.get("arm_slew_rad"):
                raise ValueError(f"Actuator slew boundary mismatch for {case}: {kind}")
            if teacher["scenario"] != candidate["scenario"]:
                raise ValueError(f"Scenario mismatch for {case}: {kind}")
            if teacher["initial_qpos_sha256"] != candidate["initial_qpos_sha256"]:
                raise ValueError(f"Initial physics mismatch for {case}: {kind}")
            if not candidate.get("full_horizon"):
                raise ValueError(
                    f"Every policy must use a full physical horizon: {case}"
                )
            for key in (
                "requested_duration_s",
                "mujoco_version",
                "render_backend",
                "opengl",
                "source_sha256",
            ):
                if teacher[key] != candidate[key]:
                    raise ValueError(
                        f"Matched evaluation contract differs for {case}/{kind}: {key}"
                    )
            if set(teacher["evaluation"]) != set(candidate["evaluation"]):
                raise ValueError(f"Robot set mismatch for {case}: {kind}")
        validations = {kind: validate(path.parent) for kind, path in paths.items()}
        for robot in teacher["evaluation"]:
            metrics = {}
            for kind, probe in probes.items():
                score = probe["evaluation"][robot]
                metrics[kind] = {
                    key: score[key]
                    for key in (
                        "physical_success",
                        "controller_declared_completion",
                        "placement_xy_error_m",
                        "settled_seconds",
                        "minimum_root_height",
                    )
                }
                checked = validations[kind]["robots"][robot]
                for key in ("strict_current_settle_success", "current_settled_seconds"):
                    metrics[kind][key] = checked[key]
                metrics[kind]["recorded_seconds"] = probe["seconds"]
                metrics[kind]["termination_reason"] = probe["termination_reason"]
            rows.append(
                {
                    "case": case,
                    "robot": robot,
                    "metrics": metrics,
                    "initial_qpos_sha256": teacher["initial_qpos_sha256"],
                    "probe_sha256": {
                        kind: sha256(path) for kind, path in paths.items()
                    },
                    "policy_sha256": {
                        kind: probe["learned_policy_sha256"]
                        for kind, probe in probes.items()
                        if kind != "authored"
                    },
                }
            )
    successes = {
        kind: sum(row["metrics"][kind]["strict_current_settle_success"] for row in rows)
        for kind in roots
    }
    paired = {}
    for kind in learned_roots:
        paired[kind] = {
            "authored_only_successes": sum(
                row["metrics"]["authored"]["strict_current_settle_success"]
                and not row["metrics"][kind]["strict_current_settle_success"]
                for row in rows
            ),
            "learned_only_successes": sum(
                row["metrics"][kind]["strict_current_settle_success"]
                and not row["metrics"]["authored"]["strict_current_settle_success"]
                for row in rows
            ),
        }
    return {
        "scope": "matched small-data learned policies versus authored sensor controller, shared frozen Unitree locomotion",
        "primary_metric": "independent strict_current_settle_success (one continuous second ending at final sample)",
        "rows": rows,
        "robot_trials_per_policy": len(rows),
        "independent_physical_cases": len(case_ids),
        "successes": successes,
        "paired_against_authored": paired,
        "statistical_unit": "physical case; co-simulated robots are not independent repeats",
        "limitation": "small frozen perturbation pilot, not a foundation-model benchmark or proof of general superiority",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--authored", type=Path, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        help="NAME=RUN_ROOT, e.g. bc=/tmp/bc-evaluation",
    )
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    learned_roots = {
        name: Path(path)
        for name, path in (value.split("=", 1) for value in args.policy)
    }
    args.output.write_text(
        json.dumps(compare(args.authored, learned_roots, args.cases), indent=2) + "\n"
    )
