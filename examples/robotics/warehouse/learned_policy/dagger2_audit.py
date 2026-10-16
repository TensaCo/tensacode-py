"""Qualification gate; labels are never confused with actual robot commands."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .dagger2 import expert_step, project_action
from .geometry_model import GeometryRuntime


def audit(directory):
    results = {}
    for path in sorted(Path(directory).glob("r*/dagger2-episode.json.gz")):
        with gzip.open(path, "rt") as stream:
            episode = json.load(stream)
        with gzip.open(path.with_name("bc-episode.json.gz"), "rt") as stream:
            actual = json.load(stream)["steps"]
        steps = episode["steps"]
        if len(steps) != len(actual) or not steps:
            raise ValueError("Qualification records must be nonempty and aligned")
        errors = []
        replay_errors = []
        runtime = GeometryRuntime(Path(directory) / "learned-policy")
        for row, recorded in zip(steps, actual):
            if abs(row["t"] - recorded["t"]) > 1e-8:
                raise ValueError("Misaligned execution timestamps")
            if row["executed_source"] != (
                "expert" if expert_step(row["t"]) else "learner"
            ):
                raise ValueError("Assistance differs from preregistered schedule")
            predicted, _ = runtime.step(row["observation"])
            for key, value in predicted.items():
                replay_errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(value, dtype=float)
                                - np.asarray(
                                    row["learner_prediction"][key], dtype=float
                                )
                            )
                        )
                    )
                )
            projected = project_action(
                row["expert_requested_action"],
                np.asarray(row["previous_executed_arm_target"]),
            )
            for key in row["action"]:
                errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(projected[key], dtype=float)
                                - np.asarray(row["action"][key], dtype=float)
                            )
                        )
                    )
                )
            for key in row["executed_action"]:
                errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(row["executed_action"][key], dtype=float)
                                - np.asarray(recorded["action"][key], dtype=float)
                            )
                        )
                    )
                )
        nonterminal = np.mean([not row["action"]["done"] for row in steps])
        closed = np.mean([row["action"]["grip"] >= 0.014 for row in steps])
        learner = np.mean([row["executed_source"] == "learner" for row in steps])
        results[episode["robot_id"]] = {
            "steps": len(steps),
            "full_85_second_horizon": len(steps) == 4250,
            "nonterminal_label_fraction": float(nonterminal),
            "closed_gripper_label_fraction": float(closed),
            "learner_executed_fraction": float(learner),
            "maximum_action_error": max(errors),
            "maximum_learner_replay_error": max(replay_errors),
            "label_coverage_gate": bool(
                nonterminal >= 0.5
                and closed >= 0.05
                and learner >= 0.15
                and max(errors) <= 1e-12
                and max(replay_errors) <= 1e-6
            ),
        }
    if len(results) != 2:
        raise ValueError("Expected two robot episodes")
    probe = json.loads((Path(directory) / "probe.json").read_text())
    no_fall = probe["termination_reason"] != "fall_harness_stop"
    return {
        "qualification_passed": no_fall
        and all(
            r["full_85_second_horizon"] and r["label_coverage_gate"]
            for r in results.values()
        ),
        "termination_reason": probe["termination_reason"],
        "no_fall": no_fall,
        "robots": results,
        "scope": "Coverage and execution gate only; independent no-fall physics audit also required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory / "dagger2-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
