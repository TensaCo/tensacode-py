"""Audit DAgger action attribution and replay learner inference, no truth inputs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .contract import actions
from .dagger import expert_block
from .data import sha256
from .model import PolicyRuntime


def audit(directory, policy):
    directory, policy = Path(directory), Path(policy)
    results = []
    for path in sorted(directory.glob("r*/dagger-episode.json.gz")):
        document = json.loads(gzip.decompress(path.read_bytes()))
        actual = json.loads(
            gzip.decompress((path.parent / "bc-episode.json.gz").read_bytes())
        )
        if len(document["steps"]) != len(actual["steps"]):
            raise ValueError("DAgger and actual action record lengths differ")
        runtime = PolicyRuntime(policy)
        config = document["dagger"]
        maximum = 0.0
        assistance = 0
        for row, executed in zip(document["steps"], actual["steps"]):
            if (
                row["observation"] != executed["observation"]
                or row["t"] != executed["t"]
            ):
                raise ValueError(
                    "Expert label did not use actual pre-action observation"
                )
            predicted, _ = runtime.step(row["observation"])
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(actions(predicted) - actions(row["learner_prediction"]))
                    )
                ),
            )
            expected = expert_block(
                config["seed"],
                int(document["robot_id"][1:]),
                row["t"],
                config["expert_probability"],
            )
            if row["executed_source"] != ("expert" if expected else "learner"):
                raise ValueError("Recorded assistance violates seeded protocol")
            if row["executed_action"] != executed["action"]:
                raise ValueError(
                    "Attributed command differs from actual issued command"
                )
            if expected and not np.array_equal(
                actions(row["action"])[:-1], actions(row["executed_action"])[:-1]
            ):
                raise ValueError("Expert-selected block did not execute expert command")
            assistance += expected
        if maximum > 1e-5:
            raise ValueError("Replayed learner predictions differ from collector")
        results.append(
            {
                "robot": document["robot_id"],
                "steps": len(document["steps"]),
                "expert_steps": assistance,
                "learner_steps": len(document["steps"]) - assistance,
                "max_replay_error": maximum,
                "episode_sha256": sha256(path),
                "passed": True,
            }
        )
    if not results:
        raise ValueError("No DAgger episodes")
    return {
        "scope": "sensor/action attribution and learner replay, not policy performance",
        "robots": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.policy)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
