"""Independently verify common actuator slew bounds from recorded commands."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from ..validate_v4 import validate
from .data import sha256


def audit(directory):
    directory = Path(directory)
    probe = json.loads((directory / "probe.json").read_text())
    limit = probe["arm_slew_rad"]
    if limit != 0.018:
        raise ValueError("Unexpected preregistered arm slew bound")
    path = directory / "records.json"
    records = (
        json.loads(path.read_text())
        if path.exists()
        else json.loads(gzip.decompress(path.with_suffix(".json.gz").read_bytes()))
    )
    validation = validate(directory)
    result = []
    for robot in sorted(probe["evaluation"]):
        episode_path = directory / robot / "bc-episode.json.gz"
        episode = json.loads(gzip.decompress(episode_path.read_bytes()))
        previous = np.asarray(
            episode["steps"][0]["observation"]["joint_position"][22:29]
        )
        maximum = 0.0
        clipped = 0
        for row, step in zip(records, episode["steps"], strict=True):
            telemetry = row["robots"][robot]["arm_slew"]
            requested = np.asarray(telemetry["requested_arm_target"])
            actual = np.asarray(step["action"]["arm_target"])
            expected = previous + np.clip(requested - previous, -limit, limit)
            if not np.allclose(actual, expected, atol=1e-7, rtol=0):
                raise ValueError("Actual command does not equal common slew projection")
            if not np.array_equal(actual, telemetry["executed_arm_target"]):
                raise ValueError("Command telemetry differs from recorded action")
            increment = float(np.max(np.abs(actual - previous)))
            if increment > limit + 1e-7:
                raise ValueError("Executed arm target violates per-step bound")
            maximum = max(maximum, increment)
            clipped += int(telemetry["clipped_joints"] > 0)
            previous = actual
        result.append(
            {
                "robot": robot,
                "steps": len(records),
                "maximum_executed_increment_rad": maximum,
                "clipped_steps": clipped,
                "clipped_fraction": clipped / len(records),
                "strict_success": validation["robots"][robot][
                    "strict_current_settle_success"
                ],
                "episode_sha256": sha256(episode_path),
            }
        )
    return {
        "policy": probe["policy"],
        "probe_sha256": sha256(directory / "probe.json"),
        "robots": result,
        "scope": "command-bound audit and independent physical scoring, not force or collision certification",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
