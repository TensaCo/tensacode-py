"""Check retained ALOHA outcomes against contact traces and policy predictions."""

import argparse
import json
from pathlib import Path

import numpy as np


def validate(directory: Path):
    result = json.loads((directory / "result.json").read_text())
    trace = json.loads((directory / "trace.json").read_text())
    predictions = np.load(directory / "predictions.npz")["actions"]
    if len(trace) != result["steps"] or not trace:
        raise ValueError("Trace length does not match outcome")
    if predictions.shape != ((len(trace) + 9) // 10, 50, 14):
        raise ValueError("Prediction shape does not match receding-horizon protocol")
    maximum = 0
    for step, row in enumerate(trace):
        if row["step"] != step or not np.array_equal(
            row["action"], predictions[step // 10, step % 10]
        ):
            raise ValueError(
                "Executed action is not the corresponding policy prediction"
            )
        if len(row["sensor_state"]) != 14 or not np.isfinite(row["sensor_state"]).all():
            raise ValueError("Invalid sensor observation")
        contacts = {tuple(pair) for pair in row["evaluation_contacts"]}
        left = ("red_box", "vx300s_left/10_left_gripper_finger") in contacts
        right = ("red_box", "vx300s_right/10_right_gripper_finger") in contacts
        table = ("red_box", "table") in contacts
        reward = (
            (4 if not table else 3) if left else (2 if not table else 1) if right else 0
        )
        if reward != row["evaluation_reward"]:
            raise ValueError("Reported reward differs from recorded simulator contacts")
        maximum = max(maximum, reward)
    if maximum != result["max_reward"] or (maximum == 4) != result["success"]:
        raise ValueError("Outcome differs from contact-derived maximum reward")
    return {
        "steps": len(trace),
        "policy_predictions": len(predictions),
        "max_reward": maximum,
        "success": maximum == 4,
        "contact_and_action_checks": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))


if __name__ == "__main__":
    main()
