"""Execute a bounded JSON work order through a compiled task policy in MuJoCo."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import run
from policy import compile_policy


def parse_recipe(value):
    if not isinstance(value, dict) or set(value) - {
        "order",
        "revision_after_first_inspection",
        "rejected_parts",
        "inject_first_pick_miss",
    }:
        raise ValueError("Unknown recipe fields")

    def order(xs):
        if (
            not isinstance(xs, list)
            or not 1 <= len(xs) <= 3
            or any(x not in ("A", "B") for x in xs)
        ):
            raise ValueError("Orders require 1–3 slots, each A or B")
        return [float(x == "B") for x in xs]

    initial = order(value.get("order"))
    revision = value.get("revision_after_first_inspection")
    rejected = value.get("rejected_parts", [])
    if (
        not isinstance(rejected, list)
        or any(type(i) is not int or not 0 <= i < 4 for i in rejected)
        or len(set(rejected)) != len(rejected)
    ):
        raise ValueError("Rejected parts must be unique inventory indices 0–3")
    miss = value.get("inject_first_pick_miss", False)
    if type(miss) is not bool:
        raise ValueError("Miss injection must be boolean")
    return dict(
        initial_order=initial,
        new_order=initial if revision is None else order(revision),
        revise=revision is not None,
        reject=bool(rejected),
        reject_indices=rejected,
        miss=miss,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--substrate",
        choices=["alg0", "alg1", "alg2", "native_feedback"],
        default="alg2",
    )
    a = p.parse_args()
    recipe = parse_recipe(json.loads(a.recipe.read_text()))
    a.output.mkdir(parents=True, exist_ok=False)
    result, cell, cases = run(
        a.seed, "custom_recipe", a.substrate, record=True, recipe=recipe
    )
    (a.output / "recipe.json").write_text(a.recipe.read_text())
    (a.output / "receipt.json").write_text(json.dumps(result, indent=2))
    (a.output / "decisions.json").write_text(json.dumps(cases, indent=2))
    (a.output / "scene.xml").write_text(cell.xml)
    (a.output / "policy.json").write_text(compile_policy().to_json())
    np.savez_compressed(
        a.output / "trajectory.npz",
        qpos=np.array([f[0] for f in cell.frames]),
        time=np.array([f[1] for f in cell.frames]),
        labels=np.array([f[2] for f in cell.frames]),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))
    if not result["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
