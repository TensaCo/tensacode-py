"""Fetch the pinned Unitree G1 robot, configuration and pretrained walking weights."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REVISION = "1425b15f73bd4095f0df53709d7c389c3eb9e790"
REPOSITORY = "https://github.com/unitreerobotics/unitree_rl_mjlab.git"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        found = subprocess.check_output(
            ["git", "-C", str(a.output), "rev-parse", "HEAD"], text=True
        ).strip()
        if found != REVISION:
            raise RuntimeError(
                "Existing checkout has a different revision; choose a fresh output path"
            )
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                REPOSITORY,
                str(a.output),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(a.output),
                "sparse-checkout",
                "set",
                "src/assets/robots/unitree_g1",
                "deploy/robots/g1/config/policy/velocity",
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(a.output), "checkout", "--detach", REVISION], check=True
        )
    print(a.output.resolve())


if __name__ == "__main__":
    main()
