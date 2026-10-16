"""Reproduce all four declared paired tests under per-process resource guards.

Each pair shares one frozen commanded demonstration. No scene is selected by its
outcome. Run this from the Python repository with the virtualenv interpreter.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CASES = {
    "nominal": [],
    "shift": ["--x", "1.30", "--y", "-.17"],
    "height": ["--height-shift", ".04"],
    "frame": ["--camera-bias", ".06", "0", "0"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ik-multistart", action="store_true")
    p.add_argument("--feedback-clock", action="store_true")
    p.add_argument("--prefix", default="")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    budget = scripts.parent / "humanoid/humanoid_budget.py"

    def run(case, mode, options):
        name = f"{args.prefix}{case}-{mode}"
        command = [
            sys.executable,
            str(budget),
            "--output",
            str(args.output / (name + "-budget")),
            "--unit",
            "tensorcode-transfer-" + name,
            "--",
            "xvfb-run",
            "-a",
            "env",
            "MUJOCO_GL=glfw",
            "__GLX_VENDOR_LIBRARY_NAME=mesa",
            sys.executable,
            str(scripts / "experiment.py"),
            "--mode",
            mode,
            "--context",
            str(args.context.resolve()),
            "--output",
            str(args.output / name),
            *options,
        ]
        if args.ik_multistart:
            command.append("--ik-multistart")
        if args.feedback_clock:
            command.append("--feedback-clock")
        subprocess.run(command, check=True)

    for case, options in CASES.items():
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(run, case, mode, options) for mode in ["fixed", "context"]
            ]
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
