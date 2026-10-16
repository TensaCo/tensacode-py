"""Run the frozen guarded G1 controller through a declared regression protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

CASES = {
    "nominal": [],
    "shift": ["--x", "1.30", "--y", "-.17"],
    "height04": ["--height-shift", ".04"],
    "frame": ["--camera-bias", ".06", "0", "0"],
    "height02": ["--height-shift", ".02"],
    "height06": ["--height-shift", ".06"],
    "combined": ["--height-shift", ".03", "--x", "1.275", "--y", "-.21"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ik-backend", choices=["scipy", "alg0", "alg1", "alg2"], default="scipy"
    )
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    protocol = {
        "declared_at": datetime.now(UTC).isoformat(),
        "backend": args.ik_backend,
        "cases": {name: CASES[name] for name in args.cases},
        "interpretation": "First six are development/regression conditions. Combined condition was first declared after freezing the final controller. Deterministic single episodes do not establish a statistical success rate.",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    def run(name):
        command = [
            sys.executable,
            str(scripts.parent / "g1/budget.py"),
            "--output",
            str(args.output / (name + "-budget")),
            "--unit",
            "tensorcode-planning-" + name,
            "--",
            "xvfb-run",
            "-a",
            "env",
            "MUJOCO_GL=glfw",
            "__GLX_VENDOR_LIBRARY_NAME=mesa",
            sys.executable,
            str(scripts / "experiment.py"),
            "--mode",
            "planned",
            "--ik-backend",
            args.ik_backend,
            "--output",
            str(args.output / name),
            *CASES[name],
        ]
        return subprocess.run(command, check=False).returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = dict(zip(args.cases, pool.map(run, args.cases)))
    (args.output / "execution.json").write_text(json.dumps(codes, indent=2) + "\n")
    if any(codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
