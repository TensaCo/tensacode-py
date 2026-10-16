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


HANDLING_CASES = {
    "nominal": ["--destination", "1.20", "-.48", ".88"],
    "receiver_shift": ["--destination", "1.16", "-.46", ".90"],
    "heavy_payload": ["--destination", "1.20", "-.48", ".88", "--mass", ".20"],
    "camera_outage": [
        "--destination",
        "1.20",
        "-.48",
        ".88",
        "--camera-outage",
        "16",
        "22",
    ],
    "no_load_compensation": [
        "--destination",
        "1.20",
        "-.48",
        ".88",
        "--disable-load-compensation",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["planned", "handling"], default="planned")
    parser.add_argument(
        "--ik-backend", choices=["scipy", "alg0", "alg1", "alg2"], default="scipy"
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    cases = HANDLING_CASES if args.mode == "handling" else CASES
    args.cases = args.cases or list(cases)
    if set(args.cases) - cases.keys():
        parser.error("Unknown cases for selected mode")
    args.output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    protocol = {
        "declared_at": datetime.now(UTC).isoformat(),
        "backend": args.ik_backend,
        "cases": {name: cases[name] for name in args.cases},
        "mode": args.mode,
        "interpretation": "First six are development/regression conditions. Combined condition was first declared after freezing the final controller. Deterministic single episodes do not establish a statistical success rate.",
    }
    if args.mode == "handling":
        protocol["interpretation"] = (
            "Nominal is a development case. New receiver, payload and outage conditions are single prospective checks only if declared before controller changes; retain all outcomes and source snapshots. No-load compensation is a matched ablation."
        )
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    def run(name):
        command = [
            sys.executable,
            str(scripts.parent / "g1/budget.py"),
            "--output",
            str(args.output / (name + "-budget")),
            "--unit",
            "tensorcode-" + args.mode + "-" + name,
            "--",
            "xvfb-run",
            "-a",
            "env",
            "MUJOCO_GL=glfw",
            "__GLX_VENDOR_LIBRARY_NAME=mesa",
            sys.executable,
            str(scripts / "experiment.py"),
            "--mode",
            args.mode,
            "--ik-backend",
            args.ik_backend,
            "--output",
            str(args.output / name),
            *cases[name],
        ]
        return subprocess.run(command, check=False).returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = dict(zip(args.cases, pool.map(run, args.cases)))
    (args.output / "execution.json").write_text(json.dumps(codes, indent=2) + "\n")
    if any(codes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
