"""Bounded infrastructure retry: isolate Isaac graphics from parent CUDA MPS.

Does not change frozen policy, task, seeds, observations, or success criteria.
Use only under the approved exclusive GPU lease and graphics utilization guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

FROZEN_RUNNER_SHA256 = (
    "0bcba63d40ff65d21c972c839e3fa2c2e6a9f676fb40a697157942380ab3428f"
)
FROZEN_PROTOCOL_SHA256 = (
    "69b43f92a6f11f002d910fb8faafcaa4185a58b060ea3e6d1a228e618c0f797d"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    for path, expected in (
        (args.runner, FROZEN_RUNNER_SHA256),
        (args.runner.with_name("native_protocol.json"), FROZEN_PROTOCOL_SHA256),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Frozen source mismatch: {path}")
    args.diagnostics.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.diagnostics / "native_launcher.py")
    metadata = {
        "amendment": "infrastructure-retry-2-2026-10-16",
        "reason": "Attempt1 stalled at Isaac Hydra initialization before any policy outcome; no evidence of resource-guard pauses.",
        "changes": [
            "Remove CUDA_MPS_* variables only from Isaac child environment",
            "Write each Isaac worker stdout/stderr to its own diagnostic log",
            "Bound worker RPC startup wait to120seconds",
        ],
        "unchanged": [
            "Frozen runner and protocol hashes",
            "All policy weights and semantic evaluation cells",
            "Parent model MPS partition",
            "Outer exclusive lease and graphics utilization guard",
        ],
        "runner_sha256": FROZEN_RUNNER_SHA256,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parent_mps": {
            k: v for k, v in os.environ.items() if k.startswith("CUDA_MPS_")
        },
        "qualification": "MPS/RTX startup interaction is a hypothesis, not an established cause. This amendment is retained even if retry fails.",
    }
    (args.diagnostics / "amendment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    from rrp.envs import simple

    original_env, original_init = simple.worker_env, simple.SimpleEnv.__init__
    count = 0

    def worker_env(*a, **kw):
        env = original_env(*a, **kw)
        return {
            key: value for key, value in env.items() if not key.startswith("CUDA_MPS_")
        }

    def worker_init(self, *a, **kw):
        nonlocal count
        kw["log"] = args.diagnostics / f"isaac-worker-{count}.log"
        kw["start_timeout"] = 120.0
        count += 1
        return original_init(self, *a, **kw)

    simple.worker_env = worker_env
    simple.SimpleEnv.__init__ = worker_init
    tail = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
    sys.argv = [str(args.runner), *tail]
    runpy.run_path(str(args.runner), run_name="__main__")


if __name__ == "__main__":
    main()
