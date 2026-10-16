"""Record the released Ψ0 on its native SIMPLE G1/Dex3 task.

Requires the pinned external RRP/SIMPLE/Ψ0 environments described in README.md.
Records actual closed-loop inference, not action replay. Never modifies weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing recording")
    args.output.mkdir(parents=True)
    # Upstream inference and observed-frame recorder, with unchanged task hooks.
    from rrp.harness.eval.video_psi0 import run_arm

    run_arm(
        SimpleNamespace(
            task="G1WholebodyTabletopGraspMP-v0",
            task_key="tabletop",
            arm="released",
            weights=None,
            stage_a=None,
            seeds=str(args.seed),
            out=str(args.output),
        )
    )


if __name__ == "__main__":
    main()
