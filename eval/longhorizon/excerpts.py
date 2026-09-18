"""Pull short, honest excerpts out of a run's log for the write-up.

    PYTHONPATH=src:. python -m eval.longhorizon.excerpts --label improved --task own/coding_ledger [--lines 24]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SP = Path(os.environ.get("LH_SCRATCH", "/tmp/lh"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--lines", type=int, default=24)
    ap.add_argument("--what", default="both", choices=("chat", "shell", "both"))
    args = ap.parse_args()
    path = SP / "lh" / "logs" / args.label / (args.task.replace("/", "__") + ".json")
    data = json.loads(path.read_text())
    row = data["row"]
    print(f"# {args.label} / {args.task}: score={row['score']} passed={row['passed']} status={row['status']} "
          f"calls={row['model_calls']} commands={row['commands']} wall={row['wall_seconds']}s")
    print(f"# failed checks: {[k for k, v in (row.get('checks') or {}).items() if not v] or '-'}")
    print(f"# checker said: {row['detail'][:300]}\n")
    if args.what in ("chat", "both"):
        for who, text in (data.get("transcript") or [])[: args.lines]:
            body = " ".join(text.split())
            print(f"{who.upper():>9}: {body[:400]}")
    if args.what in ("shell", "both"):
        print("\n# commands")
        for b in (data.get("shell_log") or [])[: args.lines]:
            out = " ".join((b.get("out") or "").split())[:200]
            print(f"  $ {b['cmd'][:160]}\n      exit={b['exit']} {out}")


if __name__ == "__main__":
    main()
