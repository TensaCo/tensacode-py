"""Start/stop only the campaign's explicitly owned private CUDA MPS daemon."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory.resolve()
    if not str(path).startswith("/tmp/tensorcode-campaign-mps-"):
        parser.error(
            "Use an explicitly campaign-owned /tmp/tensorcode-campaign-mps-* directory"
        )
    owner = path / "owner.json"
    if path.exists() and not owner.exists():
        parser.error("Refusing to use an existing directory without campaign ownership")
    if owner.exists():
        evidence = json.loads(owner.read_text())
        if (
            evidence.get("campaign") != "warehouse-2026-10-16"
            or evidence.get("host") != os.uname().nodename
        ):
            parser.error("Ownership receipt does not match this campaign and host")
    elif args.action == "stop":
        parser.error("No owned daemon to stop")
    path.mkdir(mode=0o700, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    env = dict(
        os.environ,
        CUDA_MPS_PIPE_DIRECTORY=str(path),
        CUDA_MPS_LOG_DIRECTORY=str(path / "logs"),
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="70",
    )
    result = subprocess.run(
        ["nvidia-cuda-mps-control", *(["-d"] if args.action == "start" else [])],
        input=None if args.action == "start" else "quit\n",
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    owner.write_text(
        json.dumps(
            {
                "campaign": "warehouse-2026-10-16",
                "host": os.uname().nodename,
                "requested_active_thread_percentage": 70,
                "last_action": args.action,
                "exit_code": result.returncode,
            },
            indent=2,
        )
        + "\n"
    )
    print(result.stdout, result.stderr)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
