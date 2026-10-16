"""Collect completed peer jobs only; never copy an actively written rollout."""

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


def collect(output):
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "harvested.json"
    completed = set(json.loads(receipt.read_text())) if receipt.exists() else set()
    for lane in (0, 1):
        remote = str(output / f"peer-lane{lane}-budget")
        result = subprocess.run(
            ["ssh", "gb10-direct", "cat", remote + "/ledger.jsonl"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            continue
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        starts = {row["job"]: row for row in rows if row["event"] == "started"}
        for row in rows:
            if row["event"] != "finished" or row["job"] in completed:
                continue
            start = starts[row["job"]]
            command = start["command"]
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            copy = subprocess.run(
                ["rsync", "-a", f"gb10-direct:{destination}/", str(destination) + "/"],
                check=False,
            )
            if copy.returncode:
                continue
            budget = output / f"peer-lane{lane}-budget"
            budget.mkdir(exist_ok=True)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    f"gb10-direct:{remote}/{row['job']}",
                    str(budget) + "/",
                ],
                check=True,
            )
            completed.add(row["job"])
            receipt.write_text(json.dumps(sorted(completed), indent=2) + "\n")
        (output / f"peer-lane{lane}-ledger.jsonl").write_text(result.stdout)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/tensorcode-campaign"))
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    deadline = dt.datetime(2026, 10, 16, 23, 54, tzinfo=dt.UTC).timestamp()
    while True:
        print(
            json.dumps({"time": time.time(), "harvested": len(collect(args.output))}),
            flush=True,
        )
        if not args.watch or time.time() >= deadline:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
