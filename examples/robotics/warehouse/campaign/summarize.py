"""Summarize durable campaign ledgers without treating process exit as task success."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(roots):
    jobs = []
    seen = set()
    for root in roots:
        for ledger in root.rglob("*ledger.jsonl"):
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            starts = {row["job"]: row for row in rows if row.get("event") == "started"}
            for row in rows:
                if row.get("event") != "finished":
                    continue
                start = starts[row["job"]]
                if start["unit"] in seen:
                    continue
                seen.add(start["unit"])
                resource = ledger.parent / row["job"] / "resources.jsonl"
                if not resource.exists() and ledger.name.startswith("peer-lane"):
                    resource = (
                        ledger.parent
                        / (ledger.name.removesuffix("-ledger.jsonl") + "-budget")
                        / row["job"]
                        / "resources.jsonl"
                    )
                samples = (
                    [json.loads(line) for line in resource.read_text().splitlines()]
                    if resource.exists()
                    else []
                )
                command = start["command"]
                probe = None
                if "--output" in command:
                    path = Path(command[command.index("--output") + 1]) / "probe.json"
                    if path.exists():
                        probe = json.loads(path.read_text())
                jobs.append(
                    {
                        "id": row["job"],
                        "host": start["host"],
                        "unit": start["unit"],
                        "backend": start["backend"],
                        "process_exit_code": row["exit_code"],
                        "finish_reason": row["reason"],
                        "elapsed_wall_seconds": row["time"] - start["time"],
                        "reported_physical_success": probe.get("physical_success")
                        if probe
                        else None,
                        "reported_task_success": probe.get("success") if probe else None,
                        "minimum_available_memory_fraction": min(
                            (r["memory_available_fraction"] for r in samples),
                            default=None,
                        ),
                        "maximum_sampled_cpu_fraction": max(
                            (r["cpu_fraction"] for r in samples), default=None
                        ),
                        "maximum_sampled_gpu_fraction": max(
                            (r["gpu_fraction"] or 0 for r in samples), default=None
                        ),
                        "resource_samples": len(samples),
                        "ledger": str(ledger),
                        "ledger_sha256": hashlib.sha256(
                            ledger.read_bytes()
                        ).hexdigest(),
                        "protocol": start.get("protocol"),
                    }
                )
    return {
        "schema": "tensorcode.warehouse.campaign-summary.v1",
        "jobs": jobs,
        "limitation": "Utilization is sampled, not a guarantee against outside workload spikes; task success is independently scored, not inferred from exit0.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.roots), indent=2) + "\n")


if __name__ == "__main__":
    main()
