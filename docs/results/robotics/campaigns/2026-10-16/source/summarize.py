"""Summarize durable campaign ledgers without treating process exit as task success."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(roots, start_time=None, end_time=None):
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
                if start_time is not None and start["time"] < start_time:
                    continue
                if end_time is not None and row["time"] > end_time:
                    continue
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
                cuda_path = resource.parent / "cuda-preflight.json"
                cuda = json.loads(cuda_path.read_text()) if cuda_path.exists() else None
                jobs.append(
                    {
                        "id": row["job"],
                        "host": start["host"],
                        "unit": start["unit"],
                        "backend": start["backend"],
                        "assigned_cpus": start.get("cpus"),
                        "memory_limit_gib": start.get("memory_gib"),
                        "gpu_headroom_policy": start.get("gpu_headroom_policy"),
                        "cuda_preflight": cuda,
                        "paused_samples": sum(bool(r.get("paused")) for r in samples),
                        "gpu_busy_above_80_percent_samples": sum(
                            (r.get("gpu_fraction") or 0) > 0.8 for r in samples
                        ),
                        "process_exit_code": row["exit_code"],
                        "process_succeeded": row["exit_code"] == 0
                        and row["reason"] == "process_exit",
                        "physical_report_present": probe is not None,
                        "independent_physical_validation": "not_assessed_by_resource_summary",
                        "finish_reason": row["reason"],
                        "started_unix": start["time"],
                        "finished_unix": row["time"],
                        "elapsed_wall_seconds": row["time"] - start["time"],
                        "reported_physical_success": probe.get("physical_success")
                        if probe
                        else None,
                        "reported_task_success": probe.get("success")
                        if probe
                        else None,
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
    hosts = {}
    for job in jobs:
        host = hosts.setdefault(
            job["host"],
            {
                "jobs": 0,
                "job_wall_seconds": 0,
                "minimum_available_memory_fraction": 1.0,
                "maximum_sampled_cpu_fraction": 0.0,
                "maximum_sampled_gpu_fraction": 0.0,
                "paused_samples": 0,
            },
        )
        host["jobs"] += 1
        host["job_wall_seconds"] += job["elapsed_wall_seconds"]
        host["paused_samples"] += job["paused_samples"]
        value = job["minimum_available_memory_fraction"]
        if value is not None:
            host["minimum_available_memory_fraction"] = min(
                host["minimum_available_memory_fraction"], value
            )
        for key in ("maximum_sampled_cpu_fraction", "maximum_sampled_gpu_fraction"):
            host[key] = max(host[key], job[key] or 0)
    return {
        "window": {
            "started_at_or_after_unix": start_time,
            "finished_at_or_before_unix": end_time,
        },
        "process_outcomes": {
            "finished": len(jobs),
            "succeeded": sum(j["process_succeeded"] for j in jobs),
            "nonzero_exit": sum(j["process_exit_code"] != 0 for j in jobs),
            "interrupted": sum(j["finish_reason"] != "process_exit" for j in jobs),
            "physical_reports_present": sum(j["physical_report_present"] for j in jobs),
            "independently_valid_physical_reports": None,
        },
        "hosts": hosts,
        "aggregation_note": "Job wall seconds overlap for parallel jobs and must not be interpreted as campaign elapsed time. Admission failures without a started ledger remain separate evidence.",
        "schema": "tensorcode.warehouse.campaign-summary.v1",
        "jobs": jobs,
        "limitation": "Utilization is sampled; graphics busy guards are reactive and permit recorded transients above80%. MPS bounds equivalent-SM execution allocation, not dedicated physical SMs or bandwidth. Task success is independently scored, not inferred from exit0.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.roots), indent=2) + "\n")


if __name__ == "__main__":
    main()
