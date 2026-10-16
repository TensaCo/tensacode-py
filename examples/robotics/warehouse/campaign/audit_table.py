"""Publish the complete independently audited authored robustness matrix."""

import argparse
import hashlib
import json
from pathlib import Path


def build(protocol, runs):
    rows = []
    for case in json.loads(protocol.read_text())["cases"]:
        path = runs / case["id"] / "validation-v4.json"
        report = json.loads(path.read_text())
        if not report["validated"] or not report["source_snapshot_verified"]:
            raise ValueError(f"Unaudited run: {case['id']}")
        robots = report["robots"]
        rows.append(
            {
                "id": case["id"],
                "category": "teacher-only"
                if case["id"].startswith("teacher-")
                else "robustness",
                "split": case["split"],
                "pair": case.get("pair"),
                "strict_successes": sum(
                    bool(r["strict_current_settle_success"]) for r in robots.values()
                ),
                "robots": len(robots),
                "legacy_successes": sum(
                    bool(r["physical_success"]) for r in robots.values()
                ),
                "final_xy_error_mm": {
                    key: round(value["placement_xy_error_m"] * 1000, 3)
                    for key, value in robots.items()
                },
                "current_settled_seconds": {
                    key: value["current_settled_seconds"]
                    for key, value in robots.items()
                },
                "final_phase": {
                    key: value["final_phase"] for key, value in robots.items()
                },
                "seconds": report["seconds"],
                "validation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "initial_qpos_sha256": report["initial_qpos_sha256"],
            }
        )
    return {
        "schema": "tensorcode.warehouse.robustness-audit.v1",
        "policy": "authored baseline without grasp recovery",
        "primary_metric": "One continuous second of valid released receiver contact ending at the final sample, with speed/clearance/position/root-height criteria.",
        "scope": "18 robustness cases plus3teacher-only cases; all outcomes retained. Early-stop authored development runs, not the later fixed85second BC head-to-head.",
        "cases": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    args = p.parse_args()
    result = build(args.protocol, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "| Case | Category | Strict final success | Final XY error r0 / r1 (mm) |",
        "|---|---|---:|---:|",
    ]
    for row in result["cases"]:
        errors = " / ".join(f"{v:.1f}" for v in row["final_xy_error_mm"].values())
        lines.append(
            f"| {row['id']} | {row['category']} | {row['strict_successes']}/{row['robots']} | {errors} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
