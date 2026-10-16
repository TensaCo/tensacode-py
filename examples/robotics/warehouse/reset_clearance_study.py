"""Read-only strict physical and provenance audit of the v6 development trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .sixjob_environment_study import sha, telemetry
from .validate_jobs import validate


def summarize(protocol_path, preflight_path):
    protocol = json.loads(protocol_path.read_text())
    preflight = {c["id"]: c for c in json.loads(preflight_path.read_text())["cases"]}
    reference = json.loads((Path(protocol["reference_run"]) / "probe.json").read_text())
    rows = []
    for case in protocol["cases"]:
        path = Path(case["run_path"])
        row = {"id": case["id"], "status": "pending"}
        if not (path / "clearance-proof.json").exists():
            rows.append(row)
            continue
        proof = json.loads((path / "clearance-proof.json").read_text())
        probe = json.loads((path / "probe.json").read_text())
        if proof.get("smoke_only") is not False:
            raise ValueError(
                "Smoke/missing execution kind cannot count as a physical case"
            )
        if proof["protocol_sha256"] != sha(protocol_path) or proof["case"] != case:
            raise ValueError("V6 case/protocol proof mismatch")
        if sha(path / "clearance-protocol.json") != sha(protocol_path):
            raise ValueError("V6 copied protocol mismatch")
        if proof["extension_sources"] != protocol["extension_sources"]:
            raise ValueError("V6 extension proof mismatch")
        for source, digest in protocol["extension_sources"].items():
            if sha(path / Path(source).name) != digest:
                raise ValueError("V6 extension source mismatch")
        if probe["source_sha256"] != reference["source_sha256"]:
            raise ValueError("V5 base sources changed")
        if sha(path / "scene.xml") != preflight[case["id"]]["scene_sha256"]:
            raise ValueError("V6 physical scene mismatch")
        if probe["initial_qpos_sha256"] != preflight[case["id"]]["initial_qpos_sha256"]:
            raise ValueError("V6 initial state mismatch")
        if (
            probe["requested_duration_s"] != 170
            or probe["robots"] != 3
            or probe["jobs_per_robot"] != 2
            or probe["dispatch_substrate"] != "alg2"
        ):
            raise ValueError("V6 runtime configuration mismatch")
        for name, digest in protocol["mission_sha256"].items():
            if sha(path / "mission" / name) != digest:
                raise ValueError("Reused model mission changed")
        audit = validate(path)
        records = json.loads((path / "records.json").read_text())
        diagnostics = {
            r: {j: telemetry(records, r, j) for j in ("a", "b")}
            for r in audit["robots"]
        }
        reset_receipts = {}
        for robot in audit["robots"]:
            receipt = None
            prior_phase = None
            for record in records:
                state = record["robots"][robot]
                if state["phase"] == "reset_arm" and prior_phase != "reset_arm":
                    receipt = {
                        "time": record["t"],
                        "controller_state": state["controller_state"],
                    }
                    break
                prior_phase = state["phase"]
            reset_receipts[robot] = receipt
        row.update(
            status="complete",
            validation=audit,
            telemetry=diagnostics,
            successful_placements=sum(
                s["physical_success"]
                for r in audit["robots"].values()
                for s in r.values()
            ),
            all_jobs_success=audit["success"],
            reset_admission=reset_receipts,
        )
        rows.append(row)
    return {
        "schema": "tensorcode.warehouse.reset-clearance.results.v6",
        "protocol_sha256": sha(protocol_path),
        "scope": protocol["scope"],
        "complete": all(r["status"] == "complete" for r in rows),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.protocol, args.preflight)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "completed_cases": sum(
                    r["status"] == "complete" for r in report["cases"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
