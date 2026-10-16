"""Package every completed frozen native-comparison cell; never hide failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def package(source: Path, output: Path) -> dict:
    protocol_file = Path(__file__).with_name("native_protocol.json")
    protocol = json.loads(protocol_file.read_text())
    protocol_hash = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_file, output / protocol_file.name)
    results, receipts, inputs = [], {}, {}
    for arm in protocol["arms"]:
        arm_dir = source / arm
        if not arm_dir.exists():
            continue
        shutil.copytree(arm_dir, output / arm)
        metadata = json.loads((arm_dir / "provenance.json").read_text())
        assert metadata["protocol_sha256"] == protocol_hash
        assert metadata["checkpoint_sha256"] == protocol["checkpoint_sha256"][arm]
        for condition in protocol["conditions"]:
            cid = condition["id"]
            directory = arm_dir / cid
            rows_path = directory / "rows.jsonl"
            if not rows_path.exists():
                continue
            rows = [
                json.loads(line) for line in rows_path.read_text().splitlines() if line
            ]
            assert len(rows) == len(protocol["configs"])
            assert {row["config"] for row in rows} == set(protocol["configs"])
            calls = [
                json.loads(line)
                for line in (directory / "policy_inputs.jsonl").read_text().splitlines()
                if line
            ]
            assert calls
            if condition["instruction"] is not None:
                assert {call["instruction"] for call in calls} == {
                    condition["instruction"]
                }
            inputs[arm, cid] = sorted({call["instruction"] for call in calls})
            for row in rows:
                assert row["arm"] == arm and row["condition"] == cid
                assert row["protocol_sha256"] == protocol_hash
                assert row["checkpoint_sha256"] == metadata["checkpoint_sha256"]
                receipt = row["initial_reset_audit"]
                encoded = json.dumps(receipt["receipt"], sort_keys=True)
                assert hashlib.sha256(encoded.encode()).hexdigest() == receipt["sha256"]
                receipts[arm, cid, row["config"]] = receipt["sha256"]
                suffix = "success" if row["success"] else "FAIL"
                video = (
                    Path(arm)
                    / cid
                    / "video"
                    / f"L0_{protocol['task']}_c{row['config']}r0_{suffix}.mp4"
                )
                assert (output / video).exists(), video
                results.append(
                    {
                        "id": f"{arm}-{cid}-c{row['config']}",
                        "arm": arm,
                        "condition": cid,
                        "config": row["config"],
                        "success": row["success"],
                        "outcome": row["outcome"],
                        "steps": row["steps"],
                        "video": str(video),
                        "initial_reset_audit_sha256": receipt["sha256"],
                        "source": row["source"],
                    }
                )
    counts, matching = [], []
    for arm in protocol["arms"]:
        for condition in protocol["conditions"]:
            cid = condition["id"]
            rows = [r for r in results if r["arm"] == arm and r["condition"] == cid]
            counts.append(
                {
                    "arm": arm,
                    "condition": cid,
                    "attempted": len(rows),
                    "successes": sum(r["success"] for r in rows),
                    "expected": len(protocol["configs"]),
                }
            )
    for cid in [c["id"] for c in protocol["conditions"]]:
        for config in protocol["configs"]:
            compared = {
                arm: receipts[arm, cid, config]
                for arm in protocol["arms"]
                if (arm, cid, config) in receipts
            }
            matching.append(
                {
                    "condition": cid,
                    "config": config,
                    "arms": list(compared),
                    "partial_initial_state_equal": len(compared) > 1
                    and len(set(compared.values())) == 1,
                }
            )
    paired_outcomes = []
    for condition in protocol["conditions"]:
        for config in protocol["configs"]:
            paired_outcomes.append(
                {
                    "condition": condition["id"],
                    "config": config,
                    "arms": {
                        row["arm"]: {
                            "success": row["success"],
                            "outcome": row["outcome"],
                            "steps": row["steps"],
                            "video": row["video"],
                        }
                        for row in results
                        if row["condition"] == condition["id"]
                        and row["config"] == config
                    },
                }
            )
    training_audit = (
        Path(__file__).resolve().parents[4]
        / "docs/results/robotics/baselines/psi01/training_match_audit.json"
    )
    if training_audit.exists():
        shutil.copy2(training_audit, output / training_audit.name)
    result = {
        "paired_outcomes": paired_outcomes,
        "training_comparison": "Configuration-matched local10k BC; historical dataset bytes and per-batch order were not content-hashed. Public40k is a deployment reference, not a training-matched control.",
        "protocol_id": protocol["id"],
        "protocol_sha256": protocol_hash,
        "counts": counts,
        "episodes": results,
        "initial_state_comparison": matching,
        "qualification": protocol["analysis"],
        "limitations": protocol["limitations"],
        "input_instructions": {
            f"{arm}/{cid}": texts for (arm, cid), texts in inputs.items()
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "hashes.json"
    }
    (output / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.source, args.output)
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
