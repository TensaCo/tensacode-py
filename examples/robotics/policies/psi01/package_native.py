"""Package every completed frozen native-comparison cell; never hide failures."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import struct
from pathlib import Path


def has_mp4_index(path: Path) -> bool:
    """Our encoder writes a final moov atom; exclude interrupted unindexed media."""
    if not path.exists():
        return False
    end = path.stat().st_size
    with path.open("rb") as file:
        while file.tell() + 8 <= end:
            start = file.tell()
            size, kind = struct.unpack(">I4s", file.read(8))
            header = 8
            if size == 1:
                if file.tell() + 8 > end:
                    return False
                size = struct.unpack(">Q", file.read(8))[0]
                header = 16
            if size == 0:
                size = end - start
            if size < header or start + size > end:
                return False
            if kind == b"moov":
                return True
            file.seek(start + size)
    return False


def read_jsonl_prefix(path: Path) -> tuple[list[dict], list[str]]:
    """Accept a complete JSONL prefix after interruption, retaining original bytes."""
    if not path.exists():
        return [], [f"missing {path.name}"]
    data = path.read_text()
    lines = data.splitlines()
    rows, issues = [], []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not data.endswith("\n"):
                issues.append(
                    f"truncated final line in {path.name}; original bytes retained"
                )
            else:
                raise
    return rows, issues


def package(
    source: Path,
    output: Path,
    diagnostics: Path | None = None,
    budget: Path | None = None,
    startup_failure: Path | None = None,
) -> dict:
    protocol_file = Path(__file__).with_name("native_protocol.json")
    protocol = json.loads(protocol_file.read_text())
    protocol_hash = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_file, output / protocol_file.name)
    results, receipts, inputs, evidence_issues = [], {}, {}, {}
    arm_artifact_issues = {}
    for arm in protocol["arms"]:
        arm_dir = source / arm
        if not arm_dir.exists():
            continue
        shutil.copytree(arm_dir, output / arm)
        try:
            metadata = json.loads((arm_dir / "provenance.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            arm_artifact_issues[arm] = (
                "Missing or incomplete provenance; artifacts retained, no outcomes scored."
            )
            for condition in protocol["conditions"]:
                evidence_issues[arm, condition["id"]] = [arm_artifact_issues[arm]]
            continue
        assert metadata["arm"] == arm and metadata["task"] == protocol["task"]
        assert metadata["action_tracker"] == protocol["action_tracker"]
        assert (
            metadata["runner_sha256"]
            == "0bcba63d40ff65d21c972c839e3fa2c2e6a9f676fb40a697157942380ab3428f"
        )
        assert (
            hashlib.sha256((arm_dir / "native_eval.py").read_bytes()).hexdigest()
            == metadata["runner_sha256"]
        )
        assert (
            hashlib.sha256((arm_dir / "native_protocol.json").read_bytes()).hexdigest()
            == protocol_hash
        )
        assert metadata["protocol_sha256"] == protocol_hash
        assert metadata["checkpoint_sha256"] == protocol["checkpoint_sha256"][arm]
        for condition in protocol["conditions"]:
            cid = condition["id"]
            directory = arm_dir / cid
            rows_path = directory / "rows.jsonl"
            if not directory.exists():
                continue
            rows, row_issues = read_jsonl_prefix(rows_path)
            configs = [row["config"] for row in rows]
            assert len(configs) == len(set(configs)), "duplicate episode identity"
            assert set(configs).issubset(protocol["configs"]), (
                "unknown episode identity"
            )
            calls, call_issues = read_jsonl_prefix(directory / "policy_inputs.jsonl")
            issues = row_issues + call_issues
            if not calls:
                issues.append("no recorded model-input audit")
            evidence_issues[arm, cid] = issues
            assert all(
                len(call["rgb_sha256"]) == 64 and len(call["state_sha256"]) == 64
                for call in calls
            )
            if calls and condition["instruction"] is None:
                assert {call["instruction"] for call in calls} == {
                    "grasp the object on the tabletop."
                }
            if calls and condition["instruction"] is not None:
                assert {call["instruction"] for call in calls} == {
                    condition["instruction"]
                }
            inputs[arm, cid] = sorted({call["instruction"] for call in calls})
            for row in rows:
                assert row["arm"] == arm and row["condition"] == cid
                assert row["protocol_sha256"] == protocol_hash
                assert row["checkpoint_sha256"] == metadata["checkpoint_sha256"]
                native = row["native_row"]
                assert (
                    row["task"] == protocol["task"]
                    and row["level"] == protocol["scene_level"]
                )
                assert row["repeat"] == 0 and native["seed"] == row["config"]
                assert native["task"] == f"simple/{protocol['task']}"
                assert native["env_id"] == "simple" and native["body"] == "g1_simple"
                assert native["success_privileged"] == row["success"]
                assert (
                    native["outcome"] == row["outcome"]
                    and native["steps"] == row["steps"]
                )
                env = native["provenance"]["env"]
                assert (
                    env["level"] == protocol["scene_level"]
                    and env["render"] == protocol["render"]
                )
                assert env["agent"] == "Psi0Agent" and env["sim_mode"] == "mujoco_isaac"
                for component in ("psi0", "simple"):
                    assert (
                        env["upstream"][component]["sha"]
                        == protocol["source_revisions"][component]
                    )
                    assert env["upstream"][component]["dirty"] is False
                receipt = row["initial_reset_audit"]
                assert receipt["receipt"]["config"] == row["config"]
                assert receipt["receipt"]["level"] == protocol["scene_level"]
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
                video_exists = has_mp4_index(output / video)
                if not video_exists:
                    issues.append(
                        f"video unavailable for completed config {row['config']}"
                    )
                results.append(
                    {
                        "id": f"{arm}-{cid}-c{row['config']}",
                        "arm": arm,
                        "condition": cid,
                        "config": row["config"],
                        "success": row["success"],
                        "outcome": row["outcome"],
                        "steps": row["steps"],
                        "simulated_seconds": native["time"],
                        "video": str(video) if video_exists else None,
                        "input_audit_available": bool(calls) and not call_issues,
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
                    "recorded_outcomes": len(rows),
                    "recorded_failures": sum(not r["success"] for r in rows),
                    "missing_outcomes": len(protocol["configs"]) - len(rows),
                    "status": "complete"
                    if len(rows) == len(protocol["configs"])
                    and not evidence_issues.get((arm, cid))
                    else "incomplete"
                    if (source / arm / cid).exists()
                    else "unrun",
                    "evidence_issues": evidence_issues.get((arm, cid), []),
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
    all_state_matching = []
    for config in protocol["configs"]:
        hashes = {
            f"{arm}/{cid}": value
            for (arm, cid, c), value in receipts.items()
            if c == config
        }
        all_state_matching.append(
            {
                "config": config,
                "cells": hashes,
                "partial_initial_state_equal": len(hashes) > 1
                and len(set(hashes.values())) == 1,
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
                            "simulated_seconds": row["simulated_seconds"],
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
        "all_conditions_initial_state_comparison": all_state_matching,
        "arm_artifact_issues": arm_artifact_issues,
        "completion_scope": "Only recorded native outcome rows contribute to attempted/success/failure counts. Missing or interrupted rows are unknown outcomes, never physical failures. Incomplete artifacts are retained verbatim; media/input audit availability is reported separately.",
        "verification_scope": {
            "initial_state": "Partial reset receipt (joint/root/object/palm state), not full simulator or RNG state equivalence.",
            "runtime": "Episode receipts bind clean Psi0/SIMPLE commits, native agent, render and task. RRP revision is protocol-declared; full contemporaneous external RRP source bytes were not captured.",
            "model_inputs": "Exact fixed image transform and nfe/horizon assertions are bound by frozen runner SHA and boundary tests; hashes record observed model inputs. Raw pre-corruption frames were not retained per inference, so individual mask execution is not independently reconstructed from hashes.",
        },
        "qualification": protocol["analysis"].replace(
            "training-matched control",
            "configuration-matched control (historical data-byte identity unverified)",
        ),
        "limitations": protocol["limitations"],
        "input_instructions": {
            f"{arm}/{cid}": texts for (arm, cid), texts in inputs.items()
        },
    }
    audit_dir = training_audit.parent
    for notice in ("PSI0-LICENSE.txt", "SIMPLE-LICENSE.txt"):
        if (audit_dir / notice).exists():
            shutil.copy2(audit_dir / notice, output / notice)
    for arm in protocol["arms"]:
        # An interrupted arm can have diagnostics but zero recorded outcomes.
        if not any(
            path is not None and path.exists()
            for path in (
                source / arm,
                diagnostics / arm if diagnostics is not None else None,
                budget / f"native-{arm}-retry2" if budget is not None else None,
            )
        ):
            continue
        runtime = output / "runtime" / arm
        for base, source_dir, allowed in (
            (diagnostics, arm, {"amendment.json", "native_launcher.py"}),
            (budget, f"native-{arm}-retry2", {"cuda-preflight.json", "receipt.json"}),
        ):
            if base is None or not (base / source_dir).exists():
                continue
            runtime.mkdir(parents=True, exist_ok=True)
            for path in (base / source_dir).iterdir():
                if path.name in allowed:
                    shutil.copy2(path, runtime / path.name)
                elif path.suffix in {".log", ".jsonl"}:
                    with (
                        (runtime / f"{path.name}.gz").open("wb") as target,
                        gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed,
                    ):
                        compressed.write(path.read_bytes())
    if startup_failure is not None:
        destination = output / "runtime" / "startup-attempt1"
        destination.mkdir(parents=True)
        for path in startup_failure.iterdir():
            if path.name in {"receipt.json", "cuda-preflight.json"}:
                shutil.copy2(path, destination / path.name)
            elif path.name in {"resources.jsonl", "stdout.log"}:
                with (
                    (destination / f"{path.name}.gz").open("wb") as target,
                    gzip.GzipFile(fileobj=target, mode="wb", mtime=0) as compressed,
                ):
                    compressed.write(path.read_bytes())
        ledger = startup_failure.parent / "ledger.jsonl"
        if ledger.exists():
            events = [
                json.loads(line) for line in ledger.read_text().splitlines() if line
            ]
            (destination / "events.json").write_text(
                json.dumps(
                    [
                        event
                        for event in events
                        if event.get("job") == startup_failure.name
                    ],
                    indent=2,
                )
                + "\n"
            )
        (destination / "README.txt").write_text(
            "Infrastructure attempt stalled during Isaac renderer startup before any policy outcome. Stopped by the resource coordinator; excluded from episode scores. The approved retry isolates Isaac from model MPS.\n"
        )
    lines = [
        "# Native Ψ policy comparison",
        "",
        "Same four SIMPLE tabletop scene configurations, G1/Dex3 embodiment and AMO tracker.",
        "Fresh closed-loop evaluation; every recorded outcome and video retained.",
        "",
        "| Policy | Nominal | Paraphrase | Occlusion |",
        "| --- | --- | --- | --- |",
    ]
    names = {
        "psi01": "Ψ₀.1 10k",
        "bc": "Ψ₀ BC 10k",
        "released": "Public Ψ₀ 40k",
        "psi01-no-logits": "Ψ₀.1, relation logits disabled",
    }
    for arm in protocol["arms"]:
        cells = [
            next(
                c
                for c in counts
                if c["arm"] == arm and c["condition"] == condition["id"]
            )
            for condition in protocol["conditions"]
        ]
        scores = [
            (
                f"{c['successes']}/{c['attempted']} recorded"
                if c["attempted"]
                else "Not recorded"
            )
            + (" (incomplete)" if c["status"] == "incomplete" else "")
            for c in cells
        ]
        lines.append("| " + " | ".join([names[arm], *scores]) + " |")
    lines += [
        "",
        "The fixed paraphrase preserves the generic grasp task. The occlusion is a centered",
        "15% × 15% gray rectangle in image coordinates, applied before model preprocessing.",
        "Neither condition establishes compositional instruction understanding.",
        "",
        result["training_comparison"],
        "",
        "The no-logits arm, if recorded, disables learned relation-logit terms at inference",
        "while retaining morphology tokens and weights. It is not a retrained causal ablation.",
        "",
        "`results.json` includes all paired outcomes and partial reset-state comparisons.",
        "Original rows, checkpoint identities, recorded policy-input hashes and frozen source",
        "accompany each arm. Verification scope and missing evidence are explicit in that file.",
        "Partial reset equality does not prove full simulator or RNG-state equivalence.",
        "",
        "Success uses the unchanged privileged SIMPLE judge, never supplied as policy input.",
        "Control steps include the native warm-up; simulated time is steps at 50 Hz.",
        "Videos sample every third control step at 8 fps: player time is not simulated time.",
        "",
        "`runtime/` retains the approved Isaac-only MPS-isolation amendment and resource",
        "receipts when supplied. Model CUDA is partitioned; RTX rendering is monitored and",
        "duty-cycled, not hardware-partitioned. Raw utilization bursts remain in the logs.",
        "",
        "This is a small development diagnostic, not a warehouse or hardware benchmark.",
        "Upstream code notices are preserved; they do not relicense embedded visual assets.",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
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
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--startup-failure", type=Path)
    args = parser.parse_args()
    result = package(
        args.source, args.output, args.diagnostics, args.budget, args.startup_failure
    )
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
