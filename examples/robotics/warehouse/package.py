"""Package completed warehouse evidence without converting unfinished trials to successes.

Run from the Python repository: python -m examples.robotics.warehouse.package
Only directories containing a completed probe or an explicit interruption record
are admitted. Rendering caches and short integration smokes are not task evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .validate import validate

ACTION_SOURCES = {
    "warehouse/physics.py",
    "warehouse/physics_scene.py",
    "warehouse/physics_sensors.py",
    "warehouse/scenery.py",
    "warehouse/representation.py",
    "warehouse/language.py",
    "warehouse/coordination.py",
    "transfer/handling.py",
    "transfer/planning.py",
    "transfer/sensors.py",
    "transfer/box_qp.py",
    "g1/model.py",
}
SENSOR_COMPLETION_RUNS = {
    "model-parallel-v3",
    "two-sequence-v3",
    "three-parallel-alg2-v3",
}


def write_json(path, value, *, compact=False):
    path.write_text(
        json.dumps(value, indent=None if compact else 2, allow_nan=False) + "\n"
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package_questions(destination: Path, output: Path):
    """Ask recorded observed scenes; no evaluation fields or simulator reads."""
    from .language import answer_question
    from .model_language import decode_request, ground_request
    from .representation import SceneGraph

    result = {
        "scope": "Questions answered at recorded observation time, not live execution",
        "scenes": [],
    }
    receipt_path = output / "language/location-query-receipt.json"
    for frame in ("initial", "final"):
        path = destination / f"{frame}-scene-graph.json"
        if not path.exists():
            continue
        scene = SceneGraph.from_dict(json.loads(path.read_text()))
        if not scene.entities:
            result["scenes"].append(
                {
                    "path": path.name,
                    "sha256": digest(path.read_bytes()),
                    "status": "unknown",
                    "reason": "No observed entities; no observation timestamp available.",
                }
            )
            continue
        now = max(e.observed_at for e in scene.entities.values())
        row = {
            "path": path.name,
            "sha256": digest(path.read_bytes()),
            "observation_time_s": now,
        }
        questions = [
            "Where is parcel_1?",
            "How many parcels?",
            "Where is the orange parcel?",
            "Where is parcel_9?",
            "What is on receiver_0?",
        ]
        row["authored_questions"] = [
            {"question": q, "answer": asdict(answer_question(q, scene, now))}
            for q in questions
        ]
        row["stale_control"] = {
            "question": questions[0],
            "queried_time_s": now + 3.0,
            "answer": asdict(answer_question(questions[0], scene, now + 3.0)),
        }
        if frame == "initial" and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            request = decode_request(receipt["proposal"]["raw_text"])
            row["model_question"] = {
                "question": receipt["instruction"],
                "receipt": "../../language/location-query-receipt.json",
                "receipt_sha256": digest(receipt_path.read_bytes()),
                "grounding": ground_request(request, scene, now),
            }
        result["scenes"].append(row)
    if result["scenes"]:
        write_json(destination / "questions.json", result)


def boundary_note(run_id, probe):
    if run_id == "two-parallel-v1":
        return (
            "Development result: coordinator completion was gated on evaluation-ground-truth placement. "
            "Not qualified as sensor-only completion and excluded from the final timing comparison."
        )
    if run_id in SENSOR_COMPLETION_RUNS:
        return (
            "Coordinator unlocks work from controller phase complete with no failure. "
            "Physical placement truth is evaluated separately and does not gate dispatch. "
            "Global fall termination remains evaluation-based. Sensor channels are idealized; "
            "see sensor contract."
        )
    return probe.get(
        "completion_boundary",
        probe.get(
            "completion_contract",
            (
                "Controller completion and independently evaluated placement must be checked separately; "
                "see source snapshot and validation report."
            ),
        ),
    )


def package_run(source: Path, output: Path):
    run_id = source.name
    destination = output / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    # Explicit file allowlist: no accidental model assets, caches, or unrelated output.
    filenames = (
        "probe.json",
        "probe.npz",
        "events.json",
        "source.zip",
        "scene.xml",
        "scene-graph.json",
        "initial-scene-graph.json",
        "final-scene-graph.json",
        "final-perception.json",
        "initial-perception.json",
        "instruction.json",
        "model-receipt.json",
        "interrupted.json",
    )
    for name in filenames:
        if (source / name).exists():
            shutil.copyfile(source / name, destination / name)
    for sensor_directory in sorted(source.glob("r[0-9]*")):
        if sensor_directory.is_dir():
            shutil.copytree(
                sensor_directory,
                destination / sensor_directory.name,
                dirs_exist_ok=True,
            )
    for suffix, prefix in (("-budget", ""), ("-observation-budget", "observation-")):
        budget = source.with_name(run_id + suffix)
        for name in ("resources.json", "affinity-update.json", "stdout.log"):
            if (budget / name).exists():
                shutil.copyfile(budget / name, destination / (prefix + name))
    if not (source / "probe.json").exists():
        reason = json.loads((source / "interrupted.json").read_text())
        with zipfile.ZipFile(source / "source.zip") as archive:
            hashes = {name: digest(archive.read(name)) for name in archive.namelist()}
        return {
            "id": run_id,
            "role": "interrupted_development",
            "status": "interrupted",
            "success": None,
            "reason": reason,
            "source_sha256": hashes,
            "completion_boundary_note": "No completed robot outcome or timing result exists.",
        }
    probe = json.loads((source / "probe.json").read_text())
    raw = (
        gzip.decompress((source / "records.json.gz").read_bytes())
        if (source / "records.json.gz").exists()
        else (source / "records.json").read_bytes()
    )
    records = json.loads(raw)
    (destination / "records.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    traces = {}
    for robot_id in sorted(records[0]["robots"]):
        flat = []
        for record in records:
            robot = record["robots"][robot_id]
            row = {"t": record["t"], "stage": robot["phase"]}
            for key in ("sensor_state", "controller_state", "grip_command", "task_id"):
                if key in robot:
                    row[key] = robot[key]
            flat.append(row)
        trace = robot_id + "-trace.json"
        write_json(destination / trace, flat, compact=True)
        traces[robot_id] = f"runs/{run_id}/{trace}"
    if not (destination / "instruction.json").exists():
        write_json(
            destination / "instruction.json",
            {
                "instruction": probe["instruction"],
                "tree": probe["instruction_tree"],
                "tasks": probe["tasks"],
            },
        )
    report = validate(destination)
    write_json(destination / "validation.json", report)
    package_questions(destination, output)
    source_hashes = probe["source_sha256"]
    action_hashes = {
        name: value
        for name, value in sorted(source_hashes.items())
        if name in ACTION_SOURCES
    }
    return {
        "id": run_id,
        "role": (
            "development_evaluation_coupled"
            if run_id == "two-parallel-v1"
            else "final_sensor_completion"
            if run_id in SENSOR_COMPLETION_RUNS
            else "unclassified_completed"
        ),
        "status": "completed",
        "success": report["success"],
        "robots": probe["robots"],
        "seconds": report["seconds"],
        "instruction": probe["instruction"],
        "per_robot_metrics": report["robots"],
        "source_sha256": source_hashes,
        "action_source_sha256": action_hashes,
        "action_sources_fingerprint": digest(
            json.dumps(action_hashes, sort_keys=True).encode()
        ),
        "completion_boundary_note": boundary_note(run_id, probe),
        "traces": traces,
        "validation": f"runs/{run_id}/validation.json",
        "shared_physics": probe.get("shared_physics", False),
        "dispatch_substrate": probe.get("dispatch_substrate", "not recorded"),
        "sensor_contract": probe.get("sensor_contract"),
    }


def package(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    runs, pending = [], []
    for source in sorted(root.iterdir()):
        if (
            not source.is_dir()
            or source.name.endswith("-budget")
            or "smoke" in source.name
        ):
            continue
        if (source / "probe.json").exists() or (source / "interrupted.json").exists():
            runs.append(package_run(source, output))
        elif (source / "source.zip").exists():
            pending.append(source.name)
    comparisons = []
    by_id = {run["id"]: run for run in runs}
    parallel_id, sequence_id = "model-parallel-v3", "two-sequence-v3"
    if parallel_id in by_id and sequence_id in by_id:
        parallel, sequential = by_id[parallel_id], by_id[sequence_id]
        parallel_pose = np.load(
            output / "runs" / parallel_id / "probe.npz", allow_pickle=False
        )["qpos"][0]
        sequence_pose = np.load(
            output / "runs" / sequence_id / "probe.npz", allow_pickle=False
        )["qpos"][0]
        same_pose = bool(np.array_equal(parallel_pose, sequence_pose))
        same_sources = (
            parallel["action_sources_fingerprint"]
            == sequential["action_sources_fingerprint"]
        )
        admitted = bool(
            same_pose and same_sources and parallel["success"] and sequential["success"]
        )
        comparison = {
            "id": "instruction_conditioned_concurrency",
            "parallel_run": parallel_id,
            "sequential_run": sequence_id,
            "initial_qpos_identical": same_pose,
            "action_sources_identical": same_sources,
            "both_completed_successfully": parallel["success"]
            and sequential["success"],
            "admitted": admitted,
            "scope": "Instruction-conditioned concurrency in independent calibrated workcells; not a manipulation-policy or VLA quality comparison.",
        }
        if admitted:
            comparison.update(
                {
                    "parallel_seconds": parallel["seconds"],
                    "sequential_seconds": sequential["seconds"],
                    "elapsed_reduction_percent": 100
                    * (1 - parallel["seconds"] / sequential["seconds"]),
                    "elapsed_speedup": sequential["seconds"] / parallel["seconds"],
                }
            )
        comparisons.append(comparison)
    result = {
        "schema": "tensorcode.warehouse.evidence.v1",
        "runs": runs,
        "pending_not_packaged": pending,
        "comparisons": comparisons,
        "scope": "Calibrated G1 workcells in one shared physical simulation per run; no handoff, shared-aisle avoidance, open-world vision, or general robotic intelligence claim.",
        "comparison": "Only compare final runs with identical action_sources_fingerprint; development and interrupted runs are retained but excluded.",
    }
    write_json(output / "results.json", result)
    manifest = {
        str(path.relative_to(output / "runs")): digest(path.read_bytes())
        for path in sorted((output / "runs").rglob("*"))
        if path.is_file()
    }
    write_json(output / "run-files.sha256.json", manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/tmp/tensorcode-warehouse")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/robotics/warehouse")
    )
    args = parser.parse_args()
    result = package(args.source, args.output)
    print(
        json.dumps(
            {
                "packaged": [r["id"] for r in result["runs"]],
                "pending": result["pending_not_packaged"],
            }
        )
    )


if __name__ == "__main__":
    main()
