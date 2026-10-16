"""Offline freshness-admission ablation on archived sensor observations.

Advances the query clock, not observation timestamps or measured positions.
This measures admission behavior, not robot success or collision safety.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

from .coordination import RobotState
from .jobs import Capability, JobScheduler, UnsupportedMission
from .language import parse_instruction
from .representation import SceneGraph


def evaluate(scene_path: Path, output: Path):
    raw = scene_path.read_bytes()
    scene = SceneGraph.from_dict(json.loads(raw))
    initial_time = max(e.observed_at for e in scene.entities.values())
    ids = sorted(
        e.id.removeprefix("parcel_")
        for e in scene.entities.values()
        if e.id.startswith("parcel_")
    )
    capabilities = tuple(
        Capability(f"r{i}", (f"parcel_{i}",), (f"receiver_{i}",)) for i in ids
    )
    record_path = scene_path.with_name("records.json.gz")
    recorded_bytes = record_path.read_bytes()
    records = json.loads(gzip.decompress(recorded_bytes))
    observed = min(records, key=lambda row: abs(row["t"] - initial_time))
    robots = [
        RobotState(
            f"r{i}", tuple(observed["robots"][f"r{i}"]["sensor_state"]["root_position"])
        )
        for i in ids
    ]
    instruction = " while ".join(f"move parcel_{i} to receiver_{i}" for i in ids)
    tree = parse_instruction(instruction)
    rows = []
    for delay in (0.0, 3.0):
        for arm, max_age in (
            ("freshness_guard", 2.0),
            ("stale_admission_ablation", 1e12),
        ):
            scheduler = JobScheduler.from_instruction(
                tree, scene, initial_time, capabilities
            )
            assigned = scheduler.dispatch(
                robots, scene, initial_time + delay, max_age=max_age
            )
            stale = sum(
                not scene.fresh(scene.entities[a.object_id], initial_time + delay, 2.0)
                or not scene.fresh(
                    scene.entities[a.destination_id], initial_time + delay, 2.0
                )
                for a in assigned
            )
            rows.append(
                {
                    "arm": arm,
                    "query_delay_s": delay,
                    "assigned_jobs": len(assigned),
                    "stale_jobs_admitted": stale,
                    "unsupported_jobs_admitted": 0,
                    "assignments": [asdict(a) for a in assigned],
                    "events": scheduler.events,
                    "question": scheduler.explain(f"Why is r{ids[0]} waiting?"),
                }
            )
    unsupported = None
    if len(ids) > 1:
        invalid = parse_instruction(f"move parcel_{ids[1]} to receiver_{ids[0]}")
        try:
            JobScheduler.from_instruction(invalid, scene, initial_time, capabilities)
        except UnsupportedMission as exc:
            unsupported = {"rejected": True, "task_ids": list(exc.tasks)}
        else:
            unsupported = {"rejected": False}
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [
        Path(__file__),
        *(
            Path(__file__).with_name(n)
            for n in (
                "coordination.py",
                "jobs.py",
                "language.py",
                "representation.py",
                "coordination-protocol.json",
            )
        ),
    ]
    source_hashes = {}
    with zipfile.ZipFile(
        output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in source_paths:
            data = path.read_bytes()
            info = zipfile.ZipInfo(path.name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            source_hashes[path.name] = hashlib.sha256(data).hexdigest()
    (output / "observed-scene.json").write_bytes(raw)
    result = {
        "scope": "Offline admission on actual recorded sensor scene and observed robot root estimates with controlled query-clock delay; no physical trial or safety claim.",
        "scene_source": str(scene_path),
        "scene_sha256": hashlib.sha256(raw).hexdigest(),
        "observation_time_s": initial_time,
        "robot_observation_time_s": observed["t"],
        "record_source": str(record_path),
        "record_source_sha256": hashlib.sha256(recorded_bytes).hexdigest(),
        "observed_robot_inputs": [asdict(robot) for robot in robots],
        "instruction": instruction,
        "source_sha256": source_hashes,
        "cases": rows,
        "unsupported_plan": unsupported,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path(
            "docs/results/robotics/warehouse/runs/model-parallel-v3/initial-scene-graph.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/robotics/warehouse/coordination-v4"),
    )
    args = parser.parse_args()
    result = evaluate(args.scene, args.output)
    print(
        json.dumps(
            [
                {
                    k: row[k]
                    for k in (
                        "arm",
                        "query_delay_s",
                        "assigned_jobs",
                        "stale_jobs_admitted",
                    )
                }
                for row in result["cases"]
            ]
        )
    )


if __name__ == "__main__":
    main()
