"""Package independently validated, allowlisted campaign evidence without overwrites.

Run as a module with --validator v4|jobs --output DIR RUN_DIR [RUN_DIR ...].
Source snapshots belong to the original recordings, not the current checkout.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compress(source, destination):
    raw = source.read_bytes()
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    if gzip.decompress(packed) != raw:
        raise ValueError("Compression mismatch")
    destination.write_bytes(packed)
    return {"source": source.name, "raw_sha256": digest(raw), "raw_bytes": len(raw)}


def graph_projections(graph, initial_time):
    """Project recorded observations only; never infer simulator labels."""
    initial = {}
    revision = 0
    for entry in graph.get("history", []):
        observation = entry.get("observation")
        if observation and observation["observed_at"] <= initial_time + 1e-9:
            initial[observation["id"]] = observation
            revision = max(revision, entry["revision"])
    return {
        "initial": {
            "revision": revision,
            "entities": list(initial.values()),
            "relations": [
                r
                for r in graph.get("relations", [])
                if r["observed_at"] <= initial_time + 1e-9
                and r["subject"] in initial
                and r["object"] in initial
            ],
            "history": [],
        },
        "final": {**graph, "history": []},
    }


def stage_run(source, destination, validator, resources=None):
    if validator == "v4":
        from .validate_v4 import validate

        validation_name = "validation-v4.json"
    elif validator == "jobs":
        from .validate_jobs import validate

        validation_name = "validation-jobs.json"
    else:
        raise ValueError("Choose an explicit independent validator")
    original = validate(source)
    destination.mkdir()
    for name in (
        "probe.json",
        "probe.npz",
        "events.json",
        "source.zip",
        "scene.xml",
        "scenario.json",
        "recovery-protocol.json",
        "scheduler-final.json",
        "scheduler-waiting.json",
        "scheduler.json",
        "instruction.json",
        "model-receipt.json",
        "initial-scene-graph.json",
        "final-scene-graph.json",
        "initial-perception.json",
        "final-perception.json",
    ):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    if (source / "mission").is_dir():
        (destination / "mission").mkdir()
        for name in ("instruction.json", "proposal.json", "protocol.json"):
            shutil.copyfile(source / "mission" / name, destination / "mission" / name)
    sensor_names = {
        "sensor-contract.json",
        "sensor-rgb.png",
        "sensor-depth.npz",
        "sensor-first-frame.png",
        "sensor-evidence.npz",
    }
    for directory in sorted(source.glob("r[0-9]*")):
        if not directory.is_dir():
            continue
        for file in sorted(directory.rglob("*")):
            if file.is_file() and file.name in sensor_names:
                target = destination / file.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, target)
    if resources is not None:
        for name in ("receipt.json", "resources.jsonl", "resources.json", "stdout.log"):
            file = Path(resources) / name
            if file.is_file():
                target = destination / "resources" / name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(file, target)
    compression = {}
    if (source / "records.json").exists():
        compression["records.json.gz"] = compress(
            source / "records.json", destination / "records.json.gz"
        )
    else:
        raw = gzip.decompress((source / "records.json.gz").read_bytes())
        shutil.copyfile(source / "records.json.gz", destination / "records.json.gz")
        compression["records.json.gz"] = {
            "source": "records.json.gz",
            "raw_sha256": digest(raw),
            "raw_bytes": len(raw),
        }
    graph_path = source / "scene-graph.json"
    projections = {}
    if graph_path.exists():
        compression["scene-graph.json.gz"] = compress(
            graph_path, destination / "scene-graph.json.gz"
        )
        initial_time = json.loads(
            gzip.decompress((destination / "records.json.gz").read_bytes())
        )[0]["t"]
        for frame, graph in graph_projections(
            json.loads(graph_path.read_text()), initial_time
        ).items():
            name = f"{frame}-scene-graph.json"
            if not (destination / name).exists():
                write_json(destination / name, graph)
                projections[name] = {
                    "source": "scene-graph.json.gz",
                    "history_omitted": True,
                    "method": "Observations at or before first control timestamp"
                    if frame == "initial"
                    else "Final recorded entities and relations",
                    "first_control_time_s": initial_time,
                }
    verified = validate(destination)
    if original != verified:
        raise ValueError(f"Validation changed while packaging {source.name}")
    if (
        digest(gzip.decompress((destination / "records.json.gz").read_bytes()))
        != verified["records_sha256"]
    ):
        raise ValueError("Validator and decompressed record hashes disagree")
    write_json(destination / validation_name, verified)
    manifest = {
        "schema": "tensorcode.warehouse.campaign-evidence.v1",
        "run": source.name,
        "validator": validator,
        "scope": "Lossless original run evidence; graph projections explicitly labeled. Source snapshot is the recording's source, not current checkout.",
        "compression": compression,
        "derived_graph_projections": projections,
        "files": {
            str(p.relative_to(destination)): {
                "sha256": digest(p.read_bytes()),
                "bytes": p.stat().st_size,
            }
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
    }
    write_json(destination / "manifest.json", manifest)
    return {
        "run": source.name,
        "validation": validation_name,
        "physical_success": verified.get("physical_success"),
        "strict_current_settle_success": verified.get("strict_current_settle_success"),
        "bytes": sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()),
        "manifest_sha256": digest((destination / "manifest.json").read_bytes()),
    }


def package_campaign(
    sources, output, *, validator="v4", max_bytes=100_000_000, resources=None
):
    if not sources or len({p.name for p in sources}) != len(sources):
        raise ValueError("Supply distinct, nonempty run directories")
    if any((output / p.name).exists() for p in sources):
        raise FileExistsError("Refusing to overwrite published evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".campaign-", dir=output.parent) as temp:
        staging = Path(temp)
        rows = [
            stage_run(
                source,
                staging / source.name,
                validator,
                resources=resources[source.name],
            )
            if resources and source.name in resources
            else stage_run(source, staging / source.name, validator)
            for source in sources
        ]
        total = sum(row["bytes"] for row in rows)
        if total > max_bytes:
            raise ValueError(
                f"Campaign is {total} bytes, above {max_bytes} budget; nothing published"
            )
        output.mkdir(exist_ok=True)
        for source in sources:
            (staging / source.name).rename(output / source.name)
    return {"runs": rows, "bytes": total, "validator": validator}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validator", choices=("v4", "jobs"), required=True)
    parser.add_argument("--max-bytes", type=int, default=100_000_000)
    parser.add_argument(
        "--resources-json",
        type=Path,
        help="JSON mapping run names to resource-log directories",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            package_campaign(
                args.sources,
                args.output,
                validator=args.validator,
                max_bytes=args.max_bytes,
                resources=json.loads(args.resources_json.read_text())
                if args.resources_json
                else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
