"""Build four explicit lanes from frozen robustness and BC collection protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def prepare(repo, output, robustness, collection):
    code = output / "code"
    if code.exists():
        raise FileExistsError("Use a new campaign directory; snapshots are immutable")
    code.mkdir(parents=True)
    for name in ("examples", "src"):
        shutil.copytree(
            repo / name,
            code / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name, source in [("robustness", robustness), ("collection", collection)]:
        shutil.copytree(source, code / "protocols" / name)
    source_paths = sorted(
        p.relative_to(code).as_posix() for p in code.rglob("*") if p.is_file()
    )
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in source_paths:
            archive.write(code / relative, relative)
    sha = hashlib.sha256((output / "source.zip").read_bytes()).hexdigest()
    robust = json.loads((robustness / "protocol.json").read_text())
    bc = json.loads((collection / "protocol.json").read_text())
    lanes = [[] for _ in range(4)]
    preferred = [
        "payload-80g-compensated",
        "teacher-pickup-x10mm",
        "teacher-pickup-y10mm",
        "teacher-validation-offset",
    ]
    pairs = {}
    cases = sorted(
        robust["cases"],
        key=lambda c: preferred.index(c["id"]) if c["id"] in preferred else 100,
    )
    for case in cases:
        key = case.get("pair") or case["id"]
        lane = pairs.setdefault(key, len(pairs) % 4)
        lanes[lane].append((case, "robustness", "/tmp/tensorcode-warehouse-v4/runs"))
    # Supplemental collection goes immediately after each lane's initial teacher.
    for index, case in reversed(list(enumerate(bc["cases"]))):
        lanes[index % 4].insert(
            1, (case, "collection", "/tmp/tensorcode-warehouse-bc/runs")
        )
    for index, jobs in enumerate(lanes):
        host = "host" if index < 2 else "peer"
        cwd = str(code) if host == "host" else "/tmp/tensorcode-campaign/code"
        python = (
            str(repo / ".venv/bin/python")
            if host == "host"
            else "/tmp/tensorcode-humanoid-peer/venv/bin/python"
        )
        queue = {
            "cwd": cwd,
            "cpus": list(range((index % 2) * 4, (index % 2) * 4 + 4)),
            "memory_gib": 12,
            "deadline_utc": "2026-10-16T23:54:00+00:00",
            "jobs": [],
        }
        for case, protocol, destination in jobs:
            queue["jobs"].append(
                {
                    "id": case["id"],
                    "backend": "egl",
                    "command": [
                        python,
                        "examples/robotics/warehouse/physics.py",
                        "--output",
                        destination + "/" + case["id"],
                        "--robots",
                        "2",
                        "--seconds",
                        "85",
                        "--scenario-json",
                        "protocols/" + protocol + "/" + case["id"] + ".json",
                        "--bc-record",
                    ],
                    "env": {"PYTHONPATH": cwd + "/src"},
                    "timeout_s": 900,
                    "sources": source_paths,
                    "protocol": {
                        "case": case,
                        "family": protocol,
                        "source_zip_sha256": sha,
                        "assigned_host": host,
                        "baseline_commit": "d023685",
                    },
                }
            )
        (output / f"{host}-lane{index % 2}.json").write_text(
            json.dumps(queue, indent=2) + "\n"
        )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_zip_sha256": sha,
                "lanes": [[x[0]["id"] for x in lane] for lane in lanes],
                "pair_assignment": pairs,
                "baseline_commit": "d023685",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.repo.resolve(),
        args.output.resolve(),
        args.robustness.resolve(),
        args.collection.resolve(),
    )


if __name__ == "__main__":
    main()
