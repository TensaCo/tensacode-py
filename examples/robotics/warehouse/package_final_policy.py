"""Publish the complete frozen policy matrix with lossless, deduplicated evidence.

No final inputs are changed. Publication requires the independent final audit to
pass both before and after packaging, including every failed physical outcome.
Weights and identical runtime source archives use relative symlinks. Check out
Git symlinks normally; raw GitHub symlink URLs are not direct model downloads.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
import zlib
from pathlib import Path

from .package_campaign import digest, stage_run, write_json

MAX_FILE_BYTES = 100_000_000


def bind_candidates(manifest, portable_root):
    """Locate frozen artifacts by content, never by an inferred replacement model."""
    expected = {
        item["sha256"]
        for candidate in manifest["candidates"]
        for item in candidate["files"].values()
    }
    found = {}
    filenames = {
        name for candidate in manifest["candidates"] for name in candidate["files"]
    }
    for path in sorted(portable_root.rglob("*")):
        if path.is_file() and path.name in filenames:
            sha = digest(path.read_bytes())
            if sha in expected:
                found.setdefault(sha, path.resolve())
    if expected - found.keys():
        raise ValueError("Frozen portable model artifacts are missing or changed")
    return {
        candidate["id"]: {
            name: found[item["sha256"]] for name, item in candidate["files"].items()
        }
        for candidate in manifest["candidates"]
    }


def relative_link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent))


def evidence_manifest(directory, *, allow_invalid=False):
    files = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path == directory / "manifest.json":
            continue
        if path.stat().st_size >= MAX_FILE_BYTES:
            raise ValueError(f"Artifact exceeds the individual file limit: {path}")
        row = {"sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
        if path.is_symlink():
            row["symlink_target"] = os.readlink(path)
        if path.name.endswith(".json.gz"):
            try:
                row["raw_sha256"] = digest(gzip.decompress(path.read_bytes()))
            except (OSError, EOFError, zlib.error) as exc:
                if not allow_invalid:
                    raise
                row["decompression_error"] = str(exc)
        files[str(path.relative_to(directory))] = row
    return files


def stage_partial(source, destination, models, resources=None):
    """Preserve available bytes without repairing invalid or unfinished runs."""
    destination.mkdir()
    names = {
        "probe.json",
        "probe.npz",
        "events.json",
        "source.zip",
        "scene.xml",
        "scenario.json",
        "instruction.json",
        "model-receipt.json",
        "initial-scene-graph.json",
        "final-scene-graph.json",
        "initial-perception.json",
        "final-perception.json",
        "recovery-protocol.json",
        "scheduler-final.json",
        "scheduler-waiting.json",
        "scheduler.json",
        "validation-v4.json",
        "validation-jobs.json",
        "interrupted.json",
    }
    for name in sorted(names):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    for name in ("records.json", "scene-graph.json"):
        if (source / name).is_file():
            (destination / (name + ".gz")).write_bytes(
                gzip.compress((source / name).read_bytes(), mtime=0)
            )
        elif (source / (name + ".gz")).is_file():
            shutil.copyfile(source / (name + ".gz"), destination / (name + ".gz"))
    sensor_names = {
        "sensor-contract.json",
        "sensor-rgb.png",
        "sensor-depth.npz",
        "sensor-first-frame.png",
        "sensor-evidence.npz",
        "bc-episode.json.gz",
    }
    for directory in source.glob("r[0-9]*"):
        if directory.is_dir():
            for file in directory.rglob("*"):
                if file.is_file() and file.name in sensor_names:
                    target = destination / file.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, target)
    for directory, allowed in (
        ("mission", {"instruction.json", "proposal.json", "protocol.json"}),
        ("learned-policy", set(models)),
    ):
        for name in allowed:
            original = source / directory / name
            if not original.is_file():
                continue
            target = destination / directory / name
            target.parent.mkdir(exist_ok=True)
            if directory == "learned-policy" and digest(
                original.read_bytes()
            ) == digest(models[name].read_bytes()):
                relative_link(models[name], target)
            else:
                shutil.copyfile(original, target)
    if resources:
        for name in ("receipt.json", "resources.json", "resources.jsonl", "stdout.log"):
            file = Path(resources) / name
            if file.is_file():
                target = destination / "resources" / name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(file, target)


def comparison_rows(audit):
    return [
        {k: value for k, value in row.items() if k != "path"} for row in audit["runs"]
    ]


def package(
    plan, raw_runs, output, portable_root, resources=None, *, allow_partial=False
):
    from .final_policy_audit import audit

    if output.exists():
        raise FileExistsError("Refusing to overwrite final evidence")
    manifest_path = plan / "execution-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    models = bind_candidates(manifest, portable_root)
    frozen = {
        str(path.relative_to(plan)): digest(path.read_bytes())
        for path in plan.rglob("*")
        if path.is_file()
    }
    original = audit(manifest_path, raw_runs, plan / "source.zip", plan / "cases")
    if not original["complete_and_matched"] and not allow_partial:
        raise ValueError(
            "Final matrix is incomplete, invalid or unmatched; nothing published"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".final-evidence-", dir=output.parent
    ) as temp:
        staging = Path(temp)
        for row in original["runs"]:
            key = row["policy"] + "/" + row["case"]
            source = raw_runs / key
            destination = staging / "runs" / key
            if allow_partial:
                if source.is_dir():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    stage_partial(
                        source,
                        destination,
                        models.get(row["policy"], {}),
                        (resources or {}).get(key),
                    )
                    write_json(
                        destination / "manifest.json",
                        {
                            "schema": "tensorcode.warehouse.partial-run-evidence.v1",
                            "raw_audit_status": row["status"],
                            "raw_audit_error": row.get("error"),
                            "scope": "Available artifact snapshot; no missing/corrupt evidence repaired. Not a qualified comparison.",
                            "files": evidence_manifest(destination, allow_invalid=True),
                        },
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage_run(source, destination, "v4", resources=(resources or {}).get(key))
            for robot in sorted(row["robots"]):
                episode = source / robot / "bc-episode.json.gz"
                target = destination / robot / episode.name
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(episode, target)
                if digest(episode.read_bytes()) != digest(target.read_bytes()):
                    raise ValueError("Sensor/action episode copy mismatch")
            snapshot = destination / "source.zip"
            shared = staging / "shared" / (digest(snapshot.read_bytes()) + ".zip")
            shared.parent.mkdir(exist_ok=True)
            if not shared.exists():
                shutil.copyfile(snapshot, shared)
            snapshot.unlink()
            relative_link(shared, snapshot)
            for name, target in models.get(row["policy"], {}).items():
                relative_link(target, destination / "learned-policy" / name)
            write_json(
                destination / "manifest.json",
                {
                    **json.loads((destination / "manifest.json").read_text()),
                    "schema": "tensorcode.warehouse.final-run-evidence.v1",
                    "policy": row["policy"],
                    "case": row["case"],
                    "files": evidence_manifest(destination),
                },
            )
        packaged = audit(
            manifest_path, staging / "runs", plan / "source.zip", plan / "cases"
        )
        if not allow_partial and (
            not packaged["complete_and_matched"]
            or comparison_rows(packaged) != comparison_rows(original)
        ):
            raise ValueError(
                "Packaged final audit differs from raw evidence; nothing published"
            )
        for row in packaged["runs"]:
            row["path"] = "runs/" + row["policy"] + "/" + row["case"]
        packaged["publication_mode"] = "partial" if allow_partial else "complete"
        packaged["comparative_claims_qualified"] = (
            not allow_partial and packaged["complete_and_matched"]
        )
        if allow_partial:
            packaged["publication_warning"] = (
                "INCOMPLETE/UNQUALIFIED SNAPSHOT: no comparative claim. All declared statuses retained; missing/invalid runs are not physical failures or successes."
            )
            for row in original["runs"]:
                row["path"] = "runs/" + row["policy"] + "/" + row["case"]
            write_json(staging / "raw-audit-at-copy-start.json", original)
        write_json(staging / "audit.json", packaged)
        write_json(
            staging / "bindings.json",
            {
                "schema": "tensorcode.warehouse.final-evidence-bindings.v1",
                "plan": os.path.relpath(plan.resolve(), output.resolve()),
                "frozen_plan_files": frozen,
                "portable_models": {
                    policy: {
                        name: {
                            "path": os.path.relpath(path, output.resolve()),
                            "sha256": digest(path.read_bytes()),
                        }
                        for name, path in files.items()
                    }
                    for policy, files in models.items()
                },
                "scope": "Original final evidence; model files and runtime sources use hash-verified relative symlinks. No retraining or changed inputs.",
            },
        )
        (staging / "README.md").write_text("""# Frozen final policy comparison evidence

`audit.json` includes every declared policy/case cell, including failed physical
outcomes. The protocol and execution manifest remain unchanged in `../final-plan`.
Conditions were excluded from candidate training but had previously been exercised
by the original authored baseline; this is not a completely unseen benchmark.
Co-simulated robot placements are correlated.

Every run retains full clocks, sensor/action episodes, compressed original records,
qpos, source bindings and physical validation. The independent audit verifies the
common actuator slew projection against every recorded action and first encoders.
The retained evaluator measurements are not an independent dynamics reconstruction.

Model artifacts link to the existing frozen portable bundles, and identical runtime
source archives are stored once under `shared/`. Use a normal Git checkout with
symlink support. Raw GitHub symlink URLs contain link text, not downloadable weights.
`bindings.json` and each run manifest record the paths and SHA-256 hashes.
""")
        if allow_partial:
            (
                staging / "README.md"
            ).write_text("""# INCOMPLETE / UNQUALIFIED final evidence snapshot

This publication uses `--allow-partial`. It is not a complete, matched policy
comparison, and supports no comparative claim. `audit.json` retains every declared
cell, including unrun, incomplete and invalid statuses and their audit errors.
Missing or invalid runs are not counted as physical successes or failures.

Available records, sensor/actions, qpos, sources and receipts are preserved without
repairing missing or corrupt artifacts. Files may be partial snapshots of active
runs. `raw-audit-at-copy-start.json` records status before copying; `audit.json`
independently audits the packaged snapshot. The frozen plan remains unchanged.

Matching model files use hash-verified relative links to frozen portable bundles;
changed model bytes are retained as invalid evidence. Check out Git symlinks normally:
raw GitHub symlink URLs are link text, not model downloads.
""")
        write_json(
            staging / "manifest.json",
            {
                "schema": "tensorcode.warehouse.final-bundle.v1",
                "publication_mode": "partial" if allow_partial else "complete",
                "files": evidence_manifest(staging, allow_invalid=allow_partial),
            },
        )
        current = {
            str(path.relative_to(plan)): digest(path.read_bytes())
            for path in plan.rglob("*")
            if path.is_file()
        }
        if frozen != current:
            raise ValueError("Frozen plan changed during packaging")
        staging.rename(output)
    return {
        "runs": len(original["runs"]),
        "manifest_sha256": digest((output / "manifest.json").read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--portable-models", type=Path, required=True)
    parser.add_argument("--resources-json", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Publish all statuses and available evidence as an explicitly unqualified snapshot",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            package(
                args.plan,
                args.runs,
                args.output,
                args.portable_models,
                json.loads(args.resources_json.read_text())
                if args.resources_json
                else None,
                allow_partial=args.allow_partial,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
