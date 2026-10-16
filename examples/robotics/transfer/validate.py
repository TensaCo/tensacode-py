"""Independently recompute published transfer measurements from saved traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def validate(directory: Path):
    rows = []
    contexts = set()
    all_without_demonstration = True
    for run in sorted((directory / "runs").iterdir()):
        if not run.is_dir():
            continue
        probe = json.loads((run / "probe.json").read_text())
        all_without_demonstration = all_without_demonstration and probe.get("mode") in {
            "planned",
            "expert",
        }
        if probe.get("mode") in {"fixed", "context"}:
            assert probe.get("source_context_sha256"), (
                run.name,
                "Missing demonstration hash",
            )
        snapshot_verified = False
        if (run / "source.zip").exists():
            hashes = probe.get("source_sha256")
            assert hashes, (run.name, "Source archive lacks declared hashes")
            with zipfile.ZipFile(run / "source.zip") as archive:
                for name, expected in hashes.items():
                    assert hashlib.sha256(archive.read(name)).hexdigest() == expected, (
                        run.name,
                        name,
                        "Source archive hash mismatch",
                    )
            snapshot_verified = True
        compressed = run / "records.json.gz"
        payload = (
            gzip.decompress(compressed.read_bytes())
            if compressed.exists()
            else (run / "records.json").read_bytes()
        )
        records = json.loads(payload)
        poses = np.load(run / "probe.npz")["qpos"]
        assert len(records) == len(poses) == probe["steps"], run.name
        base = np.array([r["base"] for r in records])
        objects = np.array([r["object"] for r in records])
        np.testing.assert_allclose(poses[:, :3], base, atol=1e-12)
        object_pose_lag = float(
            np.max(np.linalg.norm(poses[:, 38:41] - objects, axis=1))
        )
        times = np.array([r["t"] for r in records])
        np.testing.assert_allclose(np.diff(times), 0.02, atol=1e-10)
        bilateral = np.array(
            [all(p in r["contacts"] for p in ("pad_a", "pad_b")) for r in records]
        )
        shift = probe.get("height_shift", 0)
        lift = max(0.0, float(np.max(objects[:, 2]) - (0.805 + shift)))
        contact = float(bilateral.sum() * 0.02)
        held = float((bilateral & (objects[:, 2] > 0.905 + shift)).sum() * 0.02)
        for key, value in [
            ("max_lift", lift),
            ("bilateral_contact_seconds", contact),
            ("held_above_10cm_seconds", held),
        ]:
            assert abs(probe[key] - value) < 1e-9, (run.name, key)
        settled = next((r for r in records if r["stage"] == "settle"), None)
        if settled is not None:
            np.testing.assert_allclose(
                probe["walking_distance_m"],
                np.linalg.norm(settled["base"][:2]),
                atol=1e-12,
            )
        success = (
            probe["min_height"] > 0.65
            and probe["walking_distance_m"] > 0.7
            and held >= 1
            and probe["stage"] == "release"
        )
        assert success == probe["success"], run.name
        if probe.get("source_context_sha256"):
            contexts.add(probe["source_context_sha256"])
        audit = json.loads((run / "sensor-audit.json").read_text())
        assert audit["records_sha256"] == hashlib.sha256(payload).hexdigest(), run.name
        rows.append(
            {
                "run": run.name,
                "success": success,
                "max_lift_m": lift,
                "held_seconds": held,
                "trace_samples": len(records),
                "source_snapshot_verified": snapshot_verified,
                "max_object_transform_integration_lag_m": object_pose_lag,
            }
        )
    assert rows, "No runs found"
    assert len(contexts) == 1 or (not contexts and all_without_demonstration), (
        "Trajectory evaluation must use one frozen demonstration; only authored planned/expert "
        "bundles may omit demonstration context"
    )
    stored_contexts = {
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (directory / "runs").glob("*/context.json")
    }
    assert contexts <= stored_contexts, (
        "Declared context must match saved demonstration"
    )
    return {
        "validated": True,
        "runs": rows,
        "context_sha256": next(iter(contexts)) if contexts else None,
        "scope": "Recomputed sampled lift/contact, walking distance and success; checked root trace/qpos agreement, object transform integration lag, sensor-audit trace hashes, execution source archive hashes when present, and saved context hash. Does not independently certify every 2ms minimum-height sample or sensor boundary.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    (args.directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
