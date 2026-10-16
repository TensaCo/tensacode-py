"""CPU preparation of sensor-only geometric features, bound to raw episode hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contract import input_statistics as base_statistics
from .data import load_episode, sha256
from .geometry import geometry_features, residual_targets


def input_statistics(x):
    mean, scale = x.mean(0), np.maximum(x.std(0), 0.01)
    mean[:103], scale[:103] = base_statistics(x[:, :103])
    # Rotation-matrix entries are explicitly bounded dimensionless coordinates.
    mean[112:], scale[112:] = 0, 1
    return mean, scale


def prepare(manifest_path, calibration_path, output):
    manifest_path, calibration_path, output = map(
        Path, (manifest_path, calibration_path, output)
    )
    manifest = json.loads(manifest_path.read_text())
    if manifest["test"]:
        raise ValueError(
            "Geometry training preparation must not open final observations"
        )
    calibration = json.loads(calibration_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    index = {
        "manifest_sha256": sha256(manifest_path),
        "kinematics_sha256": sha256(calibration_path),
        "splits": {},
    }
    for split in ("train", "validation", "test"):
        index["splits"][split] = []
        for i, entry in enumerate(manifest[split]):
            source = (manifest_path.parent / entry["path"]).resolve()
            if sha256(source) != entry["sha256"]:
                raise ValueError("Raw episode hash mismatch")
            x, y, document = load_episode(source)
            observations = [r["observation"] for r in document["steps"]]
            x = np.concatenate(
                [x, geometry_features(calibration, observations)], axis=1
            )
            y = residual_targets(y, observations)
            destination = output / f"{split}-{i:03d}.npz"
            np.savez_compressed(destination, x=x, y=y)
            index["splits"][split].append(
                {
                    "file": destination.name,
                    "sha256": sha256(destination),
                    "raw_episode": entry,
                }
            )
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def load_prepared(manifest_path, calibration_path, directory):
    manifest_path, calibration_path, directory = map(
        Path, (manifest_path, calibration_path, directory)
    )
    manifest = json.loads(manifest_path.read_text())
    index = json.loads((directory / "index.json").read_text())
    if index["manifest_sha256"] != sha256(manifest_path) or index[
        "kinematics_sha256"
    ] != sha256(calibration_path):
        raise ValueError("Prepared data provenance mismatch")
    splits = {}
    for split in ("train", "validation", "test"):
        if [e["raw_episode"] for e in index["splits"][split]] != manifest[split]:
            raise ValueError("Prepared split differs from original manifest")
        splits[split] = []
        for entry in index["splits"][split]:
            path = directory / entry["file"]
            if sha256(path) != entry["sha256"]:
                raise ValueError("Prepared feature hash mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                splits[split].append(
                    (arrays["x"], arrays["y"], {}, entry["raw_episode"])
                )
    return manifest, splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.calibration, args.output)
