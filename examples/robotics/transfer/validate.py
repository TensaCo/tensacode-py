"""Independently recompute published transfer measurements from saved traces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def placement_metrics(records: list[dict]) -> dict:
    """Score released, settled placement from evaluation columns, never actions.

    The 20 cm support and 5 cm cube permit 6.5 cm center offset with 1 cm
    edge margin. The hand must be clear continuously for at least one second.
    """
    longest = consecutive = 0
    final_xy = final_clearance = None
    for row in records:
        obj = np.asarray(row["object"], dtype=float)
        surface = np.asarray(row["receiving_surface_xyz"], dtype=float)
        hand = np.asarray(row["hand"], dtype=float)
        assert obj.shape == surface.shape == hand.shape == (3,)
        assert np.isfinite(np.r_[obj, surface, hand]).all()
        xy = obj[:2] - surface[:2]
        distance = float(np.linalg.norm(hand - obj))
        if "hand_object_distance_m" in row:
            assert abs(distance - row["hand_object_distance_m"]) < 1e-9
        contacts = set(row["contacts"])
        receiving_contact = "receiving_marker" in contacts
        assert receiving_contact == row["receiving_contact"]
        speed = row["object_speed_m_s"]
        assert np.isfinite(speed) and speed >= 0
        phase = (row.get("controller_state") or {}).get("phase")
        valid = (
            phase in {"withdraw", "complete"}
            and receiving_contact
            and np.max(np.abs(xy)) <= 0.065
            and abs(obj[2] - (surface[2] + 0.025)) <= 0.01
            and speed < 0.03
            and not contacts.intersection({"pad_a", "pad_b"})
            and row["grip_command"] <= 1e-5
            and distance > 0.10
        )
        consecutive = consecutive + 1 if valid else 0
        longest = max(longest, consecutive)
        final_xy = float(np.linalg.norm(xy))
        final_clearance = distance
    return {
        "placement_settled_seconds": longest * 0.02,
        "placement_xy_error_m": final_xy,
        "hand_clearance_m": final_clearance,
    }


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
            "handling",
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
        placement = {}
        if probe.get("mode") == "handling":
            placement = placement_metrics(records)
            for key, actual in placement.items():
                assert abs(probe[key] - actual) < 1e-9, (run.name, key)
            success = (
                probe["min_height"] > 0.65
                and probe["walking_distance_m"] > 0.7
                and held >= 1
                and records[-1]["controller_state"]["phase"] == "complete"
                and not probe["controller_failed"]
                and placement["placement_settled_seconds"] >= 1
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
                **placement,
                "max_lift_m": lift,
                "held_seconds": held,
                "trace_samples": len(records),
                "source_snapshot_verified": snapshot_verified,
                "max_object_transform_integration_lag_m": object_pose_lag,
            }
        )
    assert rows, "No runs found"
    assert len(contexts) == 1 or (not contexts and all_without_demonstration), (
        "Trajectory evaluation must use one frozen demonstration; only authored planned/expert/handling "
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
        "scope": "Recomputed sampled lift/contact, walking distance and success, plus released on-surface placement and hand clearance for handling runs; checked root trace/qpos agreement, object transform integration lag, sensor-audit trace hashes, execution source archive hashes when present, and saved context hash. Does not independently certify every 2ms minimum-height sample or sensor boundary.",
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
