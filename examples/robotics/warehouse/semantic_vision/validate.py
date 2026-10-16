"""Recheck retained observations, source snapshots, counts and partial recall."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .study import digest, score_partial_labels


def validate(root):
    verified = []
    for result_path in sorted(root.glob("*/results.json")):
        run = result_path.parent
        result = json.loads(result_path.read_text())
        rows = json.loads((run / "predictions.json").read_text())
        assert result["protocol_sha256"] == digest(run / "protocol.json")
        assert result["query_evaluations"] == len(rows)
        assert result["frames"] == len({row["frame"] for row in rows})
        assert result["runtime"]["device"] == "cpu"
        for row in rows:
            frame = run / row["frame"]
            for filename, key in [
                ("sensor-rgb.png", "rgb_sha256"),
                ("sensor-depth.npz", "depth_sha256"),
                ("sensor-contract.json", "calibration_sha256"),
            ]:
                assert digest(frame / filename) == row[key]
        for query in result["per_query"]:
            selected = [row for row in rows if row["query"] == query["query"]]
            assert query["detections"] == sum(
                len(row["detections"]) for row in selected
            )
            assert query["frames_with_detections"] == sum(
                bool(row["detections"]) for row in selected
            )
        for status, key in [
            ("surface_sample", "surface_samples"),
            ("abstained", "depth_abstentions"),
        ]:
            assert result[key] == sum(
                d["depth_evidence"]["status"] == status
                for row in rows
                for d in row["detections"]
            )
        annotations = run / "evaluation-amodal-boxes.json"
        if annotations.exists():
            assert result["partial_label_scoring"] == score_partial_labels(
                rows, json.loads(annotations.read_text())
            )
        snapshot = run / "source.zip"
        if snapshot.exists():
            with zipfile.ZipFile(snapshot) as archive:
                for name, expected in result["source_sha256"].items():
                    assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        if result["split"] == "evaluation":
            assert snapshot.exists(), "Evaluation needs exact source snapshot"
            assert result["runtime"]["cpu_affinity"] == [12, 13, 14, 15]
        verified.append(
            {
                "run": run.name,
                "prediction_rows": len(rows),
                "source_snapshot": snapshot.exists(),
                "checks": "input hashes, source snapshot when available, query counts, depth status counts, independent partial recall",
            }
        )
    if not verified:
        raise ValueError("No complete study runs")
    output = {
        "verified_runs": verified,
        "scope": "Artifact consistency, not verification of neural semantics or physical pose accuracy",
    }
    (root / "validation.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(validate(parser.parse_args().root), indent=2))
