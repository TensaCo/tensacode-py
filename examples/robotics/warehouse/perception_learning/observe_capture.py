"""Produce fresh final scene evidence from a captured multi-view sensor dataset."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from warehouse.perception_learning.scene_observer import observe_views


def observe(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    final = max(row["frame_index"] for row in manifest["frames"])
    lanes = {}
    for row in manifest["frames"]:
        if row["frame_index"] == final:
            lanes.setdefault(row["lane"], []).append(row)
    result = {
        "source_run": manifest["run"],
        "frame_index": final,
        "lanes": {},
        "source": "fresh_simultaneous_rgbd_geometry_not_evaluation_truth",
    }
    for lane, entries in sorted(lanes.items()):
        frames = []
        for entry in sorted(entries, key=lambda x: x["view"]):
            path = dataset / entry["path"]
            contract = json.loads((path / "sensor-contract.json").read_text())
            frames.append(
                {
                    "rgb": np.array(Image.open(path / "sensor-rgb.png")),
                    "depth": np.load(path / "sensor-depth.npz")["depth"],
                    "calibration": contract["camera"],
                    "timestamp": entry["timestamp"],
                }
            )
        calibration = frames[0]["calibration"]
        center = np.array(calibration["position"]) + 2.5 * np.array(
            calibration["forward"]
        )
        result["lanes"][str(lane)] = observe_views(
            frames, entries[0]["timestamp"], workcell_center_xy=center[:2]
        )
    paths = [
        Path(__file__),
        Path(__file__).with_name("scene_observer.py"),
        Path(__file__).with_name("robust_geometry.py"),
        Path(__file__).parents[1] / "perception_tracking.py",
        Path(__file__).parents[1] / "perception.py",
    ]
    result["sources_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(observe(args.dataset), indent=2) + "\n")
