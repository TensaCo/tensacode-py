"""Annotated measured views; markers are fused estimates, never simulator poses."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot(dataset, output):
    manifest = json.loads((dataset / "manifest.json").read_text())
    composition = json.loads((dataset / "composition.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
    for row, frame in enumerate(composition["frames"]):
        entries = sorted(
            [
                e
                for e in manifest["frames"]
                if e["frame_index"] == frame["frame_index"] and e["lane"] == 0
            ],
            key=lambda e: e["view"],
        )
        for col, entry in enumerate(entries):
            path = dataset / entry["path"]
            cal = json.loads((path / "sensor-contract.json").read_text())["camera"]
            ax = axes[row, col]
            ax.imshow(Image.open(path / "sensor-rgb.png"))
            ax.axis("off")
            for entity in frame["scene_graph"]["entities"]:
                if entity["kind"] == "workcell" or entity["confidence"] < 0.7:
                    continue
                relative = np.array(entity["position"]) - cal["position"]
                z = relative @ np.array(cal["forward"])
                if z <= 0:
                    continue
                x = (
                    cal["width"] / 2
                    + cal["focal_pixels"] * (relative @ np.array(cal["right"])) / z
                )
                y = (
                    cal["height"] / 2
                    - cal["focal_pixels"] * (relative @ np.array(cal["up"])) / z
                )
                if not (0 <= x < cal["width"] and 0 <= y < cal["height"]):
                    continue
                color = entity["attributes"]["color"]
                color = "#49a0ff" if color == "blue" else color
                ax.plot(x, y, "+", color=color, markersize=9, markeredgewidth=2)
                ax.annotate(
                    entity["id"],
                    (x, y),
                    xytext=(7, -8),
                    textcoords="offset points",
                    fontsize=7,
                    color="white",
                    bbox={
                        "facecolor": "#18202b",
                        "alpha": 0.8,
                        "pad": 1,
                        "edgecolor": "none",
                    },
                )
            ax.set_title(
                f"{'Initial' if row == 0 else 'Terminal'} · t={frame['timestamp']:.2f}s · view{col + 1}",
                fontsize=11,
            )
    fig.suptitle(
        "Fresh RGB-D scene evidence from a failed 160 s robot trial\nMarkers: fused measured positions · both parcels remain near their pickup regions",
        fontsize=15,
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.dataset, args.output)
