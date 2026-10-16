"""Static evidence chart from recorded comparison metrics, with coverage shown."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(report, output):
    summary = json.loads(report.read_text())["summary"]
    names = ["single_inverse", "multiview_mean_inverse", "multiview_fusion"]
    labels = ["One-view inverse", "Three-view mean", "Three-view fusion"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for row, kind in enumerate(["parcel", "receiver"]):
        for col, condition in enumerate(["clean", "noise_occlusion"]):
            ax = axes[row, col]
            records = [
                next(
                    r
                    for r in summary
                    if r["split"] == "evaluation"
                    and r["kind"] == kind
                    and r["condition"] == condition
                    and r["method"] == name
                )
                for name in names
            ]
            bars = ax.bar(
                np.arange(3),
                [r["median_error_m"] * 1000 for r in records],
                color=["#8595ad", "#bd9a72", "#39ae9f"],
            )
            for bar, r in zip(bars, records):
                ax.annotate(
                    f"{r['localized']}/{r['total']} localized\n{r['false_localized_above_10mm']} errors >10 mm",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            ax.set_xticks(np.arange(3), labels, fontsize=9)
            ax.set_ylim(0, max(r["median_error_m"] * 1000 for r in records) * 1.55 + 1)
            ax.set_ylabel("Median detected-fit error (mm)")
            ax.set_title(
                kind.title()
                + " · "
                + (
                    "clean"
                    if condition == "clean"
                    else "1 mm depth noise + 20% pixel occlusion"
                ),
                fontsize=11,
            )
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Held-out camera views of recorded G1 warehouse trajectories\nKnown geometry and calibrated cameras · accuracy and abstention shown together",
        fontsize=14,
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.report, args.output)
