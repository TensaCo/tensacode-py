"""Matched replay-view evaluation; inference completes before truth scoring."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from warehouse.perception import backproject, color_mask, observe_rgbd
from warehouse.perception_learning.robust_geometry import observe_robust_rgbd
from warehouse.perception_tracking import SceneTracker, TrackingConfig

CONDITIONS = ("clean", "noise", "occlusion", "noise_occlusion")


def measured_frame(path, condition, seed):
    rgb = np.array(Image.open(path / "sensor-rgb.png"))
    depth = np.load(path / "sensor-depth.npz")["depth"].copy()
    contract = json.loads((path / "sensor-contract.json").read_text())
    if "noise" in condition:
        depth += np.random.default_rng(seed).normal(0, 0.001, depth.shape)
    if "occlusion" in condition:
        for color in ("orange", "magenta"):
            components, count = label(color_mask(rgb, color))
            for component in range(1, count + 1):
                rows, cols = np.where(components == component)
                if len(rows) >= 20:
                    cutoff = np.quantile(cols, 0.2)
                    rgb[rows[cols <= cutoff], cols[cols <= cutoff]] = 0
                    depth[rows[cols <= cutoff], cols[cols <= cutoff]] = np.nan
    hypotheses = observe_rgbd(
        rgb, depth, contract["camera"], contract["capture_timestamp"]
    )
    cloud = backproject(depth, contract["camera"])
    centroids = {}
    for color, kind, half_height in (
        ("orange", "parcel", 0.025),
        ("magenta", "receiver", 0.002),
    ):
        components, count = label(
            color_mask(rgb, color) & np.isfinite(depth) & (depth > 0)
        )
        for component in range(1, count + 1):
            points = cloud[components == component]
            if len(points) >= 20:
                center = points.mean(axis=0) - [0, 0, half_height]
                centroids[f"{kind}_{component}"] = center
    robust = observe_robust_rgbd(
        rgb, depth, contract["camera"], contract["capture_timestamp"]
    )
    # Calibrated camera look-at center (distance2.5m is a fixed rig prior), not
    # object truth. All baselines receive the same cell association boundary.
    camera = contract["camera"]
    cell_center = np.asarray(camera["position"]) + 2.5 * np.asarray(camera["forward"])

    def in_cell(h):
        return np.all(np.abs(np.asarray(h.center_xyz)[:2] - cell_center[:2]) <= 0.7)

    return (
        [h for h in hypotheses if in_cell(h)],
        centroids,
        [h for h in robust if in_cell(h)],
    )


def single(hypotheses, kind):
    candidates = [h for h in hypotheses if h.kind == kind]
    return candidates[0] if len(candidates) == 1 else None


def evaluate(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    groups = {}
    for entry in manifest["frames"]:
        groups.setdefault((entry["frame_index"], entry["lane"]), []).append(entry)
    rows = []
    for condition in CONDITIONS:
        trackers = {}
        for (frame, lane), views in sorted(groups.items()):
            tracker = trackers.setdefault(lane, SceneTracker())
            simultaneous = SceneTracker()
            observations, centroids, robust_observations = [], [], []
            for entry in sorted(views, key=lambda x: x["view"]):
                hypotheses, centers, robust = measured_frame(
                    dataset / entry["path"],
                    condition,
                    frame * 31 + lane * 7 + entry["view"],
                )
                observations.append(hypotheses)
                robust_observations.append(robust)
                centroids.append(centers)
                tracks = tracker.update(robust, entry["timestamp"])
                fused = simultaneous.update(robust, entry["timestamp"])
            # Only after sensor inference and tracking have completed do labels
            # enter the offline evaluator. They never choose detection identity.
            truth = json.loads(
                (dataset / views[0]["path"] / "evaluation_truth.json").read_text()
            )["entities"]
            for entity in truth:
                kind = entity["kind"]
                first = single(observations[0], kind)
                methods = {}
                if first is not None:
                    methods["single_inverse"] = (
                        first.center_xyz,
                        first.confidence_score >= 0.7,
                        None,
                    )
                    methods["single_corrected_centroid"] = (
                        centroids[0][first.observation_id],
                        True,
                        None,
                    )
                first_robust = single(robust_observations[0], kind)
                if first_robust is not None:
                    methods["single_robust"] = (
                        first_robust.center_xyz,
                        first_robust.confidence_score >= 0.7,
                        None,
                    )
                selected = [
                    (single(obs, kind), center)
                    for obs, center in zip(observations, centroids)
                ]
                visible = [(h, c) for h, c in selected if h is not None]
                if visible:
                    methods["multiview_mean_inverse"] = (
                        np.mean([h.center_xyz for h, _ in visible], axis=0),
                        True,
                        None,
                    )
                    methods["multiview_mean_centroid"] = (
                        np.mean([c[h.observation_id] for h, c in visible], axis=0),
                        True,
                        None,
                    )
                for name, candidates in (
                    ("multiview_fusion", fused),
                    ("temporal_multiview", tracks),
                ):
                    fresh = [
                        t
                        for t in candidates
                        if t["kind"] == kind
                        and any(
                            age is not None and age < 1e-6
                            for age in t["axis_evidence_age_s"]
                        )
                    ]
                    if len(fresh) == 1:
                        t = fresh[0]
                        methods[name] = (t["center_xyz"], t["localized"], t)
                for name in (
                    "single_inverse",
                    "single_robust",
                    "single_corrected_centroid",
                    "multiview_mean_inverse",
                    "multiview_mean_centroid",
                    "multiview_fusion",
                    "temporal_multiview",
                ):
                    prediction = methods.get(name)
                    row = {
                        "dataset": manifest["run"],
                        "split": manifest["split"],
                        "condition": condition,
                        "frame_index": frame,
                        "lane": lane,
                        "timestamp": views[0]["timestamp"],
                        "entity": entity["id"],
                        "kind": kind,
                        "method": name,
                        "detected": prediction is not None,
                        "localized": False,
                    }
                    if prediction is not None:
                        position, valid, track = prediction
                        row.update(
                            center_xyz=np.asarray(position).tolist(),
                            localized=bool(valid),
                            error_m=float(
                                np.linalg.norm(
                                    np.asarray(position) - entity["center_xyz"]
                                )
                            ),
                        )
                        if track:
                            row["position_std_xyz_m"] = track["position_std_xyz_m"]
                            row["track_id"] = track["id"]
                            row["axis_evidence_age_s"] = track["axis_evidence_age_s"]
                    rows.append(row)
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(
            (row["split"], row["kind"], row["condition"], row["method"]), []
        ).append(row)
    result = []
    for (split, kind, condition, method), group in sorted(groups.items()):
        detected = [r["error_m"] for r in group if r["detected"]]
        localized = [r["error_m"] for r in group if r["localized"]]
        result.append(
            {
                "split": split,
                "kind": kind,
                "condition": condition,
                "method": method,
                "total": len(group),
                "detected": len(detected),
                "localized": len(localized),
                "median_error_m": float(np.median(detected)) if detected else None,
                "p90_error_m": float(np.quantile(detected, 0.9)) if detected else None,
                "localized_coverage": len(localized) / len(group),
                "false_localized_above_10mm": sum(e > 0.01 for e in localized),
                "localized_median_error_m": float(np.median(localized))
                if localized
                else None,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset in args.datasets:
        rows.extend(evaluate(dataset))
    source_paths = [
        Path(__file__),
        Path(__file__).parents[1] / "perception.py",
        Path(__file__).parents[1] / "perception_tracking.py",
        Path(__file__).with_name("protocol.json"),
        Path(__file__).with_name("robust_geometry.py"),
    ]
    report = {
        "schema": "tensorcode.warehouse.perception-campaign.v1",
        "config": asdict(TrackingConfig()),
        "sources_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths
        },
        "capture_provenance": [
            {
                "run": json.loads((dataset / "manifest.json").read_text())["run"],
                "manifest": json.loads((dataset / "manifest.json").read_text()),
                "files_sha256": {
                    str(path.relative_to(dataset)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(dataset.glob("frame-*/*"))
                    if path.is_file()
                },
            }
            for dataset in args.datasets
        ],
        "summary": summarize(rows),
        "rows": rows,
        "limitations": [
            "Replay-view evaluation, not policy intervention",
            "Multiple views supply extra sensing; multiview arithmetic-mean baselines separate that advantage from fusion",
            "Known axis-aligned size/color priors",
            "Heuristic uncertainty; no calibrated probability claim",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
