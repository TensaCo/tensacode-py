"""Offline RGB-D perception audit; evaluation truth never enters the fitter.

Usage: python perception_audit.py RUN_DIRECTORY --output report.json
The handling capture layout is supported directly. Other captures can supply
``evaluation_truth.json`` with entities [{id, kind, center_xyz, dimensions_xyz}].
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

try:
    from .perception import BoxPrior, backproject, color_mask, observe_rgbd
except ImportError:
    from perception import BoxPrior, backproject, color_mask, observe_rgbd


# Fixture priors are supplied independently of evaluation truth.
WORKCELL_PRIORS = (
    BoxPrior("parcel", "orange", (0.05, 0.05, 0.05)),
    BoxPrior("receiver", "magenta", (0.20, 0.20, 0.004)),
)


def capture_hypotheses(directory, priors=WORKCELL_PRIORS):
    """Inference boundary: opens image/depth/calibration only, never truth files."""
    directory = Path(directory)
    rgb = np.array(Image.open(directory / "sensor-rgb.png").convert("RGB"))
    with np.load(directory / "sensor-depth.npz") as data:
        depth = data["depth"]
    contract = json.loads((directory / "sensor-contract.json").read_text())
    calibration = contract["camera"]
    timestamp = float(contract.get("capture_timestamp", 0.0))
    hypotheses = observe_rgbd(rgb, depth, calibration, timestamp, priors)
    cloud = backproject(depth, calibration)
    baselines = {}
    for prior in priors:
        components, count = label(
            color_mask(rgb, prior.color) & np.isfinite(depth) & (depth > 0)
        )
        for component in range(1, count + 1):
            points = cloud[components == component]
            if len(points) < 20:
                continue
            centroid = points.mean(axis=0)
            corrected = centroid.copy()
            # A simple top-view known-height baseline, not the old controller's
            # actual robust top-face estimator, which is reported separately.
            corrected[2] -= prior.dimensions_xyz[2] / 2
            baselines[f"{prior.kind}_{component}"] = {
                "visible_surface_centroid_xyz": centroid.tolist(),
                "top_view_size_corrected_centroid_xyz": corrected.tolist(),
            }
    files = ["sensor-rgb.png", "sensor-depth.npz", "sensor-contract.json"]
    provenance = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in files
    }
    return hypotheses, baselines, provenance


def evaluation_truth(directory):
    """Evaluation-only truth reader, invoked after inference has returned."""
    directory = Path(directory)
    explicit = directory / "evaluation_truth.json"
    if explicit.exists():
        return json.loads(explicit.read_text())["entities"], {}
    with gzip.open(directory / "records.json.gz", "rt") as handle:
        initial = json.load(handle)[0]
    probe = json.loads((directory / "probe.json").read_text())
    top = np.asarray(probe["receiving_surface_xyz"], float)
    entities = [
        {
            "id": "parcel",
            "kind": "parcel",
            "center_xyz": initial["object"],
            "dimensions_xyz": [0.05, 0.05, 0.05],
        },
        {
            "id": "receiver",
            "kind": "receiver",
            "center_xyz": (top - [0, 0, 0.002]).tolist(),
            "dimensions_xyz": [0.20, 0.20, 0.004],
        },
    ]
    state = initial.get("sensor_state", {})
    prior_estimates = {}
    if state.get("object_position") is not None:
        prior_estimates["parcel"] = state["object_position"]
    if state.get("destination_position") is not None:
        prior_estimates["receiver"] = (
            np.asarray(state["destination_position"]) - [0, 0, 0.002]
        ).tolist()
    return entities, prior_estimates


def audit_capture(directory):
    hypotheses, baselines, hashes = capture_hypotheses(directory)
    truth, old_estimates = evaluation_truth(directory)
    rows = []
    assigned = set()
    for kind in sorted({entity["kind"] for entity in truth}):
        expected = [e for e in truth if e["kind"] == kind]
        observed = [h for h in hypotheses if h.kind == kind]
        matches = {}
        if observed:
            cost = np.array(
                [
                    [
                        np.linalg.norm(np.asarray(h.center_xyz) - e["center_xyz"])
                        for h in observed
                    ]
                    for e in expected
                ]
            )
            ei, hi = linear_sum_assignment(cost)
            matches = dict(zip(ei, hi))
        for index, entity in enumerate(expected):
            row = {
                "evaluation_entity": entity["id"],
                "kind": kind,
                "status": "not_detected",
                "truth_center_xyz": entity["center_xyz"],
            }
            if index in matches:
                fit = observed[matches[index]]
                assigned.add(fit.observation_id)
                baseline = baselines[fit.observation_id]
                error = lambda point, center=entity["center_xyz"]: float(
                    np.linalg.norm(np.asarray(point) - center)
                )
                row.update(
                    status="localized" if fit.confidence_score >= 0.7 else "ambiguous",
                    hypothesis=fit.to_dict(),
                    center_error_m=error(fit.center_xyz),
                    baseline={
                        **baseline,
                        "visible_centroid_error_m": error(
                            baseline["visible_surface_centroid_xyz"]
                        ),
                        "size_corrected_centroid_error_m": error(
                            baseline["top_view_size_corrected_centroid_xyz"]
                        ),
                    },
                )
                if entity["id"] in old_estimates:
                    row["original_controller_estimator_error_m"] = error(
                        old_estimates[entity["id"]]
                    )
            rows.append(row)
    return {
        "capture": Path(directory).name,
        "frame_files_sha256": hashes,
        "inference_inputs": [
            "RGB pixels",
            "axial depth pixels",
            "camera calibration",
            "color/size/alignment priors",
        ],
        "evaluation_association": "one-to-one nearest same-kind truth after inference; not controller tracking",
        "entities": rows,
        "unmatched_hypotheses": [
            h.to_dict() for h in hypotheses if h.observation_id not in assigned
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "schema": "tensorcode.warehouse.perception-audit.v1",
        "scope": "Offline initial captured RGB-D frame audit; not a controller intervention or full trajectory perception test",
        "limitations": [
            "Known color, metric size and world-axis alignment",
            "Heuristic confidence is not calibrated probability",
            "Raw centroid estimates the visible surface, so size-corrected and original controller baselines are also reported",
            "No learned recognition or persistent instance tracking",
        ],
        "captures": [audit_capture(path) for path in args.captures],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
