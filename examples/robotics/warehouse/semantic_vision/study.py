"""Run frozen neural grounding on replay RGB-D; score labels only afterwards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .adapter import DepthConfig, GroundedDetector, visible_surface_sample


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    intersection = np.maximum(
        0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])
    ).prod()
    union = (
        np.maximum(0, a[2:] - a[:2]).prod()
        + np.maximum(0, b[2:] - b[:2]).prod()
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0


def score_partial_labels(rows, annotations):
    """Recall against partial AMODAL labels only; never report whole-image precision."""
    by_frame = {frame["path"]: frame["boxes"] for frame in annotations["frames"]}
    scored = []
    for row in rows:
        if row["query"] not in ("a box", "a small cube"):
            continue
        truth = by_frame.get(row["frame"], [])
        detections = row["detections"]
        pairs = sorted(
            (
                (iou(d["box_xyxy"], gt["bbox_xyxy"]), i, j)
                for i, d in enumerate(detections)
                for j, gt in enumerate(truth)
            ),
            reverse=True,
        )
        used_pred, used_truth = set(), set()
        matches = []
        for overlap, i, j in pairs:
            if overlap < 0.5 or i in used_pred or j in used_truth:
                continue
            used_pred.add(i)
            used_truth.add(j)
            matches.append({"detection": i, "label": j, "iou": overlap})
        scored.append(
            {
                "frame": row["frame"],
                "amodal_parcel_labels": len(truth),
                "matched_at_iou_0_5": len(matches),
                "matches": matches,
            }
        )
    total = sum(r["amodal_parcel_labels"] for r in scored)
    hits = sum(r["matched_at_iou_0_5"] for r in scored)
    return {
        "scope": "partial task-parcel amodal-box recall; occluded parcels included, not visible-object recall or whole-image precision",
        "frames": scored,
        "amodal_parcel_labels": total,
        "matched_at_iou_0_5": hits,
        "recall_at_iou_0_5": hits / total if total else None,
    }


def run(capture, output, model_path, split, limit=None, method="whole_image"):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite study output: {output}")
    output.mkdir(parents=True)
    protocol_path = Path(__file__).with_name("protocol.json")
    protocol = json.loads(protocol_path.read_text())
    shutil.copy2(protocol_path, output / "protocol.json")
    manifest = json.loads((capture / "manifest.json").read_text())
    frames = manifest["frames"][:limit]
    source_hashes = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    with zipfile.ZipFile(output / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in Path(__file__).parent.glob("*.py"):
            archive.write(source, source.name)
    started = time.monotonic()
    detector = GroundedDetector(model_path, device="cpu", threads=4)
    loaded = time.monotonic()
    rows = []
    unassigned = []
    queries = (
        protocol["joint_prompt"]["queries"]
        if method == "joint_prompt"
        else protocol["queries"]
    )
    for frame in frames:
        folder = capture / frame["path"]
        rgb = np.asarray(Image.open(folder / "sensor-rgb.png").convert("RGB"))
        depth = np.load(folder / "sensor-depth.npz")["depth"]
        contract = json.loads((folder / "sensor-contract.json").read_text())
        out_frame = output / frame["path"]
        out_frame.mkdir()
        for name in ("sensor-rgb.png", "sensor-depth.npz", "sensor-contract.json"):
            shutil.copy2(folder / name, out_frame / name)
        if method == "joint_prompt":
            tick = time.monotonic()
            joint = detector.detect(
                rgb,
                ". ".join(queries),
                box_threshold=protocol["box_threshold"],
                text_threshold=protocol["text_threshold"],
            )
            shared_seconds = time.monotonic() - tick
            terms = protocol["joint_prompt"]["label_terms"]
            for detection in joint:
                if not any(term in detection["label"].lower() for term in terms):
                    unassigned.append({"frame": frame["path"], **detection})
        for qi, query in enumerate(queries):
            tick = time.monotonic()
            kwargs = {
                "box_threshold": protocol["box_threshold"],
                "text_threshold": protocol["text_threshold"],
            }
            if method == "tiles_2x2":
                from .tiles import detect_tiled

                detections = detect_tiled(detector, rgb, query, **kwargs)
            elif method == "joint_prompt":
                term = terms[qi]
                detections = [
                    {
                        **d,
                        "query": query,
                        "joint_prompt": ". ".join(queries),
                        "ambiguous_label": sum(t in d["label"].lower() for t in terms)
                        > 1,
                    }
                    for d in joint
                    if term in d["label"].lower()
                ]
            else:
                detections = detector.detect(rgb, query, **kwargs)
            detections.sort(key=lambda d: d["score"], reverse=True)
            omitted = max(0, len(detections) - protocol["max_detections_per_query"])
            detections = detections[: protocol["max_detections_per_query"]]
            for detection in detections:
                detection["depth_evidence"] = visible_surface_sample(
                    detection["box_xyxy"],
                    depth,
                    contract["camera"],
                    DepthConfig(**protocol["depth"]),
                )
            row = {
                "frame": frame["path"],
                "query": query,
                "detections": detections,
                "omitted_below_top_k": omitted,
                "elapsed_s": shared_seconds
                if method == "joint_prompt"
                else time.monotonic() - tick,
                "shared_forward": method == "joint_prompt",
                "rgb_sha256": digest(folder / "sensor-rgb.png"),
                "depth_sha256": digest(folder / "sensor-depth.npz"),
                "calibration_sha256": digest(folder / "sensor-contract.json"),
            }
            rows.append(row)
            canvas = Image.fromarray(rgb).copy()
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, canvas.width, 25), fill="#14202c")
            draw.text(
                (8, 7),
                f"{query}: {len(detections)} detection(s), {row['elapsed_s']:.2f}s CPU",
                fill="white",
            )
            for detection in detections:
                box = detection["box_xyxy"]
                draw.rectangle(box, outline="#00ffff", width=2)
                draw.text(
                    (max(0, box[0]), max(26, box[1] - 12)),
                    f"{detection['score']:.2f} {detection['label']}",
                    fill="#00ffff",
                )
            canvas.save(out_frame / f"query-{qi}.jpg", quality=90)
            print(
                json.dumps(
                    {
                        "frame": row["frame"],
                        "query": query,
                        "detections": len(detections),
                        "elapsed_s": row["elapsed_s"],
                    }
                ),
                flush=True,
            )
            (output / "predictions.json").write_text(json.dumps(rows, indent=2) + "\n")
    # Only after predictions are complete: evaluator may read independent labels.
    labels_path = capture / "evaluation-amodal-boxes.json"
    scoring = None
    if labels_path.exists():
        shutil.copy2(labels_path, output / labels_path.name)
        scoring = score_partial_labels(rows, json.loads(labels_path.read_text()))
    import torch
    import transformers

    summary = {
        "schema": "tensorcode.semantic-vision-results.v1",
        "split": split,
        "method": method,
        "neural_forward_passes": len(frames)
        if method == "joint_prompt"
        else len(rows) * (4 if method == "tiles_2x2" else 1),
        "unassigned_neural_proposals": unassigned,
        "source_capture_manifest": manifest,
        "protocol_sha256": digest(protocol_path),
        "model": protocol["model"],
        "model_revision": model_path.name,
        "weights_sha256": digest(model_path / "model.safetensors"),
        "source_sha256": source_hashes,
        "runtime": {
            "device": "cpu",
            "threads": 4,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "load_s": loaded - started,
            "elapsed_s": time.monotonic() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "frames": len(frames),
        "query_evaluations": len(rows),
        "per_query": [
            {
                "query": query,
                "frames_with_detections": sum(
                    bool(row["detections"]) for row in rows if row["query"] == query
                ),
                "detections": sum(
                    len(row["detections"]) for row in rows if row["query"] == query
                ),
            }
            for query in queries
        ],
        "surface_samples": sum(
            d["depth_evidence"]["status"] == "surface_sample"
            for row in rows
            for d in row["detections"]
        ),
        "depth_abstentions": sum(
            d["depth_evidence"]["status"] == "abstained"
            for row in rows
            for d in row["detections"]
        ),
        "partial_label_scoring": scoring,
        "scope": protocol["scope"],
        "limits": protocol["evaluation_limits"],
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "evaluation"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--method",
        choices=["whole_image", "tiles_2x2", "joint_prompt"],
        default="whole_image",
    )
    args = parser.parse_args()
    run(args.capture, args.output, args.model_path, args.split, args.limit, args.method)
