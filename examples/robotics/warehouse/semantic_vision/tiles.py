"""Fixed overlapping sensor-image crops, without a target-location prior."""

import numpy as np


def overlapping_tiles(rgb):
    """Four 60%-width/height crops; each pixel is observed, with 20% overlap."""
    height, width = rgb.shape[:2]
    crop_width, crop_height = int(np.ceil(width * 0.6)), int(np.ceil(height * 0.6))
    for y in (0, height - crop_height):
        for x in (0, width - crop_width):
            yield rgb[y : y + crop_height, x : x + crop_width], (x, y)


def suppress_overlaps(detections, threshold=0.5):
    """Score-ordered NMS for one query, preserving original neural scores."""
    from .study import iou

    kept = []
    for detection in sorted(detections, key=lambda d: d["score"], reverse=True):
        if all(
            iou(detection["box_xyxy"], old["box_xyxy"]) <= threshold for old in kept
        ):
            kept.append(detection)
    return kept


def detect_tiled(detector, rgb, query, **kwargs):
    proposals = []
    for crop, (x, y) in overlapping_tiles(rgb):
        for detection in detector.detect(crop, query, **kwargs):
            box = np.asarray(detection["box_xyxy"]) + [x, y, x, y]
            proposals.append(
                {**detection, "box_xyxy": box.tolist(), "source_crop_xy": [x, y]}
            )
    return suppress_overlaps(proposals)
