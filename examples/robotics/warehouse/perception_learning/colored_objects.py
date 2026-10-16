"""Explicit color priors for two-job warehouse cells, not learned recognition.

Green parcels and yellow receivers extend the original orange/magenta workcell.
Cyan is reserved for the robot localization fiducial and is never a parcel.
Color masks are canonicalized only to reuse the versioned geometric fitter;
provenance preserves the original measured RGB-D frame hash and actual color.
"""

import hashlib
from dataclasses import replace

import numpy as np

from ..perception import color_mask
from .robust_geometry import observe_robust_rgbd

COLOR_ROLES = {
    "orange": "parcel",
    "green": "parcel",
    "magenta": "receiver",
    "yellow": "receiver",
}


def semantic_color_mask(rgb, color):
    r, g, b = np.moveaxis(np.asarray(rgb, float), -1, 0)
    yellow = (g > 45) & (r > 0.8 * g) & (r < 1.4 * g) & (g > 1.7 * b)
    if color == "green":
        return (g > 45) & (r < 0.7 * g) & (b < 0.85 * g)
    if color == "yellow":
        return yellow
    if color == "orange":
        return color_mask(rgb, "orange") & ~yellow
    if color == "magenta":
        return color_mask(rgb, "magenta")
    raise ValueError("Unsupported explicit warehouse color prior")


def observe_colored_boxes(
    rgb, depth, calibration, timestamp, colors=tuple(COLOR_ROLES)
):
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    digest = hashlib.sha256(
        np.ascontiguousarray(rgb).tobytes()
        + np.ascontiguousarray(depth, dtype=float).tobytes()
    ).hexdigest()
    result = []
    for color in colors:
        kind = COLOR_ROLES[color]
        mask = semantic_color_mask(rgb, color)
        canonical = np.zeros_like(rgb)
        canonical[mask] = [230, 120, 20] if kind == "parcel" else [200, 20, 230]
        for hypothesis in observe_robust_rgbd(canonical, depth, calibration, timestamp):
            component = hypothesis.observation_id.rsplit("_", 1)[1]
            provenance = dict(hypothesis.provenance)
            provenance.update(
                original_frame_sha256=digest,
                actual_color_prior=color,
                color_method="authored_channel_ratio_segmentation",
                metric_size_prior="5cm_cube"
                if kind == "parcel"
                else "20cm_square_4mm_receiver",
            )
            result.append(
                replace(
                    hypothesis,
                    color=color,
                    observation_id=f"{kind}_{color}_{component}",
                    provenance=provenance,
                )
            )
    return result
