"""Trim inconsistent RGB/depth silhouette pixels against an initial box hypothesis.

This is bounded geometric consensus with known shape priors. Rejected pixels and
inlier fraction remain explicit; a model unsupported by most pixels is not
promoted. The original frozen estimator remains the matched baseline.
"""

from dataclasses import replace

import numpy as np
from scipy.ndimage import label

try:
    from ..perception import DEFAULT_PRIORS, backproject, color_mask, observe_rgbd
except ImportError:
    from warehouse.perception import (
        DEFAULT_PRIORS,
        backproject,
        color_mask,
        observe_rgbd,
    )


def observe_robust_rgbd(
    rgb,
    depth,
    calibration,
    timestamp,
    *,
    surface_gate_m=0.008,
    minimum_inlier_fraction=0.7,
):
    hypotheses = observe_rgbd(rgb, depth, calibration, timestamp)
    cloud = backproject(depth, calibration)
    result = []
    for hypothesis in hypotheses:
        components, _ = label(
            color_mask(rgb, hypothesis.color) & np.isfinite(depth) & (depth > 0)
        )
        component = int(hypothesis.observation_id.rsplit("_", 1)[1])
        selected = components == component
        points = cloud[selected]
        q = (
            np.abs(points - np.asarray(hypothesis.center_xyz))
            - np.asarray(hypothesis.dimensions_xyz) / 2
        )
        sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(
            np.max(q, axis=1), 0
        )
        inliers = np.abs(sdf) <= surface_gate_m
        fraction = float(inliers.mean())
        if fraction < minimum_inlier_fraction or inliers.sum() < 20:
            result.append(hypothesis)
            continue
        isolated = np.zeros_like(rgb)
        rows, cols = np.where(selected)
        isolated[rows[inliers], cols[inliers]] = rgb[rows[inliers], cols[inliers]]
        priors = tuple(p for p in DEFAULT_PRIORS if p.kind == hypothesis.kind)
        fits = observe_rgbd(isolated, depth, calibration, timestamp, priors)
        # A fragmented geometric consensus cannot justify a unique object pose.
        if len(fits) != 1:
            result.append(hypothesis)
            continue
        fit = fits[0]
        provenance = dict(fit.provenance)
        provenance.update(
            original_frame_sha256=hypothesis.provenance["frame_sha256"],
            method="visible_plane_geometric_consensus",
            inlier_fraction=fraction,
            rejected_pixels=int((~inliers).sum()),
            surface_gate_m=surface_gate_m,
        )
        result.append(
            replace(
                fit, observation_id=hypothesis.observation_id, provenance=provenance
            )
        )
    return result
