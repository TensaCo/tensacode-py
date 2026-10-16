"""Compose simultaneous calibrated RGB-D views into inspectable scene evidence.

The caller supplies calibrated workcell bounds, not live target coordinates.
This is an observation adapter for QA/planning; it does not command the robot.
"""

import numpy as np

from ..perception_tracking import SceneTracker
from .robust_geometry import observe_robust_rgbd


def observe_views(
    frames, timestamp, *, workcell_center_xy, workcell_halfwidth_m=0.7, colors=None
):
    """Frames contain rgb/depth/calibration; all views must be simultaneous.

    Returned track IDs are local to this observation revision. A caller must
    separately ground symbolic entity identity; one observation alone does not
    establish persistent identity or physical support/contact.
    """
    center = np.asarray(workcell_center_xy, float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("Finite calibrated workcell XY center required")
    if workcell_halfwidth_m <= 0:
        raise ValueError("Positive workcell bound required")
    if len(frames) < 2:
        raise ValueError("Multiview observer requires at least two calibrated views")
    camera_positions = [
        np.asarray(frame["calibration"]["position"], float) for frame in frames
    ]
    if (
        max(np.linalg.norm(a - b) for a in camera_positions for b in camera_positions)
        < 0.05
    ):
        raise ValueError(
            "Distinct camera positions required; repeated images are not multiview evidence"
        )
    if colors is not None:
        from .colored_objects import observe_colored_boxes

        observer = lambda rgb, depth, calibration, now: observe_colored_boxes(
            rgb, depth, calibration, now, colors
        )
    else:
        observer = observe_robust_rgbd
    tracker = SceneTracker()
    per_view = []
    for frame in frames:
        if abs(float(frame.get("timestamp", timestamp)) - timestamp) > 1e-6:
            raise ValueError("Views must share the same capture timestamp")
        hypotheses = observer(
            frame["rgb"], frame["depth"], frame["calibration"], timestamp
        )
        in_cell = [
            h
            for h in hypotheses
            if np.all(
                np.abs(np.asarray(h.center_xyz)[:2] - center) <= workcell_halfwidth_m
            )
        ]
        tracks = tracker.update(in_cell, timestamp)
        per_view.append(
            {
                "hypotheses": [h.to_dict() for h in in_cell],
                "outside_calibrated_cell": len(hypotheses) - len(in_cell),
            }
        )
    return {
        "timestamp": float(timestamp),
        "tracks": tracks,
        "views": per_view,
        "association_events": tracker.events,
        "workcell_prior": {
            "center_xy": center.tolist(),
            "halfwidth_m": workcell_halfwidth_m,
        },
        "scope": "simultaneous multiview known-box geometric observation; no physical contact assertion",
    }
