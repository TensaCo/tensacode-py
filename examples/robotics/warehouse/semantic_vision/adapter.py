"""Language-conditioned RGB boxes with conservative measured depth samples.

The neural detector consumes only RGB and query text. Depth enrichment consumes
only its boxes, an axial-depth image and camera calibration. Neither path accepts
simulator bodies, poses, object colors, identities, or evaluation labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DepthConfig:
    central_box_fraction: float = 0.25
    minimum_valid_pixels: int = 9
    minimum_valid_fraction: float = 0.8
    maximum_depth_iqr_m: float = 0.08
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 10.0


def visible_surface_sample(box, depth, camera, config=None):
    """Return a median visible-surface point, or explicitly abstain.

    The central crop is not segmentation: a returned point can be background.
    It must not be used as a localized object center or an authorized grasp pose.
    """
    config = config or DepthConfig()
    depth = np.asarray(depth, dtype=float)
    height, width = int(camera["height"]), int(camera["width"])
    if depth.shape != (height, width):
        raise ValueError("Depth shape differs from camera calibration")
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise ValueError("Finite xyxy box required")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Positive box extent required")
    if not 0 < config.central_box_fraction <= 1:
        raise ValueError("Central box fraction must be in (0, 1]")
    focal = float(camera["focal_pixels"])
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("Positive focal length required")
    center = (box[:2] + box[2:]) / 2
    half = (box[2:] - box[:2]) * config.central_box_fraction / 2
    low = np.maximum(0, np.floor(center - half).astype(int))
    high = np.minimum([width, height], np.ceil(center + half).astype(int))
    xs, ys = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]))
    values = depth[ys, xs]
    valid = (
        np.isfinite(values)
        & (values >= config.minimum_depth_m)
        & (values <= config.maximum_depth_m)
    )
    result = {
        "status": "abstained",
        "interpretation": "visible surface, not object center; background possible",
        "crop_xyxy": [int(low[0]), int(low[1]), int(high[0]), int(high[1])],
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
    }
    if (
        valid.sum() < config.minimum_valid_pixels
        or valid.mean() < config.minimum_valid_fraction
    ):
        return dict(result, reason="insufficient valid depth")
    z = values[valid]
    spread = float(np.quantile(z, 0.75) - np.quantile(z, 0.25))
    result["depth_iqr_m"] = spread
    if spread > config.maximum_depth_iqr_m:
        return dict(result, reason="mixed depth surfaces")
    x = (xs[valid] + 0.5 - width / 2) * z / focal
    y = (height / 2 - ys[valid] - 0.5) * z / focal
    position, forward, right, up = [
        np.asarray(camera[key], dtype=float)
        for key in ("position", "forward", "right", "up")
    ]
    if any(
        a.shape != (3,) or not np.all(np.isfinite(a))
        for a in (position, forward, right, up)
    ):
        raise ValueError("Finite camera position and basis required")
    basis = np.stack([right, up, forward])
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-4):
        raise ValueError("Orthonormal camera basis required")
    points = position + z[:, None] * forward + x[:, None] * right + y[:, None] * up
    return dict(
        result,
        status="surface_sample",
        visible_surface_xyz=np.median(points, axis=0).tolist(),
        median_axial_depth_m=float(np.median(z)),
    )


class GroundedDetector:
    """Lazy optional Transformers dependency; CPU is the explicit default."""

    def __init__(self, model_path: str | Path, *, device="cpu", threads=4):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        torch.set_num_threads(threads)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                model_path, local_files_only=True
            )
            .to(device)
            .eval()
        )

    def detect(self, rgb, query, *, box_threshold=0.35, text_threshold=0.25):
        import torch
        from PIL import Image

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Nonempty language query required")
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        text = query.strip().lower().rstrip(".") + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = result.get("text_labels", result.get("labels"))
        return [
            {
                "box_xyxy": box.tolist(),
                "score": float(score),
                "label": str(label),
                "query": query,
            }
            for box, score, label in zip(
                result["boxes"].cpu(), result["scores"].cpu(), labels
            )
        ]
