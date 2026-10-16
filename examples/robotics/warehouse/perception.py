"""Bounded inverse RGB-D geometry for a calibrated warehouse workcell.

Known color, metric dimensions, and world-axis alignment are explicit priors.
Only pixels, depth and camera calibration enter this module. A fit is a geometric
hypothesis, not recognition of arbitrary objects or a calibrated probability.
Occlusion can leave position unobservable; those fits are marked ambiguous.
"""

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class BoxPrior:
    kind: str
    color: str
    dimensions_xyz: tuple[float, float, float]

    def __post_init__(self):
        if self.color not in {"orange", "magenta"}:
            raise ValueError("Supported segmentation priors: orange, magenta")
        if len(self.dimensions_xyz) != 3 or not all(
            np.isfinite(x) and x > 0 for x in self.dimensions_xyz
        ):
            raise ValueError("Box dimensions must be finite and positive")


DEFAULT_PRIORS = (
    BoxPrior("parcel", "orange", (0.05, 0.05, 0.05)),
    BoxPrior("receiver", "magenta", (0.20, 0.20, 0.004)),
)


@dataclass(frozen=True)
class BoxHypothesis:
    observation_id: str
    kind: str
    color: str
    center_xyz: tuple[float, float, float]
    top_center_xyz: tuple[float, float, float]
    dimensions_xyz: tuple[float, float, float]
    residual_m: float
    confidence_score: float
    position_ambiguity_m: tuple[float, float, float]
    observed_face_axes: tuple[int, ...]
    constrained_axes: tuple[int, ...]
    visible_points: int
    timestamp: float
    provenance: dict

    def to_dict(self):
        return asdict(self)


def backproject(depth, calibration):
    """Axial depth, pixel-center convention matching the G1 RGB-D renderer."""
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2:
        raise ValueError("Depth must be H x W axial meters")
    h, w = depth.shape
    focal = float(calibration["focal_pixels"])
    basis = np.array([calibration[k] for k in ("right", "up", "forward")], float)
    origin = np.asarray(calibration["position"], float)
    if not np.isfinite(focal) or focal <= 0 or origin.shape != (3,):
        raise ValueError("Invalid camera calibration")
    if (
        basis.shape != (3, 3)
        or not np.all(np.isfinite(origin))
        or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-4)
    ):
        raise ValueError("Camera axes must be finite and orthonormal")
    if calibration.get("width", w) != w or calibration.get("height", h) != h:
        raise ValueError("Calibration resolution differs from image")
    rows, cols = np.indices((h, w))
    rays = (
        (cols + 0.5 - w / 2)[..., None] * basis[0]
        + (h / 2 - rows - 0.5)[..., None] * basis[1]
    ) / focal + basis[2]
    return origin + depth[..., None] * rays


def color_mask(rgb, color):
    r, g, b = np.moveaxis(np.asarray(rgb, float), -1, 0)
    if color == "orange":
        return (r > 65) & (r > 0.92 * g) & (g > 1.8 * b) & (g > 25)
    if color == "magenta":
        return (
            (r > 45)
            & (b > 55)
            & (r > 2 * g)
            & (b > 2 * g)
            & (b > 0.8 * r)
            & (r > 0.45 * b)
        )
    raise ValueError("Unsupported color prior")


def fit_box(
    points,
    normals,
    camera_position,
    prior,
    *,
    timestamp=0.0,
    observation_id="box",
    provenance=None,
):
    """Fit visible axis-aligned planes, then infer center using metric size.

    Missing face coordinates use visible extent midpoints and retain ambiguity.
    Normals are measured local depth derivatives, never supplied object normals.
    """
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
        raise ValueError("Points and normals must both be N x 3")
    if len(points) < 12 or not np.all(np.isfinite(points)):
        raise ValueError("At least 12 finite depth points required")
    camera = np.asarray(camera_position, float)
    if (
        camera.shape != (3,)
        or not np.all(np.isfinite(camera))
        or not np.isfinite(timestamp)
    ):
        raise ValueError("Finite camera position and timestamp required")
    size = np.array(prior.dimensions_xyz)
    lo, hi = np.quantile(points, [0.01, 0.99], axis=0)
    extent = hi - lo
    if np.any(extent > size + 0.015):
        raise ValueError("Observed component exceeds known box dimensions")
    center = (lo + hi) / 2
    ambiguity = np.maximum(0.0, size - extent) / 2
    faces, constrained = [], []
    for axis in range(3):
        face = np.abs(normals[:, axis]) > 0.96
        if np.count_nonzero(face) >= max(6, 0.015 * len(points)):
            plane = np.median(points[face, axis])
            # Only the camera-facing face can be measured on an opaque box.
            side = 1 if camera[axis] > plane else -1
            center[axis] = plane - side * size[axis] / 2
            ambiguity[axis] = max(0.0005, float(np.std(points[face, axis])))
            faces.append(axis)
            constrained.append(axis)
        elif extent[axis] >= 0.85 * size[axis]:
            constrained.append(axis)
    q = np.abs(points - center) - size / 2
    sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)
    residual = float(np.sqrt(np.mean(sdf * sdf)))
    score = float(np.exp(-residual / 0.003) * len(constrained) / 3)
    if len(constrained) != 3:
        score = min(score, 0.49)
    info = dict(provenance or {})
    info.update(
        {
            "method": "axis_aligned_visible_plane_inverse_geometry",
            "priors": {
                "color": prior.color,
                "dimensions_xyz": size.tolist(),
                "orientation": "world_axis_aligned",
            },
            "confidence_semantics": "heuristic_quality_not_probability",
        }
    )
    return BoxHypothesis(
        observation_id,
        prior.kind,
        prior.color,
        tuple(center),
        tuple(center + [0, 0, size[2] / 2]),
        tuple(size),
        residual,
        score,
        tuple(ambiguity),
        tuple(faces),
        tuple(constrained),
        len(points),
        float(timestamp),
        info,
    )


def observe_rgbd(rgb, depth, calibration, timestamp, priors=DEFAULT_PRIORS):
    """Detect disconnected color components and fit each from measured depth.

    IDs are observation-local, not persistent identity. A tracker must associate
    these observations before assigning parcel_0/1 or receiver_0/1 identities.
    """
    rgb, depth = np.asarray(rgb), np.asarray(depth, float)
    if rgb.shape != (*depth.shape, 3):
        raise ValueError("RGB and depth resolutions must agree")
    cloud = backproject(depth, calibration)
    # Central differences restricted to the same segmented component prevent
    # object silhouettes and missing depth from masquerading as observed faces.
    normals = np.zeros_like(cloud)
    with np.errstate(invalid="ignore", divide="ignore"):
        cross = np.cross(
            cloud[1:-1, 2:] - cloud[1:-1, :-2], cloud[2:, 1:-1] - cloud[:-2, 1:-1]
        )
        length = np.linalg.norm(cross, axis=2, keepdims=True)
        normals[1:-1, 1:-1] = np.divide(
            cross, length, out=np.zeros_like(cross), where=length > 1e-12
        )
    digest = hashlib.sha256(
        np.ascontiguousarray(rgb).tobytes() + np.ascontiguousarray(depth).tobytes()
    ).hexdigest()
    hypotheses = []
    for prior in priors:
        mask = color_mask(rgb, prior.color) & np.isfinite(depth) & (depth > 0)
        components, count = label(mask)
        for component in range(1, count + 1):
            selected = components == component
            if selected.sum() < 20:
                continue
            interior = np.zeros_like(selected)
            interior[1:-1, 1:-1] = (
                selected[1:-1, 1:-1]
                & selected[1:-1, 2:]
                & selected[1:-1, :-2]
                & selected[2:, 1:-1]
                & selected[:-2, 1:-1]
            )
            measured_normals = normals[selected].copy()
            measured_normals[~interior[selected]] = 0
            try:
                fit = fit_box(
                    cloud[selected],
                    measured_normals,
                    calibration["position"],
                    prior,
                    timestamp=timestamp,
                    observation_id=f"{prior.kind}_{component}",
                    provenance={
                        "source": "calibrated_rgbd_pixels",
                        "frame_sha256": digest,
                        "pixel_bbox_xyxy": [
                            int(np.where(selected)[1].min()),
                            int(np.where(selected)[0].min()),
                            int(np.where(selected)[1].max()),
                            int(np.where(selected)[0].max()),
                        ],
                    },
                )
            except ValueError:
                continue
            hypotheses.append(fit)
    return hypotheses


def relation_candidates(hypotheses, tolerance_m=0.015):
    """Geometric support candidates, never assertions of physical contact."""
    result = []
    for item in hypotheses:
        for support in hypotheses:
            if item is support or support.kind != "receiver":
                continue
            bottom = item.center_xyz[2] - item.dimensions_xyz[2] / 2
            dz = abs(bottom - support.top_center_xyz[2])
            inside = all(
                abs(item.center_xyz[i] - support.center_xyz[i])
                + item.dimensions_xyz[i] / 2
                <= support.dimensions_xyz[i] / 2 + tolerance_m
                for i in (0, 1)
            )
            if inside and dz < tolerance_m:
                result.append(
                    {
                        "subject": item.observation_id,
                        "predicate": "possibly_supported_by",
                        "object": support.observation_id,
                        "vertical_gap_m": dz,
                        "confidence_score": min(
                            item.confidence_score, support.confidence_score
                        ),
                        "source": "rgbd_box_geometry_not_contact_measurement",
                    }
                )
    return result
