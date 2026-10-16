"""Independent RGB-D job beliefs with explicit calibrated color selection.

Job a is an orange parcel / magenta receiver; job b is green / blue.
Color normalization is an internal perception stage, never rendered evidence.
Call both observers every control tick to maintain separate fresh posteriors.
"""

import json
from pathlib import Path

import numpy as np

try:
    from .physics_sensors import SharedSensorBelief
except ImportError:  # Direct example-script import.
    from physics_sensors import SharedSensorBelief


def select_job_colors(rgb, job):
    """Return canonical detector input and raw-pixel masks, without editing RGB.

    Color ratios are calibration priors, not simulator segmentation identities.
    Unselected pixels become neutral gray, while the cyan pelvis marker remains
    unchanged so the shared optical localization still consumes physical pixels.
    """
    if job not in ("a", "b"):
        raise ValueError("job must be 'a' or 'b'")
    r, g, b = np.moveaxis(np.asarray(rgb, dtype=float), -1, 0)
    orange = (r > 65) & (r > 0.92 * g) & (g > 1.8 * b) & (g > 25)
    green = (g > 45) & (g > 1.7 * r) & (g > 1.5 * b)
    magenta = (
        (r > 45) & (b > 55) & (r > 2 * g) & (b > 2 * g) & (b > 0.8 * r) & (r > 0.45 * b)
    )
    blue = (b > 55) & (b > 1.8 * g) & (r < 0.3 * b)
    cyan = (r < 0.3 * b) & (g < 1.08 * b) & (g > 55) & (b <= 1.8 * g)
    parcel = orange if job == "a" else green
    receiver = magenta if job == "a" else blue
    gray = np.mean(rgb, axis=-1).astype(np.uint8)
    canonical = np.repeat(gray[..., None], 3, axis=-1)
    canonical[cyan] = rgb[cyan]
    canonical[parcel] = [220, 120, 20]
    canonical[receiver] = [170, 10, 200]
    return canonical, {"parcel": parcel, "receiver": receiver, "localizer": cyan}


class JobColorRenderer:
    """Normalize only RGB; retain original RGB and delivered depth for evidence."""

    def __init__(self, renderer, job):
        if job not in ("a", "b"):
            raise ValueError("job must be 'a' or 'b'")
        self.renderer, self.job = renderer, job
        self.depth = False
        self.raw_rgb = self.raw_depth = self.masks = None

    def __getattr__(self, name):
        return getattr(self.renderer, name)

    def enable_depth_rendering(self):
        self.depth = True
        self.renderer.enable_depth_rendering()

    def disable_depth_rendering(self):
        self.depth = False
        self.renderer.disable_depth_rendering()

    def render(self):
        frame = self.renderer.render()
        if self.depth:
            self.raw_depth = frame.copy()
            return frame
        self.raw_rgb = frame.copy()
        canonical, self.masks = select_job_colors(frame, self.job)
        return canonical


class JobSensorBelief(SharedSensorBelief):
    """One independent posterior for one visually identified parcel/receiver."""

    def __init__(self, local_model, shared_model, prefix, offset, *, job="a", **kwargs):
        if job not in ("a", "b"):
            raise ValueError("job must be 'a' or 'b'")
        super().__init__(local_model, shared_model, prefix, offset, **kwargs)
        self.job = job
        # Fixed surveyed workcell calibration, matching SECOND_JOB_OFFSET.
        # This never follows a robot, object, detection, or simulator pose.
        self.fixed_job_camera_offset = np.array([0.0, 1.0 if job == "b" else 0.0, 0.0])
        self.camera.lookat[:] += self.fixed_job_camera_offset
        self.renderer = JobColorRenderer(self.renderer, job)

    @property
    def raw_rgb(self):
        return self.renderer.raw_rgb

    @property
    def raw_depth(self):
        return self.renderer.raw_depth

    @property
    def color_masks(self):
        return self.renderer.masks

    def _camera_measurement(self, data):
        had_first = self.first_rgb is not None
        result = super()._camera_measurement(data)
        # The base detector sees normalized pixels, but all saved/viewable RGB
        # must remain the original shared-scene camera image.
        self.latest_rgb = self.raw_rgb.copy()
        if not had_first and self.first_rgb is not None:
            self.first_rgb = self.raw_rgb.copy()
        return result

    def save_evidence(self, directory):
        super().save_evidence(directory)
        path = Path(directory) / "sensor-contract.json"
        contract = json.loads(path.read_text())
        contract["job"] = self.job
        contract["fixed_job_camera_offset_xyz_m"] = (
            self.fixed_job_camera_offset.tolist()
        )
        contract["camera_mount"] = (
            "Fixed calibrated job view; no robot or target tracking"
        )
        contract["color_normalization"] = {
            "parcel": "orange" if self.job == "a" else "green",
            "receiver": "magenta" if self.job == "a" else "blue",
            "detector_encoding": "Selected parcel -> orange; selected receiver -> magenta; cyan localization pixels preserved",
            "evidence": "Original RGB; delivered RGB-D depth unchanged by color normalization",
            "limitations": "Calibrated color classes; no learned semantic instance recognition",
        }
        path.write_text(json.dumps(contract, indent=2) + "\n")
