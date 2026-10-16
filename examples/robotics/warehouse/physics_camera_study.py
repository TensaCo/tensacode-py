"""Offline static RGB-D calibration study; never used by a robot controller.

Run separate EGL and software Mesa processes against identical scene/probe files.
Image reconstruction uses pixels and camera calibration only. Simulator geometry
is accessed separately by ``score_planes`` solely for offline error evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import binary_erosion


def summary(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "max_abs_m": float(np.max(np.abs(values))),
        "p95_abs_m": float(np.quantile(np.abs(values), 0.95)),
    }


def calibration(renderer, model, width, height):
    cameras = renderer.scene.camera
    position = np.mean([c.pos for c in cameras], axis=0)
    forward = np.mean([c.forward for c in cameras], axis=0)
    forward /= np.linalg.norm(forward)
    up = np.mean([c.up for c in cameras], axis=0)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal = 0.5 * height / np.tan(np.deg2rad(model.vis.global_.fovy) / 2)
    return {
        "position": position.tolist(),
        "forward": forward.tolist(),
        "up": up.tolist(),
        "right": right.tolist(),
        "focal_pixels": float(focal),
        "width": width,
        "height": height,
    }


def reconstruct(depth, camera):
    """Pixel-center backprojection, identical camera convention to SensorBelief."""
    rows, cols = np.indices(depth.shape)
    x = (cols + 0.5 - camera["width"] / 2) / camera["focal_pixels"]
    y = (camera["height"] / 2 - rows - 0.5) / camera["focal_pixels"]
    rays = (
        np.asarray(camera["forward"])
        + x[..., None] * camera["right"]
        + y[..., None] * camera["up"]
    )
    return np.asarray(camera["position"]) + depth[..., None] * rays, rays


def image_cube_estimate(rgb, depth, points):
    """Existing orange color/top-quantile heuristic; no scene labels supplied."""
    r, g, b = np.moveaxis(rgb.astype(float), -1, 0)
    mask = (
        (r > 65)
        & (r > 0.92 * g)
        & (g > 1.8 * b)
        & (g > 25)
        & np.isfinite(depth)
        & (depth > 0)
    )
    cloud = points[mask]
    if len(cloud) < 6:
        return None, mask
    top_height = np.quantile(cloud[:, 2], 0.8)
    top = cloud[np.abs(cloud[:, 2] - top_height) < 0.006]
    if len(top) < 4:
        return None, mask
    estimate = np.median(top, axis=0)
    estimate[2] = top_height - 0.025
    return estimate, mask


def score_planes(model, data, prefix, depth, segmentation, points, rays, camera):
    """Offline scorer: ground-truth planar surfaces never enter reconstruction."""
    scores = {}
    for name in (prefix + "cube", "floor", prefix + "receiving_marker"):
        geom = model.geom(name).id
        mask = (segmentation[..., 0] == geom) & (
            segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        )
        mask = binary_erosion(mask, iterations=2)
        center = data.geom_xpos[geom]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        normal = rotation[:, 2]
        plane_point = (
            center if name == "floor" else center + normal * model.geom_size[geom, 2]
        )
        position = np.asarray(camera["position"])
        denominator = rays @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_depth = ((plane_point - position) @ normal) / denominator
        expected_points = position + expected_depth[..., None] * rays
        if name != "floor":
            local = (expected_points - center) @ rotation
            mask &= np.all(
                np.abs(local[..., :2]) < model.geom_size[geom, :2] - 0.002, axis=-1
            )
        else:
            # Keep a fixed nearby floor region, not pixels near the far plane.
            mask &= (
                np.linalg.norm(expected_points[..., :2] - position[:2], axis=-1) < 3.0
            )
        mask &= np.isfinite(expected_depth) & (expected_depth > 0)
        scores[name] = {
            "depth_error": summary((depth - expected_depth)[mask]),
            "plane_normal_error": summary(((points - plane_point) @ normal)[mask]),
            "plane_point": plane_point.tolist(),
            "plane_normal": normal.tolist(),
            "offline_mask": "geometry segmentation, eroded two pixels, top face interior",
        }
    return scores


def gl_metadata(renderer):
    from OpenGL import GL
    from OpenGL.error import GLError

    result = {
        key: GL.glGetString(value).decode()
        for key, value in (
            ("vendor", GL.GL_VENDOR),
            ("renderer", GL.GL_RENDERER),
            ("version", GL.GL_VERSION),
        )
    }
    previous = int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING))
    try:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, renderer._mjr_context.offFBO)
        result["depth_bits"] = int(GL.glGetIntegerv(GL.GL_DEPTH_BITS))
        result["depth_component_type"] = int(
            GL.glGetFramebufferAttachmentParameteriv(
                GL.GL_FRAMEBUFFER,
                GL.GL_DEPTH_ATTACHMENT,
                GL.GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE,
            )
        )
        result["depth_attachment_object_type"] = int(
            GL.glGetFramebufferAttachmentParameteriv(
                GL.GL_FRAMEBUFFER,
                GL.GL_DEPTH_ATTACHMENT,
                GL.GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,
            )
        )
    except (GLError, AttributeError, TypeError) as error:
        result["depth_query_error"] = str(error)
    finally:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, previous)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    scene_path = args.run / "scene.xml"
    probe_path = args.run / "probe.npz"
    qpos = np.load(probe_path)["qpos"][args.frame]
    result = {
        "label": args.label,
        "mujoco": mujoco.__version__,
        "backend_environment": {
            key: os.environ.get(key)
            for key in (
                "MUJOCO_GL",
                "LIBGL_ALWAYS_SOFTWARE",
                "__GLX_VENDOR_LIBRARY_NAME",
            )
        },
        "frame": args.frame,
        "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_only_truth": True,
        "captures": [],
    }
    for clipping in ("original", "bounded"):
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        if args.offscreen_samples is not None:
            model.vis.quality.offsamples = args.offscreen_samples
        if clipping == "bounded":
            model.vis.map.znear = 0.25 / model.stat.extent
            model.vis.map.zfar = 15.0 / model.stat.extent
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=240, width=320) as renderer:
            for lane in (0, 1):
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = [1.1, -0.25 + lane * 1.6, 0.65]
                camera.distance, camera.azimuth, camera.elevation = 2.5, 90, -78
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render().copy()
                camera_info = calibration(renderer, model, 320, 240)
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                points, rays = reconstruct(depth, camera_info)
                estimate, mask = image_cube_estimate(rgb, depth, points)
                prefix = f"r{lane}/"
                truth = data.geom_xpos[model.geom(prefix + "cube").id]
                filename = f"{clipping}-r{lane}.npz"
                np.savez_compressed(
                    args.output / filename,
                    rgb=rgb,
                    depth=depth,
                    segmentation=segmentation,
                    orange_mask=mask,
                )
                result["captures"].append(
                    {
                        "clipping": clipping,
                        "offscreen_samples": int(model.vis.quality.offsamples),
                        "lane": lane,
                        "camera": camera_info,
                        "near_m": float(model.vis.map.znear * model.stat.extent),
                        "far_m": float(model.vis.map.zfar * model.stat.extent),
                        "gl": gl_metadata(renderer),
                        "array_file": filename,
                        "image_cube_estimate": None
                        if estimate is None
                        else estimate.tolist(),
                        "offline_cube_error_m": None
                        if estimate is None
                        else (estimate - truth).tolist(),
                        "planes": score_planes(
                            model,
                            data,
                            prefix,
                            depth,
                            segmentation,
                            points,
                            rays,
                            camera_info,
                        ),
                    }
                )
    (args.output / "study.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output / "study.json"),
                "captures": len(result["captures"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--offscreen-samples", type=int, choices=(0, 1, 2, 4, 8), default=None
    )
    run(parser.parse_args())
