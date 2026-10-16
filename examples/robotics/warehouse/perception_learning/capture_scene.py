"""Capture initial/final RGB-D scene evidence from a physical replay, no pose labels.

Recorded robot/object configurations are used only by the renderer. The output
contains pixels, camera calibration and time, never evaluation object coordinates.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def capture(run, output, cell_shift_y=0.0, cell_halfwidth_m=0.7):
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    qpos = np.load(run / "probe.npz")["qpos"]
    records_path = run / "records.json.gz"
    records = (
        json.loads(gzip.decompress(records_path.read_bytes()))
        if records_path.exists()
        else json.loads((run / "records.json").read_text())
    )
    robot_count = sum(
        1
        for i in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").endswith(
            "/floating_base_joint"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run": run.name,
        "source_qpos_sha256": hashlib.sha256(
            (run / "probe.npz").read_bytes()
        ).hexdigest(),
        "source_scene_sha256": hashlib.sha256(
            (run / "scene.xml").read_bytes()
        ).hexdigest(),
        "scope": "replayed RGB-D observation; no evaluation object poses in this dataset",
        "frames": [],
    }
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        for frame_index in sorted({0, len(qpos) - 1}):
            data.qpos[:] = qpos[frame_index]
            mujoco.mj_forward(model, data)
            timestamp = float(records[frame_index]["t"])
            for lane in range(robot_count):
                root = model.jnt_qposadr[model.joint(f"r{lane}/floating_base_joint").id]
                offset = model.qpos0[root : root + 3].copy()
                offset[2] = 0
                for view, (azimuth, elevation) in enumerate(
                    [(90, -78), (65, -65), (115, -65)]
                ):
                    camera = mujoco.MjvCamera()
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = (
                        np.array([1.1, -0.25 + cell_shift_y, 0.65]) + offset
                    )
                    camera.distance, camera.azimuth, camera.elevation = (
                        2.5,
                        azimuth,
                        elevation,
                    )
                    renderer.disable_depth_rendering()
                    renderer.update_scene(data, camera=camera)
                    rgb = renderer.render().copy()
                    cams = renderer.scene.camera
                    position = np.mean([c.pos for c in cams], axis=0)
                    forward = np.mean([c.forward for c in cams], axis=0)
                    forward /= np.linalg.norm(forward)
                    up = np.mean([c.up for c in cams], axis=0)
                    right = np.cross(forward, up)
                    right /= np.linalg.norm(right)
                    up = np.cross(right, forward)
                    calibration = {
                        "position": position.tolist(),
                        "forward": forward.tolist(),
                        "up": up.tolist(),
                        "right": right.tolist(),
                        "focal_pixels": float(
                            240 / np.tan(np.deg2rad(model.vis.global_.fovy) / 2)
                        ),
                        "width": 640,
                        "height": 480,
                    }
                    renderer.enable_depth_rendering()
                    depth = renderer.render().copy()
                    target = output / f"frame-{frame_index:05d}-r{lane}-v{view}"
                    target.mkdir(exist_ok=True)
                    Image.fromarray(rgb).save(target / "sensor-rgb.png")
                    np.savez_compressed(target / "sensor-depth.npz", depth=depth)
                    (target / "sensor-contract.json").write_text(
                        json.dumps(
                            {
                                "camera": calibration,
                                "capture_timestamp": timestamp,
                                "lane": lane,
                                "view": view,
                                "workcell_center_xy": camera.lookat[:2].tolist(),
                                "workcell_halfwidth_m": cell_halfwidth_m,
                            }
                        )
                    )
                    manifest["frames"].append(
                        {
                            "path": target.name,
                            "frame_index": int(frame_index),
                            "timestamp": timestamp,
                            "lane": lane,
                            "view": view,
                        }
                    )
    finally:
        renderer.close()
    manifest["capture_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cell-shift-y", type=float, default=0.0)
    parser.add_argument("--cell-halfwidth-m", type=float, default=0.7)
    args = parser.parse_args()
    capture(args.run, args.output, args.cell_shift_y, args.cell_halfwidth_m)
