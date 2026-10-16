"""Render recorded physics into RGB-D; truth is written separately for scoring."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def capture(run, output, split, frames=13):
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    qpos = np.load(run / "probe.npz")["qpos"]
    report = json.loads((run / "probe.json").read_text())
    records_path = run / "records.json.gz"
    records = (
        json.loads(gzip.decompress(records_path.read_bytes()))
        if records_path.exists()
        else json.loads((run / "records.json").read_text())
    )
    angles = (
        [(90, -78), (45, -60), (135, -60)]
        if split == "development"
        else [(90, -78), (65, -65), (115, -65)]
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run": run.name,
        "split": split,
        "source_qpos_sha256": hashlib.sha256(
            (run / "probe.npz").read_bytes()
        ).hexdigest(),
        "frames": [],
    }
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        for frame_index in np.linspace(0, len(qpos) - 1, frames).astype(int):
            data.qpos[:] = qpos[frame_index]
            mujoco.mj_forward(model, data)
            timestamp = float(records[frame_index]["t"])
            for lane in range(report["robots"]):
                root = model.jnt_qposadr[model.joint(f"r{lane}/floating_base_joint").id]
                offset = model.qpos0[root : root + 3].copy()
                offset[2] = 0
                truth = []
                for kind, geom in [
                    ("parcel", "cube"),
                    ("receiver", "receiving_marker"),
                ]:
                    gid = model.geom(f"r{lane}/{geom}").id
                    truth.append(
                        {
                            "id": f"{kind}_{lane}",
                            "kind": kind,
                            "center_xyz": data.geom_xpos[gid].tolist(),
                            "dimensions_xyz": (2 * model.geom_size[gid]).tolist(),
                            "rotation": data.geom_xmat[gid].reshape(3, 3).tolist(),
                        }
                    )
                for view, (azimuth, elevation) in enumerate(angles):
                    camera = mujoco.MjvCamera()
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:] = np.array([1.1, -0.25, 0.65]) + offset
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
                            }
                        )
                    )
                    (target / "evaluation_truth.json").write_text(
                        json.dumps({"entities": truth})
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
            print(
                json.dumps(
                    {"frame": int(frame_index), "captures": len(manifest["frames"])}
                ),
                flush=True,
            )
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=["development", "evaluation"], required=True)
    parser.add_argument("--frames", type=int, default=13)
    args = parser.parse_args()
    capture(args.run, args.output, args.split, args.frames)
