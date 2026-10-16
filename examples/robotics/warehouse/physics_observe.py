"""Capture fresh final-frame RGB-D hypotheses from an exact shared physics replay.

The replay loads recorded configurations solely to render pixels. The geometric
perception function receives only RGB, axial depth and camera calibration.
No object/body pose, segmentation identity or contact label enters perception.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warehouse.perception import observe_rgbd
from warehouse.representation import Entity, SceneGraph


def capture(run: Path):
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    data = mujoco.MjData(model)
    qpos = np.load(run / "probe.npz")["qpos"]
    data.qpos[:] = qpos[-1]
    mujoco.mj_forward(model, data)
    report = json.loads((run / "probe.json").read_text())
    frame_time = json.loads((run / "records.json").read_text())[-1]["t"]
    result = {
        "frame": "final",
        "timestamp": frame_time,
        "processing_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("perception.py"),
                Path(__file__).with_name("representation.py"),
            )
        },
        "source_qpos_sha256": hashlib.sha256(
            (run / "probe.npz").read_bytes()
        ).hexdigest(),
        "cameras": {},
    }
    width, height = 320, 240
    # Freeze the exact observation revision used to ground the task DAG.
    controller_scene = json.loads((run / "scene-graph.json").read_text())
    cutoff = min((task["scene_revision"] for task in report["tasks"]), default=0)
    initial = SceneGraph()
    for entry in controller_scene["history"]:
        if entry["revision"] <= cutoff:
            initial.upsert(Entity(**entry["observation"]))
    (run / "initial-scene-graph.json").write_text(
        json.dumps(initial.to_dict(), indent=2)
    )
    final_scene = SceneGraph()
    result["provenance"] = (
        "Fresh offline RGB-D observation from final physical replay; not a controller result"
    )
    result["identity_association"] = (
        "One unique color/shape hypothesis of each kind per calibrated camera-owned workcell; ambiguous multiplicity is omitted"
    )
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        for index in range(report["robots"]):
            # Fixed camera extrinsics are an initial calibration prior, not a
            # function of any live object/robot pose.
            root = model.jnt_qposadr[model.joint(f"r{index}/floating_base_joint").id]
            offset = model.qpos0[root : root + 3].copy()
            offset[2] = 0
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = np.array([1.1, -0.25, 0.65]) + offset
            camera.distance = 2.5
            camera.azimuth = 90
            camera.elevation = -78
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
            views = renderer.scene.camera
            position = np.mean([np.asarray(c.pos) for c in views], axis=0)
            forward = np.mean([np.asarray(c.forward) for c in views], axis=0)
            forward /= np.linalg.norm(forward)
            up = np.mean([np.asarray(c.up) for c in views], axis=0)
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            calibration = {
                "position": position.tolist(),
                "forward": forward.tolist(),
                "up": up.tolist(),
                "right": right.tolist(),
                "focal_pixels": float(
                    0.5 * height / np.tan(np.deg2rad(model.vis.global_.fovy) * 0.5)
                ),
                "width": width,
                "height": height,
            }
            target = run / f"r{index}"
            target.mkdir(exist_ok=True)
            Image.fromarray(rgb).save(target / "final-rgb.png")
            np.savez_compressed(target / "final-depth.npz", depth=depth)
            (target / "final-calibration.json").write_text(
                json.dumps(calibration, indent=2)
            )
            hypotheses = observe_rgbd(rgb, depth, calibration, frame_time)
            for kind in ("parcel", "receiver"):
                candidates = [h for h in hypotheses if h.kind == kind]
                if len(candidates) != 1:
                    continue
                h = candidates[0]
                final_scene.upsert(
                    Entity(
                        f"{kind}_{index}",
                        "parcel" if kind == "parcel" else "station",
                        {"color": h.color, "lane": str(index)},
                        h.center_xyz if kind == "parcel" else h.top_center_xyz,
                        h.timestamp,
                        f"final-perception.json#r{index}/{h.observation_id}",
                        h.confidence_score,
                        max(h.position_ambiguity_m),
                    )
                )
            result["cameras"][f"r{index}"] = {
                "rgb": f"r{index}/final-rgb.png",
                "depth": f"r{index}/final-depth.npz",
                "calibration": calibration,
                "hypotheses": [h.to_dict() for h in hypotheses],
            }
    finally:
        renderer.close()
    (run / "final-scene-graph.json").write_text(
        json.dumps(final_scene.to_dict(), indent=2)
    )
    (run / "final-perception.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = capture(args.run)
    print(json.dumps({k: len(v["hypotheses"]) for k, v in result["cameras"].items()}))


if __name__ == "__main__":
    main()
