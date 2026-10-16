"""Evaluation-only projected amodal parcel boxes from recorded simulator poses.

These labels must never enter sensor inference or controller observation. They
cover task parcels only, not every warehouse box; they cannot establish detector
precision over all objects. Occluded portions remain in each amodal rectangle.
"""

import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np


def labels(run, dataset):
    model = mujoco.MjModel.from_xml_path(str(run / "scene.xml"))
    data = mujoco.MjData(model)
    qpos = np.load(run / "probe.npz")["qpos"]
    manifest = json.loads((dataset / "manifest.json").read_text())
    result = {
        "scope": "evaluation_only_partial_amodal_parcel_boxes; not all scene boxes; no occlusion correction",
        "frames": [],
    }
    gids = [
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(
            ("/cube", "/cube_b")
        )
    ]
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    for frame in manifest["frames"]:
        data.qpos[:] = qpos[frame["frame_index"]]
        mujoco.mj_forward(model, data)
        calibration = json.loads(
            (dataset / frame["path"] / "sensor-contract.json").read_text()
        )["camera"]
        origin = np.array(calibration["position"])
        forward = np.array(calibration["forward"])
        right = np.array(calibration["right"])
        up = np.array(calibration["up"])
        w, h = calibration["width"], calibration["height"]
        focal = calibration["focal_pixels"]
        boxes = []
        for gid in gids:
            points = (signs * model.geom_size[gid]) @ data.geom_xmat[gid].reshape(
                3, 3
            ).T + data.geom_xpos[gid]
            relative = points - origin
            z = relative @ forward
            if np.any(z <= 0):
                continue
            u = w / 2 + focal * (relative @ right) / z
            v = h / 2 - focal * (relative @ up) / z
            bbox = [
                max(0.0, float(u.min())),
                max(0.0, float(v.min())),
                min(float(w), float(u.max())),
                min(float(h), float(v.max())),
            ]
            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                boxes.append(
                    {
                        "evaluation_geom": mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, gid
                        ),
                        "class": "box",
                        "bbox_xyxy": bbox,
                        "amodal": True,
                    }
                )
        result["frames"].append(
            {"path": frame["path"], "frame_index": frame["frame_index"], "boxes": boxes}
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(labels(args.run, args.dataset), indent=2) + "\n")
