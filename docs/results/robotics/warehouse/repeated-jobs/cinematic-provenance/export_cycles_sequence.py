"""Export exact recorded geometry poses for a persistent Cycles animation.

Frames select nearest recorded states; no pose or object motion is interpolated.
This command only exports data. It does not start Blender or GPU computation.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from render import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transfer"))
from warehouse_photoreal import geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--start", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--segment", nargs=3, action="append", metavar=("START", "DURATION", "LABEL")
    )
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    if args.segment and (args.start is not None or args.duration is not None):
        parser.error("Use either segments or start/duration")
    if not args.segment and (args.start is None or args.duration is None):
        parser.error("Provide start/duration or explicit segments")
    segments = args.segment or [(args.start, args.duration, "Recorded replay")]
    if args.fps <= 0 or any(float(item[1]) <= 0 for item in segments):
        parser.error("Duration and fps must be positive")
    model, data, qpos, records, raw = load(args.run, args.assets, args.output)
    times = np.asarray([r["t"] for r in records])
    chapters = []
    samples = []
    for start, duration, label in segments:
        start, duration = float(start), float(duration)
        values = start + np.arange(round(duration * args.fps)) / args.fps
        chapters.append(
            {
                "label": label,
                "first_frame": sum(len(v) for v in samples),
                "frame_count": len(values),
                "source_start_s": start,
                "duration_s": len(values) / args.fps,
            }
        )
        samples.append(values)
    requested = np.concatenate(samples)
    if requested.min() < times[0] or requested.max() > times[-1]:
        parser.error("Requested interval must fit entirely within recorded data")
    indices = np.asarray([int(np.argmin(abs(times - t))) for t in requested])
    visible = np.flatnonzero(model.geom_group != 3)
    positions = np.empty((len(indices), len(visible), 3), dtype=np.float64)
    rotations = np.empty((len(indices), len(visible), 3, 3), dtype=np.float64)
    data.qpos[:] = qpos[indices[0]]
    mujoco.mj_forward(model, data)
    geoms = geometry(model, data)
    assert len(geoms) == len(visible)
    for frame, index in enumerate(indices):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        positions[frame] = data.geom_xpos[visible]
        rotations[frame] = data.geom_xmat[visible].reshape(-1, 3, 3)
    (args.output / "geometry.json").write_text(json.dumps(geoms))
    np.savez_compressed(
        args.output / "poses.npz",
        positions=positions,
        rotations=rotations,
        source_indices=indices,
        source_times=times[indices],
    )
    metadata = {
        "source_run": str(args.run),
        "source_hashes": {
            name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
            for name in ("scene.xml", "probe.npz", "probe.json")
        },
        "records_sha256": hashlib.sha256(raw).hexdigest(),
        "geometry_sha256": hashlib.sha256(
            (args.output / "geometry.json").read_bytes()
        ).hexdigest(),
        "poses_sha256": hashlib.sha256(
            (args.output / "poses.npz").read_bytes()
        ).hexdigest(),
        "fps": args.fps,
        "frame_count": len(indices),
        "video_duration_s": len(indices) / args.fps,
        "requested_start_s": float(requested[0]),
        "chapters": chapters,
        "explicit_cuts": len(chapters) > 1,
        "source_start_s": float(times[indices[0]]),
        "source_end_s": float(times[indices[-1]]),
        "maximum_sampling_error_s": float(np.max(abs(times[indices] - requested))),
        "state_interpolation": False,
        "physics_steps_generated": 0,
        "scene_composition": "One recorded shared scene, including all robots and task objects",
    }
    (args.output / "sequence.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
