"""Render one shared-scene multi-humanoid trace, without stepping physics.

Run with --preview SECONDS to inspect framing before a full video. Camera and
source timing are explicit; no independent episodes are combined.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as iio
import mujoco
import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont
from render_scene import prepare_scene


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def projected_label(scene, point, width, height):
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    delta = np.asarray(point) - camera.pos
    depth = float(delta @ camera.forward)
    if depth <= 0:
        return None
    scale = 1 if camera.orthographic else camera.frustum_near / depth
    x = float(delta @ np.cross(camera.forward, camera.up)) * scale
    y = float(delta @ camera.up) * scale
    frustum_height = camera.frustum_top - camera.frustum_bottom
    frustum_width = camera.frustum_width or frustum_height * width / height
    px = width * (0.5 + (x - camera.frustum_center) / frustum_width)
    py = height * (1 - (y - camera.frustum_bottom) / frustum_height)
    return (px, py) if 0 < px < width and 0 < py < height else None


def load(run, assets, work):
    scene = prepare_scene(run / "scene.xml", work / "scene.xml", assets)
    model = mujoco.MjModel.from_xml_path(str(scene))
    archive = np.load(run / "probe.npz")
    qpos = archive["qpos"]
    path = run / "records.json"
    if not path.exists():
        path = run / "records.json.gz"
    raw = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    records = json.loads(raw)
    if qpos.shape != (len(records), model.nq):
        raise ValueError("Recorded state shape does not match the single shared model")
    times = np.asarray([record["t"] for record in records])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Replay requires increasing source timestamps")
    return model, mujoco.MjData(model), qpos, records, raw


def render(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, data, states, records, raw = load(
        args.run, args.assets, args.output.parent / (args.output.stem + "-scene")
    )
    dt = float(np.median(np.diff([x["t"] for x in records])))
    if not np.allclose(np.diff([x["t"] for x in records]), dt, atol=1e-6):
        raise ValueError("Variable-rate traces require explicit resampling")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = args.lookat
    camera.distance, camera.azimuth, camera.elevation = (
        args.distance,
        args.azimuth,
        args.elevation,
    )
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.geomgroup[3] = 0
    detail_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in args.detail_sites
    ]
    if any(index < 0 for index in detail_ids):
        raise ValueError("Unknown requested detail site")
    sidebar = 384 if detail_ids else 0
    renderer = mujoco.Renderer(
        model, width=args.width - sidebar, height=args.height - 96
    )
    detail_height = (args.height - 96) // max(1, len(detail_ids))
    details = [
        mujoco.Renderer(model, width=sidebar, height=detail_height) for _ in detail_ids
    ]
    detail_camera = mujoco.MjvCamera()
    detail_camera.distance, detail_camera.azimuth, detail_camera.elevation = (
        1.15,
        110,
        -25,
    )
    indices = (
        range(0, len(records), args.stride)
        if args.preview is None
        else [
            min(range(len(records)), key=lambda i: abs(records[i]["t"] - args.preview))
        ]
    )
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font, small = (
        ImageFont.truetype(font_path, 24),
        ImageFont.truetype(font_path, 16),
    )
    writer = (
        None
        if args.preview is not None
        else iio.get_writer(
            args.output,
            fps=1 / (dt * args.stride),
            codec="libx264",
            quality=8,
            macro_block_size=16,
            ffmpeg_params=["-threads", "2"],
        )
    )
    gl_backend = None
    poster = min(indices, key=lambda i: abs(records[i]["t"] - args.poster_time))
    try:
        for index in indices:
            frame_started = time.monotonic()
            data.qpos[:] = states[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera, scene_option=option)
            canvas = Image.new("RGB", (args.width, args.height), "#111b21")
            canvas.paste(Image.fromarray(renderer.render()), (0, 48))
            if gl_backend is None:
                gl_backend = {
                    "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
                    "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
                    "version": GL.glGetString(GL.GL_VERSION).decode(),
                }
            draw = ImageDraw.Draw(canvas)
            for slot, (detail, site_id) in enumerate(zip(details, detail_ids)):
                detail_camera.lookat[:] = data.site_xpos[site_id]
                detail.update_scene(data, detail_camera, scene_option=option)
                left, top = args.width - sidebar, 48 + slot * detail_height
                canvas.paste(Image.fromarray(detail.render()), (left, top))
                draw.rectangle((left, top, args.width, top + 28), fill="#14232b")
                draw.text(
                    (left + 12, top + 6),
                    args.detail_sites[slot].split("/")[0].upper()
                    + " · SAME SCENE DETAIL",
                    font=small,
                    fill="#7ee5c6",
                )
            draw.text((24, 12), args.title, font=title_font, fill="#e8f3f6")
            draw.text(
                (args.width - 180, 14),
                f"{records[index]['t']:.1f}s · 1×",
                font=title_font,
                fill="#7ee5c6",
            )
            draw.text(
                (24, args.height - 32),
                "One shared physics scene · recorded state replay · presentation lighting",
                font=small,
                fill="#afbdc4",
            )
            robots = records[index].get("robots", {})
            if isinstance(robots, dict):
                for slot, (name, state) in enumerate(sorted(robots.items())):
                    phase = state.get("phase", "recorded")
                    job = state.get("current_job")
                    identity = name.upper() + (f" / {str(job).upper()}" if job else "")
                    text = f"{identity}  {str(phase).replace('_', ' ')}"
                    left = 24 + slot * 280
                    draw.rounded_rectangle(
                        (left, 64, left + 264, 99), radius=5, fill="#14232b"
                    )
                    draw.text((left + 10, 73), text, font=small, fill="#d5e4ea")
                    if args.robot_labels:
                        body = mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_BODY, name + "/pelvis"
                        )
                        pixel = (
                            projected_label(
                                renderer.scene,
                                data.xpos[body] + [0, 0, 0.7],
                                args.width - sidebar,
                                args.height - 96,
                            )
                            if body >= 0
                            else None
                        )
                        if pixel:
                            x, y = pixel[0], pixel[1] + 48
                            draw.rounded_rectangle(
                                (x - 22, y - 12, x + 22, y + 12),
                                radius=5,
                                fill="#14232b",
                            )
                            draw.text(
                                (x - 12, y - 8),
                                name.upper(),
                                font=small,
                                fill="#7ee5c6",
                            )
            if writer:
                writer.append_data(np.asarray(canvas))
            if index == poster:
                canvas.save(args.output.with_suffix(".png"))
            if index % 100 == 0:
                print(f"rendered {index}/{len(records)}", flush=True)
            if args.max_render_fps:
                time.sleep(
                    max(0, 1 / args.max_render_fps - (time.monotonic() - frame_started))
                )
    finally:
        renderer.close()
        for detail in details:
            detail.close()
        if writer:
            writer.close()
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": str(args.run),
                "scene_sha256": digest(args.run / "scene.xml"),
                "state_sha256": digest(args.run / "probe.npz"),
                "records_sha256": hashlib.sha256(raw).hexdigest(),
                "shared_scene": True,
                "physics_steps_generated": 0,
                "state_modified": False,
                "presentation": "Lighting only; original scene geometry",
                "fps": 1 / (dt * args.stride),
                "video_start_s": records[0]["t"],
                "recorded_end_s": records[-1]["t"],
                "last_replayed_time_s": records[indices[-1]]["t"],
                "recorded_frames": len(records),
                "video_duration_s": len(indices) * dt * args.stride if writer else None,
                "camera": {
                    "lookat": args.lookat,
                    "distance": args.distance,
                    "azimuth": args.azimuth,
                    "elevation": args.elevation,
                },
                "detail_sites": args.detail_sites,
                "robot_labels": args.robot_labels,
                "max_render_fps": args.max_render_fps,
                "gl_backend": gl_backend,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Shared-scene humanoid warehouse")
    parser.add_argument("--lookat", type=float, nargs=3, default=[1, 0, 0.8])
    parser.add_argument("--distance", type=float, default=7)
    parser.add_argument("--azimuth", type=float, default=135)
    parser.add_argument("--elevation", type=float, default=-25)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--poster-time", type=float, default=30)
    parser.add_argument("--preview", type=float)
    parser.add_argument("--detail-sites", nargs="*", default=[])
    parser.add_argument("--robot-labels", action="store_true")
    parser.add_argument(
        "--max-render-fps",
        type=float,
        default=0,
        help="Wall-clock render pacing; does not change video playback speed",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    if len(args.detail_sites) > 3:
        parser.error("At most three detail cameras are supported")
    render(args)


if __name__ == "__main__":
    main()
