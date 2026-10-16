"""Render full sensor-grounded handling traces with an unobscured workcell view.

Uses the shared recorded-state loader and warehouse presentation scene. No
simulation steps or invented phase transitions are generated during rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from warehouse_render import camera, font, load

BG = "#10191e"
WHITE = "#edf3fb"
GREEN = "#78edce"
MAGENTA = "#e894ef"
MUTED = "#acb7bf"


def phase(record):
    state = record.get("controller_state") or {}
    return state.get("phase") or record.get("stage", "replay")


def chapters(records):
    result = []
    for record in records:
        name = phase(record)
        if not result or result[-1]["phase"] != name:
            if result:
                result[-1]["end_s"] = record["t"]
            result.append({"phase": name, "start_s": record["t"]})
    result[-1]["end_s"] = records[-1]["t"] + 0.02
    return result


def display_chapters(exact):
    names = {
        "walk": "Walk",
        "settle": "Walk",
        "approach": "Reach",
        "align": "Reach",
        "descend": "Reach",
        "close": "Grasp",
        "lift": "Lift",
        "transport": "Carry",
        "locate_destination": "Locate tray",
        "lower": "Lower",
        "place": "Place",
        "open": "Release",
        "withdraw": "Retract",
    }
    result = []
    for event in exact:
        label = names.get(event["phase"], event["phase"].replace("_", " ").title())
        if result and result[-1]["phase"] == label:
            result[-1]["end_s"] = event["end_s"]
        else:
            result.append(
                {"phase": label, "start_s": event["start_s"], "end_s": event["end_s"]}
            )
    return result


def project(scene, point, width, height):
    """Project a trace estimate through the actual presentation camera."""
    cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    delta = np.asarray(point) - cam.pos
    depth = np.dot(delta, cam.forward)
    if depth <= 0:
        return None
    right = np.cross(cam.forward, cam.up)
    scale = 1 if cam.orthographic else cam.frustum_near / depth
    x, y = np.dot(delta, right) * scale, np.dot(delta, cam.up) * scale
    fh = cam.frustum_top - cam.frustum_bottom
    fw = cam.frustum_width or fh * width / height
    px = width * (0.5 + (x - cam.frustum_center) / fw)
    py = height * (1 - (y - cam.frustum_bottom) / fh)
    if 0 < px < width and 0 < py < height:
        return px, py
    return None


def estimate_marker(draw, scene, point, label, color, width, height):
    if point is None or len(point) != 3 or not np.isfinite(point).all():
        return
    pixel = project(scene, point, width, height)
    if pixel is None:
        return
    x, y = pixel[0], pixel[1] + 56
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=2)
    tx, ty = (
        (min(width - 150, x + 20), y + 18)
        if label.startswith("Tray")
        else (max(10, x - 154), y - 45)
    )
    edge = tx if label.startswith("Tray") else tx + 140
    draw.line((x - 7, y - 7, edge, ty + 14), fill=color, width=1)
    draw.rounded_rectangle((tx, ty, tx + 140, ty + 25), radius=4, fill=BG)
    draw.text((tx + 7, ty + 5), label, font=font(13, True), fill=color)


def render(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, data, qpos, records = load(
        args.run, args.assets, args.output.parent / (args.output.stem + "-scene")
    )
    exact_timeline = chapters(records)
    timeline = display_chapters(exact_timeline)
    full = mujoco.Renderer(model, height=816, width=1088)
    detail = mujoco.Renderer(model, height=336, width=448)
    wide = camera(3.15, (0.65, -0.28, 0.70), azimuth=90, elevation=-24)
    pickup = np.asarray(records[0]["object"], dtype=float)
    receiving = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "receiving_marker")
    if receiving >= 0:
        data.qpos[:] = qpos[0]
        mujoco.mj_forward(model, data)
        destination = data.geom_xpos[receiving]
        focus = (pickup + destination) / 2
    else:
        focus = pickup
    close = camera(0.78, focus + [0, 0, 0.07], azimuth=110, elevation=-28)
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.geomgroup[4] = 1
    options.sitegroup[:] = 0
    sensor_path = args.run / "sensor-rgb.png"
    initial_rgb = (
        Image.open(sensor_path).convert("RGB").resize((448, 336))
        if sensor_path.exists()
        else None
    )
    indices = (
        [min(len(records) - 1, round(args.preview * 50))]
        if args.preview is not None
        else range(0, len(records), args.stride)
    )
    writer = (
        None
        if args.preview is not None
        else iio.get_writer(
            args.output,
            fps=50 / args.stride,
            codec="libx264",
            quality=8,
            macro_block_size=16,
            ffmpeg_params=["-threads", "2"],
            ffmpeg_log_level="error",
        )
    )
    try:
        for index in indices:
            rec = records[index]
            data.qpos[:] = qpos[index]
            mujoco.mj_forward(model, data)
            full.update_scene(data, wide, scene_option=options)
            detail.update_scene(data, close, scene_option=options)
            canvas = Image.new("RGB", (1536, 1024), BG)
            canvas.paste(Image.fromarray(full.render()), (0, 56))
            canvas.paste(Image.fromarray(detail.render()), (1088, 56))
            if initial_rgb is not None:
                canvas.paste(initial_rgb, (1088, 408))
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 15), args.title, font=font(25, True), fill=WHITE)
            draw.text((1340, 17), f"{rec['t']:05.1f} s / 1×", font=font(22), fill=GREEN)
            draw.rectangle((1088, 56, 1536, 87), fill=BG)
            draw.text(
                (1102, 63), "HAND + RECEIVING WORKCELL", font=font(16, True), fill=GREEN
            )
            draw.rectangle((1088, 408, 1536, 439), fill=BG)
            draw.text(
                (1102, 415),
                "INITIAL RGB-D CAMERA · CAPTURED FRAME",
                font=font(14, True),
                fill=MUTED,
            )
            state = rec.get("sensor_state") or {}
            if state.get("camera_valid"):
                estimate_marker(
                    draw,
                    full.scene,
                    state.get("object_position"),
                    "Object estimate",
                    GREEN,
                    1088,
                    816,
                )
            if state.get("destination_position") is not None:
                estimate_marker(
                    draw,
                    full.scene,
                    state.get("destination_position"),
                    "Tray estimate",
                    MAGENTA,
                    1088,
                    816,
                )
            draw.rounded_rectangle((20, 76, 340, 120), radius=6, fill=BG)
            draw.text(
                (33, 87),
                phase(rec).replace("_", " ").upper(),
                font=font(20, True),
                fill=GREEN,
            )
            pad = state.get("pad_force_n", [])
            if isinstance(pad, dict):
                pad = list(pad.values())
            forces = " / ".join(f"{float(v):.1f}" for v in pad)
            draw.text(
                (1102, 762), "MEASURED PAD FORCE", font=font(14, True), fill=MUTED
            )
            draw.text(
                (1102, 788),
                (forces + " N") if forces else "not recorded",
                font=font(23, True),
                fill=GREEN,
            )
            age = state.get(
                "destination_measurement_age_s", state.get("destination_age_s")
            )
            controller = rec.get("controller_state") or {}
            payload = controller.get("payload_mass_estimate_kg")
            if payload is not None:
                draw.text(
                    (1102, 826),
                    f"Estimated payload: {float(payload):.3f} kg",
                    font=font(16),
                    fill=GREEN,
                )
            if age is not None:
                draw.text(
                    (1102, 851 if payload is not None else 830),
                    f"Tray observation age: {float(age):.2f}s",
                    font=font(14),
                    fill=MUTED,
                )
            active = next(
                (
                    i
                    for i, s in enumerate(timeline)
                    if s["start_s"] <= rec["t"] < s["end_s"]
                ),
                len(timeline) - 1,
            )
            visible = (
                timeline
                if len(timeline) <= 12
                else timeline[max(0, active - 5) : max(0, active - 5) + 12]
            )
            card = 1488 / max(len(visible), 1)
            for j, segment in enumerate(visible):
                x = 24 + j * card
                color = GREEN if segment["start_s"] <= rec["t"] else "#53636b"
                draw.rectangle((x, 893, x + card - 7, 897), fill=color)
                draw.text(
                    (x, 909),
                    segment["phase"].replace("_", " ").upper(),
                    font=font(min(16, max(11, int(card / 9))), True),
                    fill=color,
                )
                draw.text(
                    (x, 938), f"{segment['start_s']:.1f}s", font=font(14), fill=MUTED
                )
            draw.text(
                (24, 986),
                "Recorded physics · sensor estimates shown · background racks are scenery only",
                font=font(17),
                fill=MUTED,
            )
            if writer:
                writer.append_data(np.asarray(canvas))
            if (
                args.preview is not None
                or index == 0
                or abs(index / 50 - args.poster_time) < 0.025
            ):
                canvas.save(args.output.with_suffix(".png"))
            if index % 100 == 0:
                print(f"rendered {index}/{len(records)}", flush=True)
    finally:
        if writer:
            writer.close()
        full.close()
        detail.close()
    probe_path = args.run / "probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "run": str(args.run),
                "recorded_probe_success": probe.get("success"),
                "source_probe_sha256": hashlib.sha256(
                    probe_path.read_bytes()
                ).hexdigest()
                if probe_path.exists()
                else None,
                "state_sha256": hashlib.sha256(
                    (args.run / "probe.npz").read_bytes()
                ).hexdigest(),
                "records_sha256": hashlib.sha256(
                    (args.run / "records.json").read_bytes()
                ).hexdigest(),
                "scene_sha256": hashlib.sha256(
                    (args.run / "scene.xml").read_bytes()
                ).hexdigest(),
                "sensor_image_sha256": hashlib.sha256(
                    sensor_path.read_bytes()
                ).hexdigest()
                if sensor_path.exists()
                else None,
                "fps": 50 / args.stride,
                "source_rate_hz": 50,
                "video_start_s": 0,
                "video_dimensions": [1536, 1024],
                "video_duration_s": len(range(0, len(records), args.stride))
                / (50 / args.stride)
                if args.preview is None
                else None,
                "recorded_frames": len(records),
                "recorded_end_s": records[-1]["t"],
                "chapters": exact_timeline,
                "display_chapters": timeline,
                "state_modified": False,
                "warehouse_props": "Presentation-only noncolliding background; not supplied to sensor or controller",
                "overlay_sources": "controller_state.phase and payload_mass_estimate_kg; sensor_state.object_position, destination_position, pad_force_n, destination_measurement_age_s",
                "camera_image": "Actual initial saved sensor-rgb.png; not regenerated with warehouse decorations",
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--title", default="Humanoid handling / physical pick, transfer and place"
    )
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--poster-time", type=float, default=19)
    p.add_argument("--preview", type=float)
    args = p.parse_args()
    if args.stride < 1:
        p.error("--stride must be positive")
    render(args)


if __name__ == "__main__":
    main()
