"""Synchronized comparison of matched shared-physics recovery experiments.

Each panel replays a separate, explicitly labeled trial. Within a panel every
robot and detail camera uses the same recorded full-model state. Different
timestamps are rejected rather than silently aligned or extended.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from render import (
    GL,
    Image,
    ImageDraw,
    ImageFont,
    digest,
    iio,
    load,
    mujoco,
    np,
    projected_label,
)


def camera(distance, lookat, azimuth=25, elevation=-20):
    result = mujoco.MjvCamera()
    result.distance, result.azimuth, result.elevation = distance, azimuth, elevation
    result.lookat[:] = lookat
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=float)
    parser.add_argument(
        "--labels",
        nargs=2,
        default=["Baseline / no contact recovery", "Recovery / bounded tactile retry"],
    )
    parser.add_argument("--poster-time", type=float, default=24)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-render-fps", type=float, default=15)
    args = parser.parse_args()
    if args.stride < 1 or args.max_render_fps <= 0:
        parser.error("Stride and pacing must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [args.baseline, args.recovery]
    probes = [json.loads((path / "probe.json").read_text()) for path in paths]
    outcomes = [
        f"{sum(bool(v['physical_success']) for v in probe['evaluation'].values())} / {len(probe['evaluation'])} placed"
        for probe in probes
    ]
    states = [
        load(path, args.assets, args.output.parent / f"{args.output.stem}-scene-{i}")
        for i, path in enumerate(paths)
    ]
    times = np.asarray([r["t"] for r in states[0][3]])
    other = np.asarray([r["t"] for r in states[1][3]])
    if times.shape != other.shape or not np.allclose(times, other, atol=1e-9, rtol=0):
        raise ValueError("Comparison requires matching recorded timestamps")
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, atol=1e-6, rtol=0):
        raise ValueError("Comparison requires uniform source timing")
    indices = (
        range(0, len(times), args.stride)
        if args.preview is None
        else [int(np.argmin(abs(times - args.preview)))]
    )
    poster = min(indices, key=lambda i: abs(times[i] - args.poster_time))
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    wide = [mujoco.Renderer(s[0], width=960, height=600) for s in states]
    close = [mujoco.Renderer(s[0], width=960, height=360) for s in states]
    cameras = [camera(5.3, (1, 0.7, 0.95)) for _ in states]
    details = [camera(0.78, (1.2, -0.3, 0.9), 110, -25) for _ in states]
    site_ids = [
        mujoco.mj_name2id(s[0], mujoco.mjtObj.mjOBJ_SITE, "r0/grasp") for s in states
    ]
    if min(site_ids) < 0:
        raise ValueError("Both trials must contain r0/grasp")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title = ImageFont.truetype(font_path, 25)
    small = ImageFont.truetype(font_path, 18)
    labels = args.labels
    accents = ["#ffc08b", "#7ee5c6"]
    writer = (
        None
        if args.preview is not None
        else iio.get_writer(
            args.output,
            fps=1 / (dt * args.stride),
            codec="libx264",
            quality=8,
            macro_block_size=8,
            ffmpeg_params=["-threads", "2"],
        )
    )
    backend = None
    try:
        for index in indices:
            started = time.monotonic()
            canvas = Image.new("RGB", (1920, 1080), "#111b21")
            draw = ImageDraw.Draw(canvas)
            for panel, (model, data, qpos, records, _) in enumerate(states):
                x = panel * 960
                data.qpos[:] = qpos[index]
                mujoco.mj_forward(model, data)
                wide[panel].update_scene(data, cameras[panel], scene_option=options)
                canvas.paste(Image.fromarray(wide[panel].render()), (x, 48))
                details[panel].lookat[:] = data.site_xpos[site_ids[panel]]
                close[panel].update_scene(data, details[panel], scene_option=options)
                canvas.paste(Image.fromarray(close[panel].render()), (x, 648))
                if backend is None:
                    backend = {
                        "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
                        "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
                        "version": GL.glGetString(GL.GL_VERSION).decode(),
                    }
                draw.text((x + 20, 12), labels[panel], font=title, fill=accents[panel])
                draw.text(
                    (x + 792, 15),
                    f"{times[index]:05.1f}s / 1×",
                    font=small,
                    fill=accents[panel],
                )
                robots = records[index]["robots"]
                for slot, (name, state) in enumerate(sorted(robots.items())):
                    body = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, name + "/pelvis"
                    )
                    point = (
                        projected_label(
                            wide[panel].scene, data.xpos[body] + [0, 0, 0.7], 960, 600
                        )
                        if body >= 0
                        else None
                    )
                    if point:
                        px, py = x + point[0], 48 + point[1]
                        draw.rounded_rectangle(
                            (px - 20, py - 12, px + 20, py + 12),
                            radius=4,
                            fill="#14232b",
                        )
                        draw.text(
                            (px - 13, py - 10),
                            name.upper(),
                            font=small,
                            fill=accents[panel],
                        )
                    phase = str(state.get("phase", "recorded")).replace("_", " ")
                    left = x + 20 + slot * 450
                    draw.rounded_rectangle(
                        (left, 62, left + 430, 96), radius=4, fill="#14232b"
                    )
                    draw.text(
                        (left + 8, 69),
                        f"{name.upper()} · {phase}",
                        font=small,
                        fill=accents[panel],
                    )
                controller = robots["r0"].get("controller_state") or {}
                forces = controller.get("pad_force_n")
                force_text = (
                    f"{forces[0]:.2f} / {forces[1]:.2f} N"
                    if forces is not None
                    else "not recorded"
                )
                draw.rectangle((x, 648, x + 960, 680), fill="#14232b")
                draw.text(
                    (x + 18, 655),
                    f"R0 CONTACT DETAIL · pads {force_text}",
                    font=small,
                    fill=accents[panel],
                )
                attempts = controller.get("recovery_attempts", 0)
                events = controller.get("recovery_events", [])
                if events:
                    event = events[-1]
                    correction_mm = 1000 * float(
                        np.linalg.norm(event["correction_xyz"])
                    )
                    caption = f"Retry triggered {event['time']:.2f}s · pad {event['pad_index']}: {event['force_n']:.2f} N · correction {correction_mm:.1f} mm"
                elif controller.get("failed"):
                    caption = "Controller stopped: " + str(
                        controller.get("transition_reason", "failed")
                    ).replace("_", " ")
                else:
                    caption = "R0: measured tactile contact controls the next action"
                draw.rounded_rectangle(
                    (x + 16, 605, x + 944, 637), radius=4, fill="#14232b"
                )
                draw.text((x + 25, 612), caption, font=small, fill=accents[panel])
                draw.text(
                    (x + 710, 655),
                    f"Retries: {attempts}",
                    font=small,
                    fill=accents[panel],
                )
                draw.text(
                    (x + 20, 1020),
                    "Final recorded outcome: " + outcomes[panel],
                    font=title,
                    fill=accents[panel],
                )
            draw.line((960, 0, 960, 1080), fill="#82939a", width=2)
            draw.text(
                (20, 1055),
                "Matched trials · each panel is one shared 2-robot scene · unchanged recorded states",
                font=small,
                fill="#aebdc4",
            )
            if writer:
                writer.append_data(np.asarray(canvas))
            if index == poster:
                canvas.save(args.output.with_suffix(".png"))
            if index % 100 == 0:
                print(f"rendered {index}/{len(times)}", flush=True)
            time.sleep(max(0, 1 / args.max_render_fps - (time.monotonic() - started)))
    finally:
        if writer:
            writer.close()
        for renderer in wide + close:
            renderer.close()
    sources = []
    for path, state in zip(paths, states):
        sources.append(
            {
                "run": str(path),
                "scene_sha256": digest(path / "scene.xml"),
                "state_sha256": digest(path / "probe.npz"),
                "probe_sha256": digest(path / "probe.json"),
                "records_sha256": hashlib.sha256(state[4]).hexdigest(),
            }
        )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "sources": sources,
                "labels": labels,
                "gl_backend": backend,
                "fps": 1 / (dt * args.stride),
                "video_start_s": float(times[0]),
                "recorded_end_s": float(times[-1]),
                "last_replayed_time_s": float(times[indices[-1]]),
                "video_duration_s": len(indices) * dt * args.stride if writer else None,
                "dimensions": [1920, 1080],
                "recorded_frames_per_trial": len(times),
                "timestamps_identical": True,
                "state_modified": False,
                "physics_steps_generated": 0,
                "comparison": "Two matched trials; each panel independently contains all robots from its single shared scene",
                "overlays": "Recorded phase, controller pad forces and recovery_attempts; final outcomes audited separately",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
