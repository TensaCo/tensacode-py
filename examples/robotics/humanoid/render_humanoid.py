"""Render recorded physical G1 trajectories with synchronized close-ups and evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont

BG = "#0d1929"
WHITE = "#edf3fb"
MUTED = "#acbacd"
GREEN = "#62e4c3"
ORANGE = "#ffae68"


def font(size=22, bold=False):
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans"
        + ("-Bold" if bold else "")
        + ".ttf",
        size,
    )


def load(path):
    m = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    d = mujoco.MjData(m)
    return (
        m,
        d,
        np.load(path / "probe.npz")["qpos"],
        json.loads((path / "records.json").read_text()),
    )


def camera(distance, lookat, azimuth=135):
    c = mujoco.MjvCamera()
    c.azimuth = azimuth
    c.elevation = -15
    c.distance = distance
    c.lookat[:] = lookat
    return c


def render_main(run, output):
    m, d, q, records = load(run)
    r = mujoco.Renderer(m, height=720, width=960)
    detail = mujoco.Renderer(m, height=360, width=480)
    wide = camera(3.35, [0.65, 0, 0.66])
    close = camera(0.68, [1.25, -0.25, 0.85])
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    site = m.site("grasp").id
    output.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(
        output / "humanoid-walk-grasp.mp4",
        fps=25,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_log_level="error",
    ) as writer:
        for i in range(0, len(q), 2):
            d.qpos[:] = q[i]
            mujoco.mj_forward(m, d)
            rec = records[i]
            r.update_scene(d, wide, scene_option=opt)
            close.lookat[:] = d.site_xpos[site] + [0.025, 0, 0]
            detail.update_scene(d, close, scene_option=opt)
            canvas = Image.new("RGB", (1440, 896), BG)
            canvas.paste(Image.fromarray(r.render()), (0, 80))
            canvas.paste(Image.fromarray(detail.render()), (960, 80))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (28, 18),
                "TensorCode / Unitree G1 / walking + physical grasp",
                font=font(28, True),
                fill=WHITE,
            )
            draw.text((1200, 23), f"t = {rec['t']:.2f} s", font=font(23), fill=MUTED)
            draw.rounded_rectangle((24, 98, 430, 145), radius=8, fill=BG)
            draw.text(
                (38, 106),
                f"ALG.2  /  {rec['stage'].upper()}",
                font=font(24, True),
                fill=GREEN,
            )
            draw.text((982, 458), "HAND CLOSE-UP", font=font(21, True), fill=GREEN)
            draw.text((982, 497), "Green pads • orange cube", font=font(19), fill=MUTED)
            lift = max(0, rec["object"][2] - 0.805) * 100
            draw.text(
                (982, 548),
                f"Cube lift: {lift:5.1f} cm",
                font=font(28, True),
                fill=ORANGE,
            )
            for j, (pad, f) in enumerate(rec["pad_force_n"].items()):
                draw.text(
                    (982, 601 + 38 * j),
                    f"{pad.replace('pad_', 'Pad ').upper()}: {f:5.2f} N contact",
                    font=font(21),
                    fill=GREEN if f > 0.05 else MUTED,
                )
            draw.text(
                (982, 693),
                f"Pelvis height: {rec['base'][2]:.3f} m",
                font=font(21),
                fill=WHITE,
            )
            draw.text(
                (982, 733), "Free base • no object weld", font=font(19), fill=MUTED
            )
            x0, y0, w = 30, 830, 1380
            draw.rectangle((x0, y0, x0 + w, y0 + 6), fill="#33445b")
            draw.rectangle(
                (x0, y0, x0 + w * i / max(len(q) - 1, 1), y0 + 6), fill=GREEN
            )
            draw.text(
                (30, 854),
                "MuJoCo physics replay • 1× simulation time • supplied walking weights + authored manipulation",
                font=font(20),
                fill=MUTED,
            )
            writer.append_data(np.asarray(canvas))
            if abs(rec["t"] - 19) < 0.011:
                canvas.save(output / "humanoid-poster.png")
            if i % 250 == 0:
                print(f"rendered {i}/{len(q)}", flush=True)
    r.close()
    detail.close()


def render_control(closed_path, open_path, output):
    states = [load(open_path), load(closed_path)]
    rs = [mujoco.Renderer(x[0], height=480, width=640) for x in states]
    (output / "renderer.json").write_text(
        json.dumps(
            {
                "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
                "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
                "video_fps": 25,
            },
            indent=2,
        )
    )
    cams = [camera(0.95, [1.23, -0.25, 0.87]) for _ in states]
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    count = min(len(s[2]) for s in states)
    with iio.get_writer(
        output / "grip-control.mp4",
        fps=25,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_log_level="error",
    ) as writer:
        for i in range(600, count, 2):
            canvas = Image.new("RGB", (1280, 640), BG)
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (22, 16), "Open-gripper control", font=font(26, True), fill=ORANGE
            )
            draw.text(
                (662, 16),
                "Finger closure enabled",
                font=font(25, True),
                fill=GREEN,
            )
            for k, (m, d, q, records) in enumerate(states):
                d.qpos[:] = q[i]
                mujoco.mj_forward(m, d)
                rs[k].update_scene(d, cams[k], scene_option=opt)
                canvas.paste(Image.fromarray(rs[k].render()), (k * 640, 65))
                rec = records[i]
                lift = max(0, rec["object"][2] - 0.805) * 100
                draw.text(
                    (22 + k * 640, 558),
                    f"t={rec['t']:.2f}s  /  lift {lift:.1f} cm  /  {rec['stage']}",
                    font=font(22, True),
                    fill=ORANGE if k == 0 else GREEN,
                )
            draw.text(
                (22, 604),
                "Same planned arm path • left fingers stay open • right lifts through physical pad contact",
                font=font(20),
                fill=MUTED,
            )
            writer.append_data(np.asarray(canvas))
            if i == 950:
                canvas.save(output / "control-poster.png")
    for r in rs:
        r.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control-only", action="store_true")
    a = p.parse_args()
    if not a.control_only:
        render_main(a.results / "alg2", a.output)
    render_control(a.results.parent / "native", a.results / "open-gripper", a.output)
    print("Rendered annotated main sequence and open-gripper control.", flush=True)


if __name__ == "__main__":
    main()
