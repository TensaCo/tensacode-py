"""Render measured kitting trajectories; replay never changes recorded physics."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    root = a.results
    runs = []
    for key in ("fixed_recipe", "alg2"):
        m = mujoco.MjModel.from_xml_path(str(root / f"{key}.xml"))
        d = mujoco.MjData(m)
        runs.append(
            (
                m,
                d,
                mujoco.Renderer(m, height=480, width=640),
                np.load(root / f"{key}.npz"),
                json.loads((root / f"{key}.json").read_text()),
            )
        )
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.43]
    camera.distance = 1.65
    camera.azimuth = 90
    camera.elevation = -48
    duration = max(float(r[3]["time"][-1]) for r in runs)
    frames = int(round(duration * 25))
    with iio.get_writer(
        root / "comparison.mp4",
        fps=25,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_log_level="error",
    ) as writer:
        for k in range(frames):
            t = (k + 1) / 25
            canvas = Image.new("RGB", (1280, 704), "#101925")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (24, 14),
                "TensorCode / order-driven kitting / measured MuJoCo replay",
                font=font(25),
                fill="white",
            )
            draw.text(
                (24, 53),
                "Order A B → B A A  ·  reject part A0  ·  first kit grasp offset by 8 cm",
                font=font(19),
                fill="#a5bed4",
            )
            for side, (m, d, renderer, trace, receipt) in enumerate(runs):
                index = min(
                    int(np.searchsorted(trace["time"], t)), len(trace["time"]) - 1
                )
                d.qpos[:] = trace["qpos"][index]
                mujoco.mj_forward(m, d)
                renderer.update_scene(d, camera)
                canvas.paste(Image.fromarray(renderer.render()), (640 * side, 122))
                draw = ImageDraw.Draw(canvas)
                color = "#67dfb2" if side else "#ffab86"
                x = side * 640 + 20
                title = (
                    "ALG.2 / feedback + revised recipe"
                    if side
                    else "Fixed recipe / no exception handling"
                )
                draw.text((x, 90), title, font=font(21), fill=color)
                done = t >= float(trace["time"][-1])
                label = (
                    str(trace["labels"][index])
                    if not done
                    else (
                        "Verified: 3 correct slots + quarantine"
                        if receipt["success"]
                        else "Stopped: order incorrect; no automatic recovery"
                    )
                )
                draw.text((x, 610), label, font=font(18), fill=color)
                draw.text(
                    (x, 645),
                    f"t={min(t, float(trace['time'][-1])):.1f}s  |  {'recorded endpoint held' if done else 'contact-only manipulation'}",
                    font=font(17),
                    fill="white",
                )
            draw.text(
                (24, 680),
                "Green squares: order slots   ·   red square: quarantine   ·   A=orange, B=blue   ·   supplied state / quality",
                font=font(15),
                fill="#a5bed4",
            )
            writer.append_data(np.asarray(canvas))
            if k == frames - 1:
                canvas.save(root / "poster.png")
    for _, _, renderer, _, _ in runs:
        renderer.close()
    (root / "render.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "fps": 25,
                "duration_seconds": frames / 25,
                "source": "Recorded qpos replay; shorter baseline held at endpoint",
                "rendering": "Mesa software rendering requested by resource guard",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
