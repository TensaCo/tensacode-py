"""Build a portable HTML evidence report with embedded annotated MuJoCo videos.

Run: MUJOCO_GL=egl OMP_NUM_THREADS=2 .venv/bin/python examples/build_robotics_report.py
Requires the robotics extra plus imageio-ffmpeg. Reuses committed frozen policy;
never retrains. Rendering replays must reproduce the recorded tracking metrics.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from robotics_policy import policy_algorithm, rollout

from tensorcode.algorithm import Algorithm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/results/algorithm-robotics"
ORANGE = "#ffae68"
TEAL = "#62e4c3"
INK = "#102033"
WHITE = "#edf3fb"
MUTED = "#acbacd"


def font(size=20, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def video(label, baseline, corrected, path):
    """Annotate actual simulator renders, at true geometry and real simulation speed."""
    frames_a, frames_b = baseline[2], corrected[2]
    ta, tb = baseline[1], corrected[1]
    errors = [
        np.asarray([np.array(x["position"]) - x["target"] for x in t]) for t in (ta, tb)
    ]
    rms = [np.sqrt(np.mean(e**2, axis=1)) * 1000 for e in errors]
    times = np.array([x["t"] for x in ta])
    upper = max(15, np.ceil(max(x.max() for x in rms) / 5) * 5)
    with iio.get_writer(
        path,
        fps=25,
        codec="libx264",
        quality=7,
        macro_block_size=16,
        ffmpeg_log_level="error",
    ) as writer:
        for i, (fa, fb) in enumerate(zip(frames_a, frames_b, strict=True)):
            canvas = Image.new("RGB", (1280, 800), INK)
            canvas.paste(Image.fromarray(fa), (0, 100))
            canvas.paste(Image.fromarray(fb), (640, 100))
            d = ImageDraw.Draw(canvas)
            d.text((24, 15), f"TensorCode / {label}", font=font(22, True), fill=WHITE)
            d.text((1080, 20), f"t = {i * 0.04:.2f} s", font=font(20), fill=WHITE)
            d.text(
                (24, 58),
                "BASELINE  /  authored impedance",
                font=font(22, True),
                fill=ORANGE,
            )
            d.text(
                (664, 58),
                "CORRECTED  /  trained in ALG.2",
                font=font(22, True),
                fill=TEAL,
            )
            j = min(i * 2, len(ta) - 1)
            for x, color, series in [(24, ORANGE, rms[0]), (664, TEAL, rms[1])]:
                d.rounded_rectangle((x, 112, x + 345, 153), radius=8, fill=INK)
                d.text(
                    (x + 10, 120),
                    f"20-joint RMS now: {series[j]:.2f} mrad",
                    font=font(18, True),
                    fill=color,
                )
            # The same physical camera and geometry are used for both simulations.
            d.line([(410, 474), (305, 413)], fill=WHITE, width=2)
            d.text((390, 480), "Fixed base", font=font(18), fill=WHITE)
            d.line([(1050, 474), (945, 413)], fill=WHITE, width=2)
            d.text((1030, 480), "Fixed base", font=font(18), fill=WHITE)
            d.text(
                (24, 585),
                "TRACKING ERROR / all 20 joints / identical axis / lower is better",
                font=font(18, True),
                fill=WHITE,
            )
            x0, y0, w, h = 85, 635, 1090, 100
            for v in (0, upper / 2, upper):
                y = y0 + h - v / upper * h
                d.line((x0, y, x0 + w, y), fill="#314258", width=1)
                d.text((14, y - 10), f"{v:g}", font=font(16), fill=MUTED)
            for s, color in zip(rms, (ORANGE, TEAL), strict=True):
                points = [
                    (x0 + t / 6 * w, y0 + h - value / upper * h)
                    for t, value in zip(times[: j + 1], s[: j + 1], strict=True)
                ]
                if len(points) > 1:
                    d.line(points, fill=color, width=3)
            x = x0 + i * 0.04 / 6 * w
            d.line((x, y0, x, y0 + h), fill=WHITE, width=1)
            d.text((20, 610), "mrad", font=font(15), fill=MUTED)
            d.text((85, 743), "0 s", font=font(15), fill=MUTED)
            d.text((1140, 743), "6 s", font=font(15), fill=MUTED)
            d.text(
                (24, 775),
                "Real MuJoCo replay • 1× simulation time • true geometry • error graph makes small differences visible",
                font=font(16),
                fill=MUTED,
            )
            writer.append_data(np.asarray(canvas))
            if i == 100:
                canvas.save(path.with_suffix(".png"))


def bars(rows):
    maxval = max(r["joint_rmse_rad"] for r in rows) * 1000
    svg = [
        '<svg viewBox="0 0 960 230" role="img" aria-label="Joint tracking error comparison in milliradians; lower is better">'
    ]
    for i, (r, name) in enumerate(
        zip(
            rows,
            [
                "Nominal · baseline",
                "Nominal · corrected",
                "Shifted · baseline",
                "Shifted · corrected",
            ],
            strict=True,
        )
    ):
        y = 15 + i * 53
        value = r["joint_rmse_rad"] * 1000
        c = ORANGE if i % 2 == 0 else TEAL
        svg += [
            f'<text x="0" y="{y + 23}" fill="{WHITE}" font-size="18">{name}</text>',
            f'<rect x="240" y="{y}" width="{value / maxval * 570:.2f}" height="32" rx="4" fill="{c}"/>',
            f'<text x="{255 + value / maxval * 570:.2f}" y="{y + 23}" fill="{WHITE}" font-size="18">{value:.3f}</text>',
        ]
    return "".join(svg) + "</svg>"


def main():
    torch.set_num_threads(2)
    report = json.loads((OUT / "report.json").read_text())
    policy = Algorithm.from_json((OUT / "policy.json").read_text())
    xml = (OUT / "robot.xml").read_text()
    replays = []
    videos = {}
    for key, label, phase, scale, indices in [
        ("nominal", "Nominal actuator disturbance", 0.7, 1.0, (0, 3)),
        ("shifted", "New trajectory + 25% stronger disturbance", 2.1, 1.25, (4, 5)),
    ]:
        pair = []
        for p, substrate, index in [
            (policy_algorithm(), "alg0", indices[0]),
            (policy, "alg2", indices[1]),
        ]:
            result = rollout(
                xml,
                p,
                target_substrate=substrate,
                seconds=6,
                phase=phase,
                disturbance_scale=scale,
                record=True,
            )
            for metric in ("joint_rmse_rad", "worst_joint_error_rad", "max_torque_nm"):
                if not np.isclose(
                    result[0][metric],
                    report["rollouts"][index][metric],
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise RuntimeError(f"Replay mismatch: {key} {substrate} {metric}")
            pair.append(result)
            replays.append(
                {
                    "scenario": key,
                    "substrate": substrate,
                    "metrics": result[0],
                    "trajectory": result[1],
                }
            )
        path = OUT / f"comparison-{key}.mp4"
        video(label, *pair, path)
        videos[key] = (
            "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
        )
        print(f"Verified and rendered {key}: {path.stat().st_size:,} bytes", flush=True)
    evidence = {
        "policy_sha256": hashlib.sha256((OUT / "policy.json").read_bytes()).hexdigest(),
        "replayed_from_committed_frozen_policy": True,
        "tracking_metrics_match_original": True,
        "video_fps": 25,
        "simulation_seconds": 6,
        "replays": replays,
    }
    (OUT / "presentation-replay.json").write_text(
        json.dumps(evidence, separators=(",", ":"))
    )
    render_html(report, evidence, videos)


def render_html(report, evidence, videos):
    template = Path(__file__).with_name("robotics_report.html").read_text()
    rows = report["rollouts"]
    nominal_gain = 100 * (1 - rows[3]["joint_rmse_rad"] / rows[0]["joint_rmse_rad"])
    shifted_gain = 100 * (1 - rows[5]["joint_rmse_rad"] / rows[4]["joint_rmse_rad"])
    runtime = "".join(
        f"<tr><th>{r['substrate'].upper()}</th><td>{r['joint_rmse_rad'] * 1000:.3f}</td><td>{r['mean_compute_ms']:.3f}</td><td>{r['p99_compute_ms']:.3f}</td><td>{r['max_compute_ms']:.3f}</td></tr>"
        for r in rows[1:4]
    )
    for key, value in {
        "NOMINAL_VIDEO": videos["nominal"],
        "SHIFTED_VIDEO": videos["shifted"],
        "NOMINAL_GAIN": f"{nominal_gain:.1f}",
        "SHIFTED_GAIN": f"{shifted_gain:.1f}",
        "BARS": bars([rows[i] for i in (0, 3, 4, 5)]),
        "RUNTIME": runtime,
        "POLICY_HASH": evidence["policy_sha256"],
        "REPORT_JSON": json.dumps(report, indent=2),
        "NOMINAL_POSTER": "data:image/png;base64,"
        + base64.b64encode((OUT / "comparison-nominal.png").read_bytes()).decode(),
        "SHIFTED_POSTER": "data:image/png;base64,"
        + base64.b64encode((OUT / "comparison-shifted.png").read_bytes()).decode(),
    }.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise RuntimeError("Unresolved template field")
    (OUT / "index.html").write_text(template)
    print(f"Wrote self-contained report: {OUT / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
