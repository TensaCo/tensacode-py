"""Warehouse presentation of recorded MuJoCo trajectories, with no state edits.

The racks, cartons, markings, and walls are rendering-only, noncolliding scenery.
This is a rasterized physics replay, not a photorealism or perception benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size=22, bold=False):
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans"
        + ("-Bold" if bold else "")
        + ".ttf",
        size,
    )


def warehouse_scene(source: Path, assets: Path | None, work: Path):
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    meshdir = Path(compiler.get("meshdir", ""))
    if assets:
        meshdir = assets / "src/assets/robots/unitree_g1/xmls/assets"
    elif not meshdir.is_absolute():
        meshdir = source.parent / meshdir
    compiler.set("meshdir", str(meshdir.resolve()))
    asset = root.find("asset")
    world = root.find("worldbody")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    for child in list(visual):
        visual.remove(child)
    ET.SubElement(
        visual,
        "global",
        offwidth="1600",
        offheight="1000",
        azimuth="135",
        elevation="-15",
    )
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(
        visual,
        "headlight",
        diffuse=".35 .36 .4",
        ambient=".3 .31 .34",
        specular=".15 .15 .15",
    )
    ET.SubElement(visual, "rgba", haze=".65 .68 .73 1")
    ET.SubElement(visual, "map", znear=".01", zfar="40", fogstart="12", fogend="25")
    for light in list(world.findall("light")):
        world.remove(light)
    ET.SubElement(
        world,
        "light",
        pos="1 0 5",
        dir="0 .3 -1",
        directional="true",
        diffuse=".8 .78 .72",
        ambient=".1 .1 .1",
        specular=".25 .25 .25",
        castshadow="true",
    )
    ET.SubElement(
        world,
        "light",
        pos="-2 2 3",
        dir="1 -1 -1",
        directional="true",
        diffuse=".4 .45 .55",
        castshadow="false",
    )
    materials = {
        "wh_floor": (".48 .51 .53 1", ".06"),
        "wh_blue": (".055 .14 .22 1", ".25"),
        "wh_orange": (".92 .32 .055 1", ".2"),
        "wh_card": (".57 .40 .23 1", ".03"),
        "wh_wall": (".65 .69 .72 1", ".05"),
        "wh_dark": (".12 .15 .18 1", ".1"),
        "wh_yellow": (".94 .68 .1 1", ".05"),
        "wh_wood": (".34 .24 .15 1", ".02"),
    }
    for name, (rgba, specular) in materials.items():
        ET.SubElement(
            asset, "material", name=name, rgba=rgba, specular=specular, shininess=".15"
        )
    rng = np.random.default_rng(42)
    for texture_name, material_name, spread in [
        ("concrete", "wh_floor", 8),
        ("cardboard", "wh_card", 6),
        ("paint", "wh_wall", 3),
    ]:
        grain = np.clip(rng.normal(210, spread, (512, 512)), 0, 255).astype(np.uint8)
        texture_file = work / f"{texture_name}.png"
        Image.fromarray(grain).convert("RGB").save(texture_file)
        ET.SubElement(
            asset,
            "texture",
            name="wh_" + texture_name,
            type="2d",
            file=str(texture_file.resolve()),
        )
        material = asset.find(f"material[@name='{material_name}']")
        material.set("texture", "wh_" + texture_name)
        material.set("texrepeat", "2 2")
        material.set("texuniform", "true")
    floor = world.find("geom[@name='floor']")
    if floor is not None:
        floor.set("material", "wh_floor")
    pedestal = world.find("geom[@name='pedestal']")
    if pedestal is not None:
        pedestal.set("rgba", ".20 .24 .27 1")
    silver = asset.find("material[@name='silver']")
    if silver is not None:
        silver.set("specular", ".6")
        silver.set("shininess", ".6")

    def box(pos, size, material, **kw):
        return ET.SubElement(
            world,
            "geom",
            type="box",
            pos=" ".join(map(str, (pos[0], -pos[1], pos[2]))),
            size=" ".join(map(str, size)),
            material=material,
            contype="0",
            conaffinity="0",
            group="4",
            **kw,
        )

    # Background storage lanes. All new geometries have zero collision masks.
    box((0.5, -3.3, 2), (6, 0.08, 2), "wh_wall")
    box((-4, 0, 2), (0.08, 3.3, 2), "wh_wall")
    for x in np.arange(-3.6, 5.4, 1.5):
        for y in [-2.9, -2.1]:
            box((x, y, 1.4), (0.045, 0.045, 1.4), "wh_blue")
        for z in [0.12, 0.95, 1.8, 2.65]:
            box((x + 0.7, -2.1, z), (0.7, 0.035, 0.045), "wh_orange")
            box((x + 0.7, -2.9, z), (0.7, 0.035, 0.045), "wh_orange")
            box((x + 0.7, -2.5, z - 0.04), (0.7, 0.4, 0.025), "wh_dark")
            if z < 2.5:
                for j in range(3):
                    px = x + 0.23 + j * 0.46
                    pz = z + 0.25
                    box((px, -2.5, pz), (0.20, 0.30, 0.20), "wh_card")
                    box((px, -2.195, pz), (0.10, 0.002, 0.07), "wh_wall")
                    for b in range(7):
                        box(
                            (px - 0.07 + b * 0.02, -2.191, pz - 0.015),
                            (0.004 if b % 2 else 0.007, 0.001, 0.035),
                            "wh_dark",
                        )
                    box((px, -2.5, pz + 0.202), (0.027, 0.3, 0.001), "wh_wood")
    # Industrial floor expansion joints, walking aisle, and workcell boundary.
    for x in range(-5, 7):
        box((x, 0, 0.006), (0.003, 4, 0.001), "wh_dark")
    for y in range(-3, 5):
        box((0, y, 0.006), (7, 0.003, 0.001), "wh_dark")
    for y in [-1.5, 1.2]:
        box((0.6, y, 0.002), (3, 0.025, 0.001), "wh_yellow")
    for x in [-2.4, 3.6]:
        box((x, -0.15, 0.002), (0.025, 1.35, 0.001), "wh_yellow")
    # Pallet staging off the motion path.
    for py in [1.8, 2.9]:
        for px in [2.7, 3.5]:
            for k in range(4):
                box((px - 0.3 + k * 0.2, py, 0.06), (0.075, 0.36, 0.035), "wh_wood")
            box((px, py, 0.27), (0.34, 0.30, 0.17), "wh_card")
    output = work / "warehouse-scene.xml"
    ET.ElementTree(root).write(output)
    return output


def load(run, assets, work):
    work.mkdir(parents=True, exist_ok=True)
    scene = warehouse_scene(run / "scene.xml", assets, work)
    model = mujoco.MjModel.from_xml_path(str(scene))
    qpos = np.load(run / "probe.npz")["qpos"]
    records = json.loads((run / "records.json").read_text())
    if len(qpos) != len(records) or qpos.shape[1] != model.nq:
        raise ValueError("State trace does not match model")
    return model, mujoco.MjData(model), qpos, records


def camera(distance=3.2, lookat=(0.55, -0.05, 0.72), azimuth=135, elevation=-17):
    c = mujoco.MjvCamera()
    c.distance, c.azimuth, c.elevation = distance, azimuth, elevation
    c.lookat[:] = lookat
    return c


def render(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [args.run] + ([args.comparison] if args.comparison else [])
    states = [
        load(p, args.assets, args.output.parent / (args.output.stem + f"-scene-{k}"))
        for k, p in enumerate(paths)
    ]
    paired = len(states) == 2
    width, height = (768, 768) if paired else (1536, 864)
    renderers = [mujoco.Renderer(s[0], height=height, width=width) for s in states]
    details = [mujoco.Renderer(s[0], height=240, width=360) for s in states]
    cams = [
        camera(distance=3.05 if paired else 3.25, lookat=(0.65, -0.12, 0.72))
        for _ in states
    ]
    close = [
        camera(distance=0.6, lookat=(1.25, -0.25, 0.9), elevation=-12) for _ in states
    ]
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.geomgroup[4] = 1
    opt.sitegroup[:] = 0
    frame_count = (
        max(len(s[2]) for s in states)
        if args.hold_ended
        else min(len(s[2]) for s in states)
    )
    first = max(0, int(args.start * 50))
    last = min(frame_count, int(args.end * 50)) if args.end else frame_count
    indices = (
        [min(last - 1, int(args.preview * 50))]
        if args.preview is not None
        else range(first, last, args.stride)
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
    labels = args.labels or (
        ["TensorCode transfer", "Fixed reference skill"]
        if paired
        else ["TensorCode / humanoid transfer"]
    )
    initial_z = [s[3][0]["object"][2] for s in states]
    sensor_images = [
        Image.open(path / "sensor-rgb.png").convert("RGB").resize((224, 168))
        if (path / "sensor-rgb.png").exists()
        else None
        for path in paths
    ]
    try:
        for index in indices:
            canvas = Image.new(
                "RGB",
                (
                    width * len(states),
                    height + (128 if paired and args.subtitle else 96),
                ),
                "#10191e",
            )
            draw = ImageDraw.Draw(canvas)
            for k, (m, d, q, records) in enumerate(states):
                record_index = min(index, len(q) - 1)
                d.qpos[:] = q[record_index]
                mujoco.mj_forward(m, d)
                renderers[k].update_scene(d, cams[k], scene_option=opt)
                canvas.paste(Image.fromarray(renderers[k].render()), (k * width, 48))
                rec = records[record_index]
                accent = "#78edce" if k == 0 else "#ffc380"
                draw.text(
                    (k * width + 24, 12), labels[k], font=font(22, True), fill=accent
                )
                draw.rounded_rectangle(
                    (k * width + 20, 66, k * width + 225, 106), radius=7, fill="#10191e"
                )
                draw.text(
                    (k * width + 32, 75),
                    rec.get("stage", "replay").replace("_", " ").upper(),
                    font=font(18, True),
                    fill="#f3f5f6",
                )
                if rec.get("stage") not in ["walk", "approach"] and index > 350:
                    close[k].lookat[:] = d.site_xpos[m.site("grasp").id] + [0.025, 0, 0]
                    details[k].update_scene(d, close[k], scene_option=opt)
                    inset = Image.fromarray(details[k].render())
                    ix, iy = k * width + width - 372, height - 206
                    canvas.paste(inset, (ix, iy))
                    draw.rectangle(
                        (ix - 1, iy - 1, ix + 361, iy + 241), outline=accent, width=2
                    )
                if paired and sensor_images[k] is not None:
                    sx, sy = k * width + 20, 122
                    canvas.paste(sensor_images[k], (sx, sy))
                    draw.rectangle((sx, sy, sx + 224, sy + 26), fill="#10191e")
                    draw.text(
                        (sx + 7, sy + 5),
                        "RGB-D INPUT · FIRST FRAME",
                        font=font(12, True),
                        fill="#edf3fb",
                    )
                if index >= len(q):
                    draw.rounded_rectangle(
                        (k * width + 250, 66, k * width + width - 20, 106),
                        radius=7,
                        fill="#10191e",
                    )
                    draw.text(
                        (k * width + 262, 77),
                        "RECORDED RUN ENDED · final frame",
                        font=font(16, True),
                        fill="#ffc380",
                    )
                lift = max(0.0, rec["object"][2] - initial_z[k]) * 100
                draw.text(
                    (k * width + 24, height + 61),
                    f"{rec['t']:04.1f}s   ·   lift {lift:.1f} cm",
                    font=font(20),
                    fill=accent,
                )
            if paired:
                draw.text(
                    (width * len(states) - 280, height + 65),
                    "Warehouse scenery only",
                    font=font(17),
                    fill="#acb7bf",
                )
            if not paired:
                draw.text(
                    (width - 564, height + 63),
                    "Physics replay · warehouse scenery only",
                    font=font(17),
                    fill="#acb7bf",
                )
            if paired and args.subtitle:
                draw.text(
                    (24, height + 101), args.subtitle, font=font(17), fill="#acb7bf"
                )
            if writer:
                writer.append_data(np.asarray(canvas))
            if (
                args.preview is not None
                or index == first
                or abs(index / 50 - args.poster_time) < 0.025
            ):
                canvas.save(args.output.with_suffix(".png"))
            if index % 100 == 0:
                print(f"rendered {index}/{last}", flush=True)
    finally:
        if writer:
            writer.close()
        for r in renderers + details:
            r.close()
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "runs": [str(p) for p in paths],
                "state_sha256": [
                    hashlib.sha256((p / "probe.npz").read_bytes()).hexdigest()
                    for p in paths
                ],
                "records_sha256": [
                    hashlib.sha256((p / "records.json").read_bytes()).hexdigest()
                    for p in paths
                ],
                "visual_only_scenery": True,
                "state_modified": False,
                "fps": 50 / args.stride,
                "start_s": args.start,
                "end_s": args.end,
                "renderer": "MuJoCo OpenGL rasterizer",
                "ended_run_display": "Labeled final recorded frame held; no simulated continuation"
                if args.hold_ended
                else "Comparison ends with shorter recording",
                "recorded_frames": [len(s[2]) for s in states],
                "description": "Recorded states replayed unchanged. Warehouse scenery has no collision or sensor role.",
            },
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--comparison", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets", type=Path)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--subtitle", default="")
    p.add_argument(
        "--hold-ended",
        action="store_true",
        help="Continue longer recording; clearly label frozen final frame of ended run",
    )
    p.add_argument("--poster-time", type=float, default=19)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float)
    p.add_argument("--preview", type=float, help="Render a single timestamp to PNG")
    args = p.parse_args()
    if args.labels and len(args.labels) != (2 if args.comparison else 1):
        p.error("Supply one label per run")
    if args.stride < 1:
        p.error("--stride must be positive")
    render(args)


if __name__ == "__main__":
    main()
