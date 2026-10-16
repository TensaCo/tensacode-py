"""Encode denoised recorded-state frames with source-clock/provenance labels."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.sequence / "sequence.json").read_text())
    poses = np.load(args.sequence / "poses.npz")
    frames = [
        args.frames / f"frame-{i:05d}.png" for i in range(metadata["frame_count"])
    ]
    if any(not p.exists() for p in frames):
        raise ValueError("Missing frames; refusing to shorten or invent motion")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    hashes = []
    with iio.get_writer(
        args.output,
        fps=metadata["fps"],
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_params=["-threads", "2"],
    ) as writer:
        for index, path in enumerate(frames):
            image = Image.open(path).convert("RGB")
            if image.size != (1280, 720):
                raise ValueError("Unexpected rendered dimensions")
            canvas = Image.new("RGB", (1280, 768), "#111b21")
            canvas.paste(image, (0, 0))
            draw = ImageDraw.Draw(canvas)
            if metadata.get("explicit_cuts"):
                chapter_index, chapter = next(
                    (i, item)
                    for i, item in enumerate(metadata["chapters"])
                    if item["first_frame"]
                    <= index
                    < item["first_frame"] + item["frame_count"]
                )
                draw.rectangle((0, 0, 1280, 40), fill="#111b21")
                draw.text(
                    (18, 10),
                    f"Chapter {chapter_index + 1} · {chapter['label']} · explicit source-time cuts",
                    font=font,
                    fill="#e6eff3",
                )
            draw.text(
                (18, 734),
                "Cycles + OIDN · unchanged recorded motion",
                font=font,
                fill="#e6eff3",
            )
            draw.text(
                (1000, 734),
                f"Source {poses['source_times'][index]:05.2f}s / 1×",
                font=font,
                fill="#7ee5c6",
            )
            writer.append_data(np.asarray(canvas))
            if index == len(frames) // 2:
                canvas.save(args.output.with_suffix(".png"))
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "sequence": metadata,
                "presentation": "Path-traced Cycles frames, CPU OIDN denoising, AgX display transform; not the controller camera input",
                "fps": metadata["fps"],
                "duration_s": len(frames) / metadata["fps"],
                "dimensions": [1280, 768],
                "denoised_frame_sha256": hashes,
                "video_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "source_interpolation": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
