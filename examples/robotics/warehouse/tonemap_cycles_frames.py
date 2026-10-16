"""Apply Blender's AgX display transform to denoised scene-linear PFM frames.

Run inside Blender with PYTHONHOME appropriate for that Blender build. This is
display conversion only, with no image synthesis, geometry or motion changes.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    import bpy
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exposure", type=float, default=-0.6)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = args.exposure
    frames = []
    for source in sorted(args.input.glob("*.pfm")):
        with source.open("rb") as stream:
            if stream.readline().strip() != b"PF":
                raise ValueError("Expected RGB PFM")
            width, height = map(int, stream.readline().split())
            scale = float(stream.readline())
            pixels = np.frombuffer(
                stream.read(), dtype="<f4" if scale < 0 else ">f4"
            ).reshape(height, width, 3)
        if not np.isfinite(pixels).all():
            raise ValueError("Nonfinite radiance pixels")
        rgba = np.ones((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = pixels * abs(scale)
        image = bpy.data.images.new(
            "Denoised scene-linear frame",
            width=width,
            height=height,
            alpha=True,
            float_buffer=True,
        )
        image.pixels.foreach_set(rgba.ravel())
        output = args.output / (source.stem + ".png")
        image.save_render(str(output), scene=scene)
        bpy.data.images.remove(image)
        frames.append(
            {
                "input": source.name,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "output": output.name,
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
    if not frames:
        raise ValueError("No PFM frames found")
    (args.output / "tonemap.json").write_text(
        json.dumps(
            {
                "display_transform": "AgX",
                "exposure": args.exposure,
                "blender_version": bpy.app.version_string,
                "frames": frames,
            },
            indent=2,
        )
    )
    print(f"Tonemapped {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
