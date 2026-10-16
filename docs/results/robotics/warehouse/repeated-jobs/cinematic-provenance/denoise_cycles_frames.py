"""CPU-only OIDN denoising of retained scene-linear radiance frames.

Invoke under the resource guard. Input and output hashes are retained so the
denoised presentation can always be distinguished from raw Cycles output.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oidn", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 2:
        parser.error("This bounded CPU pipeline supports one or two threads")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for source in sorted(args.input.glob("radiance-*.pfm")):
        output = args.output / source.name.replace("radiance-", "frame-")
        started = time.monotonic()
        result = subprocess.run(
            [
                str(args.oidn),
                "--device",
                "cpu",
                "--threads",
                str(args.threads),
                "--affinity",
                "0",
                "--maxmem",
                "8192",
                "--hdr",
                str(source),
                "-o",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        frames.append(
            {
                "input": source.name,
                "input_sha256": sha(source),
                "output": output.name,
                "output_sha256": sha(output),
                "seconds": time.monotonic() - started,
                "cli_output": result.stdout,
            }
        )
        (args.output / "denoise.json").write_text(
            json.dumps(
                {
                    "denoiser": "OpenImageDenoise RT HDR",
                    "device": "CPU",
                    "threads": args.threads,
                    "binary_sha256": sha(args.oidn),
                    "frames": frames,
                    "raw_frames_retained": True,
                },
                indent=2,
            )
        )
        print(f"Denoised {len(frames)} frames", flush=True)
    if not frames:
        raise ValueError("No source radiance frames found")


if __name__ == "__main__":
    main()
