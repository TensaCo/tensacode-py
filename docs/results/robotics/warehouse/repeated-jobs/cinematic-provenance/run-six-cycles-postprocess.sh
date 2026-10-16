#!/usr/bin/env bash
set -euo pipefail
root=/home/brandonin/Documents/tensacode/tensacode/python
frames=/tmp/tensorcode-warehouse-jobs/six-cycles-video
"$root/.venv/bin/python" "$root/examples/robotics/warehouse/denoise_cycles_frames.py" --oidn /tmp/tensorcode-oidn/install/bin/oidnDenoise --input "$frames/radiance" --output "$frames/denoised" --threads 2
env PYTHONHOME=/usr /usr/bin/blender --background --factory-startup --python-exit-code 1 --python "$root/examples/robotics/warehouse/tonemap_cycles_frames.py" -- --input "$frames/denoised" --output "$frames/denoised-png"
"$root/.venv/bin/python" "$root/examples/robotics/warehouse/assemble_cycles_video.py" --sequence /tmp/tensorcode-warehouse-jobs/six-cycles-sequence --frames "$frames/denoised-png" --output "$root/docs/results/robotics/warehouse/repeated-jobs/cinematic.mp4"
