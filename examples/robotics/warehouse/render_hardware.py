"""Paced hardware-EGL replay with an independent GPU headroom monitor.

Run this wrapper through g1/budget.py for CPU/RAM limits. It overrides only
software-renderer environment settings, not GPU/MPS configuration. NVIDIA
utilization >=75% pauses rendering until it drops below60%. Failure to query
utilization stops the child; no GPU workload is allowed to run unmonitored.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-report", type=Path, required=True)
    parser.add_argument("--render-fps", type=float, default=15)
    parser.add_argument(
        "--entrypoint",
        choices=["render.py", "render_comparison.py"],
        default="render.py",
    )
    args, forwarded = parser.parse_known_args()
    if not 0 < args.render_fps <= 25:
        parser.error("--render-fps must be between0 and25")
    env = os.environ.copy()
    for key in ("LIBGL_ALWAYS_SOFTWARE", "GALLIUM_DRIVER", "LP_NUM_THREADS"):
        env.pop(key, None)
    env["MUJOCO_GL"] = "egl"
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
        "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    samples = []
    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name(args.entrypoint)),
            *forwarded,
            "--max-render-fps",
            str(args.render_fps),
        ],
        env=env,
    )
    paused = False
    try:
        while child.poll() is None:
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=5,
                text=True,
            )
            utilization = max(float(line.strip()) for line in value.splitlines())
            if utilization >= 75 and not paused:
                child.send_signal(signal.SIGSTOP)
                paused = True
            elif utilization < 60 and paused:
                child.send_signal(signal.SIGCONT)
                paused = False
            samples.append(
                {
                    "time": time.time(),
                    "gpu_utilization_percent": utilization,
                    "paused": paused,
                }
            )
            time.sleep(0.5)
        result = child.returncode
    finally:
        if paused and child.poll() is None:
            child.send_signal(signal.SIGCONT)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        args.gpu_report.parent.mkdir(parents=True, exist_ok=True)
        args.gpu_report.write_text(
            json.dumps(
                {
                    "renderer": "NVIDIA EGL",
                    "render_fps_cap": args.render_fps,
                    "gpu_configuration_modified": False,
                    "pause_threshold_percent": 75,
                    "samples": samples,
                    "exit_code": child.returncode,
                },
                indent=2,
            )
        )
    raise SystemExit(result)


if __name__ == "__main__":
    main()
