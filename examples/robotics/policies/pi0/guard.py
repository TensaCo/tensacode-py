"""Bound a conversion/inference process and retain global resource samples."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def sample():
    memory = {
        k: int(v.split()[0])
        for k, v in (
            row.split(":", 1) for row in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    return memory["MemAvailable"] / memory["MemTotal"], sum(cpu), cpu[3] + cpu[4]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", default="12,13,14,15")
    parser.add_argument("--mps-pipe", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command required")
    previous = sample()
    if previous[0] < 0.55:
        raise RuntimeError("Need 55% available RAM before conversion")
    cpus = sorted({int(value) for value in args.cpus.split(",")})
    if len(cpus) != 4 or not set(cpus).issubset(os.sched_getaffinity(0)):
        parser.error("--cpus must select exactly four available CPUs")
    unit = f"tensorcode-pi0-convert-{os.getpid()}"
    args.output.mkdir(parents=True, exist_ok=False)
    launch = [
        "systemd-run",
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        f"--unit={unit}",
        "-p",
        "CPUQuota=400%",
        "-p",
        "MemoryMax=48G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "AllowedCPUs=" + ",".join(map(str, cpus)),
        "--setenv=CUDA_VISIBLE_DEVICES=" + ("0" if args.mps_pipe else ""),
        "--setenv=OMP_NUM_THREADS=4",
        "--setenv=OPENBLAS_NUM_THREADS=4",
        "--setenv=JAX_PLATFORMS=cpu",
        "--setenv=MUJOCO_GL=egl",
        "--setenv=__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        "--setenv=LIBGL_ALWAYS_SOFTWARE=1",
        "--setenv=GALLIUM_DRIVER=llvmpipe",
        "--setenv=LP_NUM_THREADS=2",
        "--working-directory=" + os.getcwd(),
    ]
    if args.mps_pipe:
        if not args.mps_pipe.is_dir():
            raise FileNotFoundError(args.mps_pipe)
        launch.extend(
            [
                "--setenv=CUDA_MPS_PIPE_DIRECTORY=" + str(args.mps_pipe),
                "--setenv=CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70",
            ]
        )
    launch.extend(command)
    rows, paused = [], False
    with (args.output / "stdout.log").open("w") as log:
        process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            time.sleep(1)
            current = sample()
            cpu_fraction = 1 - (current[2] - previous[2]) / max(
                1, current[1] - previous[1]
            )
            rows.append(
                {
                    "time": time.time(),
                    "available_memory_fraction": current[0],
                    "cpu_fraction": cpu_fraction,
                    "paused": paused,
                }
            )
            previous = current
            if current[0] < 0.25:
                subprocess.run(["systemctl", "--user", "stop", unit], check=False)
                break
            signal = (
                "SIGSTOP"
                if cpu_fraction > 0.75 and not paused
                else "SIGCONT"
                if cpu_fraction < 0.6 and paused
                else None
            )
            if signal:
                subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "kill",
                        "--signal=" + signal,
                        "--kill-whom=all",
                        unit,
                    ],
                    check=False,
                )
                paused = signal == "SIGSTOP"
        code = process.wait()
    (args.output / "resources.json").write_text(
        json.dumps(
            {
                "cpu_cores": cpus,
                "memory_max_gib": 48,
                "gpu_compute": bool(args.mps_pipe),
                "mps_requested_active_thread_percentage": 70 if args.mps_pipe else None,
                "samples": rows,
                "exit_code": code,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
