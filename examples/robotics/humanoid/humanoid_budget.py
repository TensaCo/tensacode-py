"""Run a CPU-only demonstration in a bounded user systemd unit.

Leaves >=20% CPU capacity outside its affinity/quota and uses a conservative RAM
limit. Stops its own unit on low global available memory; pauses it when total CPU
utilization exceeds 75%. Does not stop, renice or reconfigure other workloads.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def sample():
    values = {
        k: int(v.split()[0])
        for k, v in (
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    cpu = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
    return {
        "available_fraction": values["MemAvailable"] / values["MemTotal"],
        "available_gib": values["MemAvailable"] / 1024**2,
        "cpu_total": sum(cpu),
        "cpu_idle": cpu[3] + cpu[4],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--unit", default=f"tensorcode-humanoid-{os.getpid()}")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not command:
        p.error("command required")
    start = sample()
    if start["available_fraction"] < 0.3:
        raise RuntimeError("Insufficient free memory; no workload started")
    a.output.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))[: min(6, max(1, int(os.cpu_count() * 0.5)))]
    unit = [
        "systemd-run",
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        f"--unit={a.unit}",
        "-p",
        f"CPUQuota={len(cpus) * 100}%",
        "-p",
        "MemoryMax=12G",
        "-p",
        "MemorySwapMax=0",
        "-p",
        "AllowedCPUs=" + ",".join(map(str, cpus)),
        "--setenv=CUDA_VISIBLE_DEVICES=",
        "--setenv=OMP_NUM_THREADS=2",
        "--setenv=OPENBLAS_NUM_THREADS=2",
        "--setenv=MUJOCO_GL=egl",
        "--setenv=LIBGL_ALWAYS_SOFTWARE=1",
        "--setenv=GALLIUM_DRIVER=llvmpipe",
        "--setenv=LP_NUM_THREADS=4",
        "--setenv=__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        "--working-directory=" + os.getcwd(),
        *command,
    ]
    log = []
    paused = False
    last = start
    with (a.output / "stdout.log").open("w") as stream:
        proc = subprocess.Popen(unit, stdout=stream, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(1)
            now = sample()
            delta = now["cpu_total"] - last["cpu_total"]
            usage = 1 - (now["cpu_idle"] - last["cpu_idle"]) / max(delta, 1)
            entry = {
                "time": time.time(),
                "available_memory_fraction": now["available_fraction"],
                "available_gib": now["available_gib"],
                "total_cpu_fraction": usage,
                "paused": paused,
            }
            log.append(entry)
            last = now
            if now["available_fraction"] < 0.23:
                subprocess.run(["systemctl", "--user", "stop", a.unit], check=False)
                entry["stopped_for_memory_headroom"] = True
                break
            if usage > 0.75 and not paused:
                subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "kill",
                        "--signal=SIGSTOP",
                        "--kill-whom=all",
                        a.unit,
                    ],
                    check=False,
                )
                paused = True
            elif paused and usage < 0.6:
                subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "kill",
                        "--signal=SIGCONT",
                        "--kill-whom=all",
                        a.unit,
                    ],
                    check=False,
                )
                paused = False
        code = proc.wait()
    report = {
        "host": os.uname().nodename,
        "cpu_count": os.cpu_count(),
        "allowed_cpus": cpus,
        "cpu_quota_cores": len(cpus),
        "memory_limit_gib": 12,
        "gpu_compute": "disabled",
        "renderer": "Mesa llvmpipe software",
        "minimum_memory_available_fraction": min(
            [start["available_fraction"]]
            + [x["available_memory_fraction"] for x in log]
        ),
        "samples": log,
        "exit_code": code,
    }
    (a.output / "resources.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
