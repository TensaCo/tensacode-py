"""Run a reviewed JSON job queue locally, with durable evidence and headroom guards.

One worker owns one CPU lane. Coordinate lanes across workers: this worker never
changes other workloads. CUDA jobs require an existing private MPS directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def sample():
    memory = {
        k: int(v.split()[0])
        for k, v in (
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    cpu = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    return {
        "time": time.time(),
        "memory_available_fraction": memory["MemAvailable"] / memory["MemTotal"],
        "cpu_total": sum(cpu),
        "cpu_idle": cpu[3] + cpu[4],
        "gpu_fraction": float(gpu.stdout.strip().splitlines()[0]) / 100
        if gpu.returncode == 0
        else None,
    }


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def control(unit, action, sig=None):
    args = ["systemctl", "--user", action]
    if sig:
        args += ["--signal=" + sig, "--kill-whom=all"]
    subprocess.run([*args, unit], capture_output=True, check=False)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashes(root, paths):
    result = {}
    for relative in paths:
        path = (root / relative).resolve()
        path.relative_to(root)
        result[relative] = digest(path)
    return result


def validate(config):
    cpus = config["cpus"]
    if (
        not cpus
        or len(set(cpus)) != len(cpus)
        or not set(cpus).issubset(os.sched_getaffinity(0))
    ):
        raise ValueError("cpus must be unique available CPU indices")
    if len(cpus) > int(os.cpu_count() * 0.8):
        raise ValueError("CPU lane must leave at least20% capacity")
    if not 1 <= config["memory_gib"] <= 64:
        raise ValueError("memory_gib must be1–64")
    identifiers = set()
    for job in config["jobs"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job["id"]) or job["id"] in identifiers:
            raise ValueError("job IDs must be unique safe identifiers")
        identifiers.add(job["id"])
        if not job["command"] or not all(isinstance(x, str) for x in job["command"]):
            raise ValueError("command must be an argument list")
        if job.get("backend", "cpu") not in ("cpu", "egl", "cuda"):
            raise ValueError("backend must be cpu, egl, or cuda")
        if any(k.startswith("CUDA_") for k in job.get("env", {})):
            raise ValueError("CUDA settings are managed by the guard")
        if job.get("backend") == "cuda":
            pipe = Path(config.get("mps_pipe", ""))
            if (
                not pipe.is_absolute()
                or not pipe.is_dir()
                or pipe == Path("/tmp/nvidia-mps")
            ):
                raise ValueError(
                    "CUDA requires an explicit existing private MPS directory"
                )
            owner = pipe / "owner.json"
            if not owner.is_file():
                raise ValueError(
                    "Private MPS directory needs a campaign ownership receipt"
                )
            ownership = json.loads(owner.read_text())
            if (
                ownership.get("campaign") != "warehouse-2026-10-16"
                or ownership.get("host") != os.uname().nodename
            ):
                raise ValueError("MPS ownership must match this campaign and host")
    deadline = dt.datetime.fromisoformat(config["deadline_utc"])
    if deadline.tzinfo is None:
        raise ValueError("deadline_utc must include a UTC offset")
    return deadline.timestamp()


def run(config, output):
    deadline = validate(config)
    root = Path(config["cwd"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = output / "ledger.jsonl"
    existing_config = output / "config.json"
    if existing_config.exists() and json.loads(existing_config.read_text()) != config:
        raise ValueError("Resume requires the same frozen queue configuration")
    history = (
        [json.loads(line) for line in ledger.read_text().splitlines()]
        if ledger.exists()
        else []
    )
    completed = {row["job"] for row in history if row.get("event") == "finished"}
    for row in history:
        if row.get("event") == "started" and row["job"] not in completed:
            if row.get("host") != os.uname().nodename or not row["unit"].startswith(
                "tensorcode-campaign-"
            ):
                raise ValueError("Cannot recover a job owned by another host")
            control(row["unit"], "kill", "SIGCONT")
            control(row["unit"], "stop")
            append(
                ledger,
                {
                    "event": "finished",
                    "job": row["job"],
                    "time": time.time(),
                    "exit_code": None,
                    "reason": "worker_interrupted_no_automatic_retry",
                },
            )
            completed.add(row["job"])
    existing_config.write_text(json.dumps(config, indent=2) + "\n")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for job in config["jobs"]:
        if job["id"] in completed:
            continue
        if stopping or time.time() >= deadline:
            break
        while (
            (output / "pause-admission").exists()
            and time.time() < deadline
            and not stopping
        ):
            time.sleep(1)
        if stopping or time.time() >= deadline:
            break
        prior = sample()
        while (
            prior["memory_available_fraction"] < 0.35
            and time.time() < deadline
            and not stopping
        ):
            time.sleep(2)
            prior = sample()
        if stopping or time.time() >= deadline:
            break
        backend = job.get("backend", "cpu")
        gpu_lease = None
        if backend != "cpu":
            gpu_lease = Path("/tmp/tensorcode-campaign-gpu.lock").open("a")  # noqa: SIM115 — released in the job finally block
            mode = fcntl.LOCK_EX if backend == "cuda" else fcntl.LOCK_SH
            while True:
                try:
                    fcntl.flock(gpu_lease, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if stopping or time.time() >= deadline:
                        gpu_lease.close()
                        return
                    time.sleep(1)
        jobdir = output / job["id"]
        jobdir.mkdir(exist_ok=True)
        unit = f"tensorcode-campaign-{job['id']}-{os.getpid()}"
        backend = job.get("backend", "cpu")
        env = dict(job.get("env", {}))
        env.update(
            CUDA_VISIBLE_DEVICES="0" if backend == "cuda" else "",
            OMP_NUM_THREADS=str(len(config["cpus"])),
            OPENBLAS_NUM_THREADS=str(len(config["cpus"])),
        )
        if backend == "cuda":
            env.update(
                CUDA_MPS_PIPE_DIRECTORY=config["mps_pipe"],
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="70",
            )
        if backend == "egl":
            env.update(
                MUJOCO_GL="egl",
                LIBGL_ALWAYS_SOFTWARE="0",
                __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
            )
        command = job["command"]
        if backend == "cuda":
            command = [
                config.get("cuda_probe_python", sys.executable),
                str(Path(__file__).with_name("cuda_launch.py")),
                "--physical-sms",
                str(config.get("physical_gpu_sms", 48)),
                "--receipt",
                str(jobdir / "cuda-preflight.json"),
                "--",
                *command,
            ]
        args = [
            "systemd-run",
            "--user",
            "--wait",
            "--pipe",
            "--collect",
            f"--unit={unit}",
            "-p",
            f"CPUQuota={len(config['cpus']) * 100}%",
            "-p",
            f"MemoryMax={config['memory_gib']}G",
            "-p",
            "MemorySwapMax=0",
            "-p",
            "AllowedCPUs=" + ",".join(map(str, config["cpus"])),
            "-p",
            f"RuntimeMaxSec={max(1, int(min(deadline - time.time(), job.get('timeout_s', 14400))))}",
            "--working-directory=" + str(root),
            *[f"--setenv={k}={v}" for k, v in env.items()],
            "taskset",
            "--cpu-list",
            ",".join(map(str, config["cpus"])),
            *command,
        ]
        verified_inputs = {}
        for name, expected in job.get("inputs_sha256", {}).items():
            path = Path(name)
            if not path.is_absolute():
                path = root / path
            actual = digest(path)
            if actual != expected:
                raise ValueError(f"Frozen input hash mismatch: {name}")
            verified_inputs[name] = actual
        receipt = {
            "event": "started",
            "job": job["id"],
            "time": time.time(),
            "host": os.uname().nodename,
            "command": job["command"],
            "unit": unit,
            "backend": backend,
            "protocol": job.get("protocol", {}),
            "source_sha256": hashes(root, job.get("sources", [])),
            "inputs_sha256": verified_inputs,
            "cpus": config["cpus"],
            "memory_gib": config["memory_gib"],
            "gpu_headroom_policy": "exclusive CUDA with verified MPS<=80% equivalent-SM allocation; "
            + (
                "sampled GPU75% pause for native graphics"
                if job.get("gpu_busy_guard")
                else "busy-time sampled only"
            )
            if backend == "cuda"
            else "sampled EGL75% pause"
            if backend == "egl"
            else "CUDA disabled",
        }
        append(ledger, receipt)
        (jobdir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        reason, paused = "process_exit", False
        with (jobdir / "stdout.log").open("a") as log:
            process = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    time.sleep(1)
                    current = sample()
                    current["cpu_fraction"] = 1 - (
                        current["cpu_idle"] - prior["cpu_idle"]
                    ) / max(1, current["cpu_total"] - prior["cpu_total"])
                    current["paused"] = paused
                    append(jobdir / "resources.jsonl", current)
                    prior = current
                    if (
                        stopping
                        or time.time() >= deadline
                        or current["memory_available_fraction"] < 0.25
                    ):
                        reason = (
                            "signal"
                            if stopping
                            else "deadline"
                            if time.time() >= deadline
                            else "memory_headroom"
                        )
                        control(unit, "kill", "SIGCONT")
                        control(unit, "stop")
                        break
                    gpu = current["gpu_fraction"]
                    pressure = current["cpu_fraction"] > 0.75 or (
                        (backend == "egl" or job.get("gpu_busy_guard", False))
                        and (gpu is None or gpu > 0.75)
                    )
                    recovered = current["cpu_fraction"] < 0.60 and (
                        (backend != "egl" and not job.get("gpu_busy_guard", False))
                        or (gpu is not None and gpu < 0.60)
                    )
                    if pressure and not paused:
                        control(unit, "kill", "SIGSTOP")
                        paused = True
                    elif recovered and paused:
                        control(unit, "kill", "SIGCONT")
                        paused = False
                code = process.wait(timeout=30)
            finally:
                if process.poll() is None:
                    control(unit, "kill", "SIGCONT")
                    control(unit, "stop")
                if gpu_lease is not None:
                    gpu_lease.close()
        append(
            ledger,
            {
                "event": "finished",
                "job": job["id"],
                "time": time.time(),
                "exit_code": code,
                "reason": reason,
            },
        )
    append(
        ledger, {"event": "worker_stopped", "time": time.time(), "deadline": deadline}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.output.resolve())


if __name__ == "__main__":
    main()
