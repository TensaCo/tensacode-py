"""Reproduction-only host guard; original captured runs used cgroup quotas.

Usage: python run_guarded.py LOG.json COMMAND [ARGS...]
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

log_path = Path(sys.argv[1])
command = sys.argv[2:]
if not command:
    raise SystemExit("A command is required")
allowed = sorted(os.sched_getaffinity(0))[:4]
rows = []
child = None
paused = False
started = time.monotonic()
try:
    while time.monotonic() - started < 1800:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        comfortable = cpu < 65 and ram.available / ram.total >= .25
        pressure = cpu > 75 or ram.available / ram.total < .20
        if child is None and comfortable:
            child = subprocess.Popen(
                command,
                start_new_session=True,
                preexec_fn=lambda: os.sched_setaffinity(0, allowed),
            )
        elif child is not None and child.poll() is not None:
            raise SystemExit(child.returncode)
        elif child is not None and pressure and not paused:
            os.killpg(child.pid, signal.SIGSTOP)
            paused = True
        elif child is not None and comfortable and paused:
            os.killpg(child.pid, signal.SIGCONT)
            paused = False
        rows.append({"elapsed_seconds": time.monotonic() - started,
                     "global_cpu_percent": cpu,
                     "available_ram_fraction": ram.available / ram.total,
                     "paused": paused, "child_started": child is not None,
                     "allowed_cpus": allowed})
        log_path.write_text(json.dumps(rows, indent=2) + "\n")
    raise SystemExit("30-minute guarded-run deadline exceeded")
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGCONT)
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
