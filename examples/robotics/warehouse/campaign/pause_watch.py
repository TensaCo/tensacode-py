"""Maintain global guards for active child units while queue admission is paused."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from run import append, control, sample

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("host", choices=["host", "peer"])
a = parser.parse_args()
units = []
for lane in (0, 1):
    rows = [
        json.loads(x)
        for x in (a.root / f"{a.host}-lane{lane}-budget/ledger.jsonl")
        .read_text()
        .splitlines()
    ]
    units.append(next(r["unit"] for r in reversed(rows) if r["event"] == "started"))
prior = sample()
paused = set()
while units:
    time.sleep(1)
    now = sample()
    now["cpu_fraction"] = 1 - (now["cpu_idle"] - prior["cpu_idle"]) / max(
        1, now["cpu_total"] - prior["cpu_total"]
    )
    prior = now
    append(a.root / f"{a.host}-admission-pause-resources.jsonl", now)
    for unit in units[:]:
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit], check=False
        )
        if active.returncode:
            units.remove(unit)
            continue
        if now["memory_available_fraction"] < 0.25:
            control(unit, "kill", "SIGCONT")
            control(unit, "stop")
        elif (
            now["cpu_fraction"] > 0.75 or (now["gpu_fraction"] or 0) > 0.75
        ) and unit not in paused:
            control(unit, "kill", "SIGSTOP")
            paused.add(unit)
        elif (
            now["cpu_fraction"] < 0.6
            and (now["gpu_fraction"] or 0) < 0.6
            and unit in paused
        ):
            control(unit, "kill", "SIGCONT")
            paused.remove(unit)
print("All active children finished; queue workers remain paused.", flush=True)
