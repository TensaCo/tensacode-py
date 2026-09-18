"""Scaling of the reference in-memory Store for the structural operations the examples use.

    python eval/graph_bench.py

Synthetic, uniform data (devices, rooms, status claims from three sources over a day).
This bounds the prototype's costs; it is not a database benchmark.
"""

from __future__ import annotations

import json
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import tensorcode as tc  # noqa: E402


@dataclass(frozen=True)
class Device:
    serial: str
    room: tc.Ref


T0 = datetime(2026, 9, 16, tzinfo=timezone.utc)


def build(n_devices: int, rng: random.Random) -> tuple[tc.Store, float, int]:
    reg = tc.TypeRegistry()
    reg.register(Device)
    w = tc.Store(reg)
    w.declare("status", functional=True)
    claims = 0
    t = time.perf_counter()
    for i in range(n_devices):
        dev = tc.Ref(f"dev:{i}")
        room = tc.Ref(f"room:{i // 20}")
        w.put(dev, Device(f"S{i}", room))
        w.tell(tc.Claim(dev, "in", room), tc.Evidence(tc.Ref("obs:inventory"), T0))
        claims += 1
        for k in range(8):
            start = T0 + timedelta(minutes=rng.randrange(0, 1440))
            status = rng.choice(["up", "up", "up", "down"])
            src = rng.choice(["poller", "ticket", "agent"])
            w.tell(tc.Claim(dev, "status", status, tc.Interval(start, start + timedelta(minutes=rng.randrange(1, 90)))), tc.Evidence(tc.Ref(f"obs:{src}-{i}-{k}"), start))
            claims += 1
    return w, time.perf_counter() - t, claims


def timeit(fn, reps: int) -> dict:
    xs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        xs.append(1e3 * (time.perf_counter() - t))
    return {"p50_ms": round(float(np.percentile(xs, 50)), 4), "p95_ms": round(float(np.percentile(xs, 95)), 4)}


def main() -> None:
    rng = random.Random(0)
    results = []
    for n in (125, 1250, 12500):
        tracemalloc.start()
        w, build_s, claims = build(n, rng)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        noon = T0 + timedelta(hours=12)
        dev = lambda: tc.Ref(f"dev:{rng.randrange(n)}")  # noqa: E731
        row = {
            "devices": n,
            "claims": claims,
            "build_claims_per_s": round(claims / build_s),
            "peak_traced_mb": round(peak / 2**20, 1),
            "claims(subject, predicate, at)": timeit(lambda: w.claims(dev(), "status", at=noon), 2000),
            "conflicts(subject)": timeit(lambda: w.conflicts(dev()), 2000),
            "neighborhood(depth=1)": timeit(lambda: w.neighborhood(dev(), 1), 500),
            "match(room members down @noon)": timeit(lambda: w.match((tc.Var("d"), "in", tc.Ref(f"room:{rng.randrange(max(1, n // 20))}")), (tc.Var("d"), "status", "down"), at=noon), 200),
            "patch SetField": timeit(lambda: w.apply(tc.Patch((tc.SetField(dev(), ("serial",), "X"),), w.revision)), 200),
        }
        t = time.perf_counter()
        data = json.dumps(w.to_json())
        row["to_json_s"] = round(time.perf_counter() - t, 3)
        row["json_mb"] = round(len(data) / 2**20, 2)
        reg = tc.TypeRegistry()
        reg.register(Device)
        t = time.perf_counter()
        restored, report = tc.Store.from_json(json.loads(data), reg)
        row["from_json_s"] = round(time.perf_counter() - t, 3)
        row["round_trip_lossless"] = report.lossless and len(restored.claims()) == len(w.claims())
        t = time.perf_counter()
        row["all_conflicts"] = len(w.conflicts())
        row["all_conflicts_s"] = round(time.perf_counter() - t, 3)
        results.append(row)
        print(json.dumps(row))
    out = ROOT / "eval/results/graph_bench.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
