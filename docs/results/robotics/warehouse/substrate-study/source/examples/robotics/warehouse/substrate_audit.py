"""Frozen-protocol dispatch parity and CPU latency audit; no simulator inputs.

Run from the Python repository with ``python -m
examples.robotics.warehouse.substrate_audit``. Fixture records model the public
observation interface and are explicitly synthetic, never robot outcome evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

from .coordination import Assignment, Coordinator, RobotState
from .language import Task
from .representation import Entity, SceneGraph

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "docs/results/robotics/warehouse/substrate-study"


def fixtures(protocol):
    """Deterministic records generated only after reading the frozen protocol."""
    base = {
        "age": 1.0,
        "confidence": 1.0,
        "position": [0.0, 0.0, 0.0],
        "robots": [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        "gate": "open",
    }
    cases = []

    def add(identity, expected=None, **changes):
        cases.append({**base, **changes, "id": identity, "expected_robot": expected})

    add("ordinary_exact_tie", "a")
    for label, age, expected in [
        ("below", math.nextafter(2.0, -math.inf), "a"),
        ("at", 2.0, "a"),
        ("above", math.nextafter(2.0, math.inf), None),
        ("future", -math.ulp(1.0), None),
    ]:
        add(f"freshness_{label}", expected, age=age)
    for label, confidence, expected in [
        ("below", math.nextafter(0.7, -math.inf), None),
        ("at", 0.7, "a"),
        ("above", math.nextafter(0.7, math.inf), "a"),
    ]:
        add(f"confidence_{label}", expected, confidence=confidence)
    for gate, expected in [
        ("wrong_pair", "b"),
        ("occupied", None),
        ("failed_dependency", None),
        ("unavailable", "b"),
    ]:
        add(gate, expected, gate=gate)
    add(
        "ulp_cost_a_farther",
        "b",
        robots=[[math.nextafter(1.0, math.inf), 0, 0], [1, 0, 0]],
    )
    add(
        "ulp_cost_b_farther",
        "a",
        robots=[[1, 0, 0], [math.nextafter(1.0, math.inf), 0, 0]],
    )
    add(
        "finite_large_tie",
        "a",
        robots=[[1e150, 1e150, 1e150], [-1e150, -1e150, -1e150]],
    )
    add("finite_small_rounded_tie", "a", robots=[[1e-200, 0, 0], [0, 0, 0]])
    rng = random.Random(protocol["seed"])
    for i in range(protocol["random_cases"]):
        add(
            f"seeded_{i:03d}",
            gate=rng.choice(
                ["open", "wrong_pair", "occupied", "failed_dependency", "unavailable"]
            ),
            age=rng.choice([rng.uniform(0, 3), math.nextafter(2.0, math.inf), 2.0]),
            confidence=rng.choice([0.69, 0.7, 1.0]),
            position=[rng.uniform(-10, 10) for _ in range(3)],
            robots=[[rng.uniform(-10, 10) for _ in range(3)] for _ in range(2)],
        )
    return cases


def prepare(case):
    scene = SceneGraph()
    # now=0 preserves nextafter boundaries exactly, avoiding subtraction rounding.
    for identity, kind, position in [
        ("parcel", "parcel", case["position"]),
        ("receiver", "station", [2, 1, 0]),
    ]:
        scene.upsert(
            Entity(
                identity,
                kind,
                {},
                tuple(position),
                -case["age"],
                f"synthetic-audit/{case['id']}",
                case["confidence"],
            )
        )
    tasks = [
        Task("move", "parcel", "receiver", (), scene.revision, ("synthetic-audit",))
    ]
    if case["gate"] == "failed_dependency":
        tasks = [
            Task(
                "dependency",
                "parcel",
                "receiver",
                (),
                scene.revision,
                ("synthetic-audit",),
            ),
            Task(
                "move",
                "parcel",
                "receiver",
                ("dependency",),
                scene.revision,
                ("synthetic-audit",),
            ),
        ]
    robots = [
        RobotState(
            identity,
            tuple(position),
            available=not (i == 0 and case["gate"] == "unavailable"),
            allowed_object_ids=("parcel",),
            allowed_destination_ids=("receiver",),
            allowed_pairs=(("parcel", "different_receiver"),)
            if i == 0 and case["gate"] == "wrong_pair"
            else (("parcel", "receiver"),),
        )
        for i, (identity, position) in enumerate(zip(("a", "b"), case["robots"]))
    ]
    return tasks, robots, scene


def reset(coordinator, case):
    coordinator.completed.clear()
    coordinator.failed = (
        {"dependency"} if case["gate"] == "failed_dependency" else set()
    )
    coordinator.active = {}
    if case["gate"] == "occupied":
        coordinator.active["external"] = Assignment(
            "external", "external_robot", "other", "other_dest", ("aisle",), 0.0
        )


def dispatch(coordinator, prepared):
    tasks, robots, scene = prepared
    return coordinator.dispatch(
        tasks, robots, scene, now=0.0, routes={"move": ("aisle",)}
    )


def evaluate(case, coordinator):
    reset(coordinator, case)
    result = dispatch(coordinator, prepare(case))
    return {"assignments": [asdict(a) for a in result], "gates": coordinator.last_gates}


def percentile(values, quantile):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * quantile) - 1)]


def tie_probe(output, machines):
    """Separate exploratory probe; never folded into primary parity denominators."""
    import itertools

    import numpy as np
    import torch

    protocol_bytes = (output / "tie-probe-protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    rng = random.Random(protocol["seed"])
    retained = []
    screened = 0
    for i in range(protocol["vectors"]):
        vector = [rng.uniform(0.1, 10) for _ in range(3)]
        permutations = list(itertools.permutations(vector))
        scores = [
            (
                float(np.square(p).sum()),
                float(torch.tensor(p, dtype=torch.float64).square().sum()),
            )
            for p in permutations
        ]
        for j, k in itertools.combinations(range(len(permutations)), 2):
            screened += 1
            numpy_choice = "a" if scores[j][0] <= scores[k][0] else "b"
            torch_choice = "a" if scores[j][1] <= scores[k][1] else "b"
            if numpy_choice == torch_choice:
                continue
            case = {
                "id": f"exploratory_tie_{i}_{j}_{k}",
                "age": 1.0,
                "confidence": 1.0,
                "position": [0.0, 0.0, 0.0],
                "gate": "open",
                "robots": [list(permutations[j]), list(permutations[k])],
            }
            results = {s: evaluate(case, c) for s, c in machines.items()}
            chosen = {
                s: result["assignments"][0]["robot_id"] for s, result in results.items()
            }
            retained.append(
                {
                    "case": case,
                    "results": results,
                    "chosen": chosen,
                    "actual_dispatch_mismatch": len(set(chosen.values())) > 1,
                }
            )
    return {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "screened_pairs": screened,
        "screened_disagreements": len(retained),
        "actual_dispatch_mismatches": sum(
            r["actual_dispatch_mismatch"] for r in retained
        ),
        "counterexamples": retained,
    }


def run(output):
    import torch

    protocol_path = output / "protocol.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    affinity = sorted(os.sched_getaffinity(0))[: protocol["cpu_affinity_count"]]
    os.sched_setaffinity(0, affinity)
    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(protocol["threads"])
    cases = fixtures(protocol)
    machines = {s: Coordinator(substrate=s) for s in protocol["substrates"]}
    rows, mismatches, numeric = [], [], []
    started = time.time()
    with torch.inference_mode():
        for case in cases:
            results = {s: evaluate(case, c) for s, c in machines.items()}
            reference = results["alg0"]
            for substrate, result in results.items():
                if result["assignments"] != reference["assignments"]:
                    mismatches.append(
                        {
                            "case": case["id"],
                            "substrate": substrate,
                            "kind": "assignments",
                        }
                    )
                for task, robots in reference["gates"].items():
                    for robot, gates in robots.items():
                        actual = result["gates"][task][robot]
                        if {k: v for k, v in actual.items() if k != "cost"} != {
                            k: v for k, v in gates.items() if k != "cost"
                        }:
                            mismatches.append(
                                {
                                    "case": case["id"],
                                    "substrate": substrate,
                                    "kind": "gates",
                                }
                            )
                        if actual["cost"] != gates["cost"]:
                            numeric.append(
                                {
                                    "case": case["id"],
                                    "substrate": substrate,
                                    "task": task,
                                    "robot": robot,
                                    "alg0_cost": gates["cost"],
                                    "actual_cost": actual["cost"],
                                }
                            )
            if not case["id"].startswith("seeded_"):
                chosen = (
                    reference["assignments"][0]["robot_id"]
                    if reference["assignments"]
                    else None
                )
                if chosen != case["expected_robot"]:
                    raise AssertionError((case["id"], chosen, case["expected_robot"]))
            rows.append({"case": case, "results": results})
        timings = {s: [] for s in machines}
        prepared_cases = [(case, prepare(case)) for case in cases]
        for repeat in range(-protocol["warmup_passes"], protocol["timing_passes"]):
            for i, (case, prepared) in enumerate(prepared_cases):
                names = protocol["substrates"]
                offset = (repeat + i) % len(names)
                for substrate in names[offset:] + names[:offset]:
                    coordinator = machines[substrate]
                    reset(coordinator, case)
                    start = time.perf_counter_ns()
                    dispatch(coordinator, prepared)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if repeat >= 0:
                        timings[substrate].append(elapsed)
    with torch.inference_mode():
        probe = tie_probe(output, machines)
    (output / "tie-counterexamples.json").write_text(
        json.dumps(probe, indent=2, allow_nan=False) + "\n"
    )
    files = [
        Path(__file__),
        Path(__file__).with_name("coordination.py"),
        Path(__file__).with_name("representation.py"),
        ROOT / "src/tensorcode/algorithm/tensor.py",
        ROOT / "src/tensorcode/algorithm/runtime.py",
    ]
    report = {
        "schema": "tensorcode.warehouse.substrate-audit.v1",
        "started_unix": started,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "case_count": len(cases),
        "decision_mismatches": mismatches,
        "numerical_cost_differences": numeric,
        "all_sampled_decisions_equal": not mismatches,
        "timing_ms": {
            s: {
                "samples": len(v),
                "p50": percentile(v, 0.5),
                "p95": percentile(v, 0.95),
                "max": max(v),
            }
            for s, v in timings.items()
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_affinity": affinity,
            "torch_threads": torch.get_num_threads(),
            "load_average_start_note": "Concurrent workloads uncontrolled; not a hard real-time measurement.",
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        "limitations": protocol["claim_limit"],
    }
    for name, value in [
        ("results.json", report),
        ("cases.json", rows),
        ("timings.json", timings),
    ]:
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
