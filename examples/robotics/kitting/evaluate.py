"""Paired, seeded physics evaluation of compositional order fulfilment.

Uses simulator poses and supplied barcode/quality metadata, not learned vision.
No object is attached, teleported or overwritten after initialization.
"""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from cell import Cell, SLOTS, REJECT
from policy import Policy, TASKS, compile_policy

KINDS = np.array([0.0, 0.0, 1.0, 0.0])
SCENARIOS = {
    "nominal": dict(revise=False, reject=False, miss=False),
    "quality_reject": dict(revise=False, reject=True, miss=False),
    "order_revision": dict(revise=True, reject=False, miss=False),
    "missed_grasp": dict(revise=False, reject=False, miss=True),
    "combined": dict(revise=True, reject=True, miss=True),
    "reordered_aab": dict(
        revise=True, reject=True, miss=True, new_order=[0.0, 0.0, 1.0]
    ),
    "reordered_aba": dict(
        revise=True, reject=True, miss=True, new_order=[0.0, 1.0, 0.0]
    ),
}


def run(seed, scenario, controller, record=False, recipe=None):
    settings = SCENARIOS[scenario] if recipe is None else recipe
    new_order = settings.get("new_order", [1.0, 0.0, 0.0])
    cell = Cell(seed, record)
    mode = (
        "native"
        if controller == "native_feedback"
        else controller
        if controller.startswith("alg")
        else "alg0"
    )
    policy = Policy(mode)
    done = np.zeros(12)
    quality = np.zeros(4)
    available = np.ones(4)
    order = list(settings.get("initial_order", [0.0, 1.0]))
    actual_quality = np.ones(4)
    actual_quality[
        settings.get("reject_indices", [0] if settings["reject"] else [])
    ] = 0
    events = []
    cases = []
    attempts = 0
    retries = 0
    failed_grasps = 0
    latency = []
    revision_applied = False
    miss_used = False
    sealed = False
    fixed = controller == "fixed_recipe"
    retry = controller not in ("fixed_recipe", "no_retry")

    def log(event, **fields):
        events.append(dict(time=float(cell.data.time), event=event, **fields))

    for _ in range(40):
        if cell.data.time > 40:
            log("deadline_exceeded")
            break
        enabled = np.ones(12)
        for i in range(4):
            if done[i] and quality[i] >= 0.5:
                done[4 + i] = 1
        for slot in range(len(order), 3):
            done[8 + slot] = 1
        next_slot = next((i for i in range(len(order)) if done[8 + i] < 0.5), 0)
        inputs = dict(
            completed=done.copy(),
            enabled=enabled,
            kind=KINDS.copy(),
            quality=quality.copy(),
            available=available.copy(),
            wanted=order[next_slot],
        )
        start = time.perf_counter()
        out = policy(inputs)
        latency.append(time.perf_counter() - start)
        cases.append(
            {
                "inputs": {k: np.asarray(v).tolist() for k, v in inputs.items()},
                "outputs": {k: np.asarray(v).tolist() for k, v in out.items()},
            }
        )
        ready = np.flatnonzero(out["ready"] > 0.5)
        if not len(ready):
            log("blocked")
            break
        index = int(ready[0])
        task = list(TASKS)[index]
        if index < 4:
            quality[index] = 1.0 if fixed else actual_quality[index]
            done[index] = 1
            log(
                "inspection",
                part=index,
                quality_pass=bool(actual_quality[index]),
                used_quality_pass=bool(quality[index]),
            )
            if index == 0 and settings["revise"]:
                revision_applied = True
                log(
                    "order_changed",
                    previous=list(order),
                    current=new_order,
                    accepted=not fixed,
                )
                if not fixed:
                    order = list(new_order)
                    done[8:11] = 0
            continue
        if index == 11:
            sealed = True
            log("seal_requested")
            break
        item = index - 4 if index < 8 else int(np.argmax(out["part_score"]))
        if index >= 8 and out["part_score"][item] <= 0:
            log("inventory_shortage", task=task)
            break
        destination = REJECT if index < 8 else SLOTS[index - 8]
        attempt_miss = bool(settings["miss"] and not miss_used and index >= 8)
        miss_used |= attempt_miss
        attempts += 1
        log(
            "transfer_started",
            task=task,
            part=item,
            destination=destination.tolist(),
            injected_grasp_offset_m=0.08 if attempt_miss else 0.0,
        )
        ok = cell.transfer(item, destination, miss=attempt_miss, feedback=retry)
        log(
            "transfer_result",
            task=task,
            part=item,
            verified=ok,
            position=cell.position(item).tolist(),
        )
        if not ok:
            failed_grasps += 1
            if retry and retries < 2:
                retries += 1
                log("retry", task=task, part=item)
                continue
        # Open-loop ablations advance on commanded completion, not observed success.
        if not ok and retry:
            log("intervention_required")
            break
        available[item] = 0
        done[index] = 1
    final_order = (
        new_order if revision_applied else settings.get("initial_order", [0.0, 1.0])
    )
    final_positions = [cell.position(i).tolist() for i in range(4)]
    correct = []
    slot_parts = []
    for slot, wanted in enumerate(final_order):
        occupants = [
            i
            for i in range(4)
            if np.linalg.norm(np.array(final_positions[i])[:2] - SLOTS[slot]) < 0.05
            and 0.40 < final_positions[i][2] < 0.46
        ]
        slot_parts.append(occupants)
        correct.append(
            len(occupants) == 1
            and KINDS[occupants[0]] == wanted
            and actual_quality[occupants[0]] == 1
        )
    quarantined = all(
        np.linalg.norm(np.array(final_positions[i])[:2] - REJECT) < 0.06
        and 0.40 < final_positions[i][2] < 0.50
        for i in range(4)
        if actual_quality[i] < 0.5
    )
    success = bool(all(correct) and quarantined and sealed)
    result = dict(
        seed=seed,
        scenario=scenario,
        controller=controller,
        success=success,
        sealed=sealed,
        false_completion=bool(sealed and not success),
        correct_slots=int(sum(correct)),
        required_slots=len(final_order),
        quarantined=bool(quarantined),
        simulated_seconds=float(cell.data.time),
        attempts=attempts,
        retries=retries,
        failed_transfers=failed_grasps,
        final_order=final_order,
        slot_parts=slot_parts,
        final_positions=final_positions,
        policy_mean_ms=float(np.mean(latency) * 1000),
        policy_max_ms=float(max(latency) * 1000),
        events=events,
    )
    return result, cell, cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104, 105]
    )
    p.add_argument("--development", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    controllers = [
        "fixed_recipe",
        "no_retry",
        "native_feedback",
        "alg0",
        "alg1",
        "alg2",
    ]
    results = []
    case_count = 0
    max_error = 0.0
    artifact = compile_policy()
    (a.output / "policy.json").write_text(artifact.to_json())
    (a.output / "policy.py").write_text(artifact.to_python("policy"))
    for seed in a.seeds:
        for scenario in SCENARIOS:
            for controller in controllers:
                record = (
                    seed == a.seeds[0]
                    and scenario == "combined"
                    and controller in ("fixed_recipe", "alg2")
                )
                result, cell, cases = run(seed, scenario, controller, record)
                results.append(result)
                for case in cases:
                    expected = artifact.run(case["inputs"])
                    for key in expected:
                        max_error = max(
                            max_error,
                            float(np.max(np.abs(expected[key] - case["outputs"][key]))),
                        )
                    case_count += 1
                if record:
                    stem = a.output / controller
                    stem.with_suffix(".xml").write_text(cell.xml)
                    np.savez_compressed(
                        stem.with_suffix(".npz"),
                        qpos=np.array([f[0] for f in cell.frames]),
                        time=np.array([f[1] for f in cell.frames]),
                        labels=np.array([f[2] for f in cell.frames]),
                    )
                    stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
                    (a.output / f"{controller}-decisions.json").write_text(
                        json.dumps(cases, indent=2)
                    )
                print(seed, scenario, controller, result["success"], flush=True)
    summary = {
        c: {
            "episodes": sum(r["controller"] == c for r in results),
            "successes": sum(r["success"] for r in results if r["controller"] == c),
            "false_completions": sum(
                r["false_completion"] for r in results if r["controller"] == c
            ),
            "mean_seconds": float(
                np.mean(
                    [r["simulated_seconds"] for r in results if r["controller"] == c]
                )
            ),
            "retries": sum(r["retries"] for r in results if r["controller"] == c),
        }
        for c in controllers
    }
    report = dict(
        scope="Authored Cartesian cell; supplied poses, identity and quality; no trained perception, VLA or diffusion comparison.",
        protocol="development" if a.development else "frozen seeded evaluation",
        seeds=a.seeds,
        scenarios=SCENARIOS,
        summary=summary,
        decision_cases_checked=case_count,
        decision_max_abs_error=max_error,
        versions={
            k: importlib.metadata.version(k) for k in ("mujoco", "numpy", "torch")
        },
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        episodes=results,
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
