"""Measurement 4: does a probability on a claim do any work?

Two things are measured, both against ground truth rather than against our own judgement:

* **calibration** — the predictor from measurement 2 states a probability before each action.
  Binning those statements against what actually happened says whether the number means
  anything. A confidence that does not track being right is decoration.
* **updating** — a fact is reported repeatedly by sources of differing reliability, some of
  them wrong. Combining the reports in log-odds is compared against last-writer-wins, which
  is what a store without probability does when a new claim arrives.

The updating arm reproduces the pattern the village simulation hit (many minds, one fact,
four-way disagreement) with ground truth known by construction, rather than reading its tree.

    python -m eval.structures.probability --facts 400
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from tensorcode.expectation import Predictor, calibration, combine, disagreement
from tensorcode.outcomes import Score, Unknown

from .expectation_cw import ABSOLUTE, CHANGE, available, changes, observe, perform
from examples.browser_agents.worlds import desktop
from examples.browser_agents.worlds.runtime import CwWorld

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "structures_probability.json"


def calibration_on_screen_prediction(steps: int, seed: int) -> dict[str, Any]:
    """Every probability the predictor stated before acting, against whether it held."""
    world = CwWorld(desktop.note_world("set up a project called demo", seed), seed)
    surface = world.actor()
    predictor = Predictor()
    pairs: list[tuple[float, bool]] = []
    for step in range(steps):
        before = observe(surface)
        actions = available(surface)
        if not actions:
            break
        action = actions[step % len(actions)]
        cue = f"{action}|terminal={before['has_terminal']}"
        stated = predictor.expectation(cue, CHANGE)
        if not perform(surface, action):
            continue
        truth = changes(before, observe(surface))
        if not isinstance(stated, Unknown) and stated.p is not None:
            held = all(truth.get(aspect) == value for aspect, value in stated.predicted)
            pairs.append((stated.p.value, held))
        for aspect in CHANGE:
            predictor.observe(cue, aspect, truth[aspect])
    got = calibration(pairs)
    if isinstance(got, Unknown):
        return {"error": got.reason}
    return {"n": got.n, "ece": got.ece, "bins": [list(row) for row in got.bins],
            "mean_stated": round(sum(p for p, _ in pairs) / len(pairs), 4),
            "observed_rate": round(sum(1 for _, hit in pairs if hit) / len(pairs), 4)}


def updating(facts: int, reports: int, seed: int) -> dict[str, Any]:
    """Conflicting reports about one fact: pooled evidence against last-writer-wins."""
    rng = random.Random(seed)
    reliabilities = (0.9, 0.75, 0.6, 0.55)  # what each source is actually worth
    pooled_right = last_right = 0
    pooled_pairs: list[tuple[float, bool]] = []
    contested = 0
    for _ in range(facts):
        truth = rng.random() < 0.5
        scores: list[Score] = []
        said: list[bool] = []
        for index in range(reports):
            reliability = reliabilities[index % len(reliabilities)]
            claim = truth if rng.random() < reliability else not truth
            said.append(claim)
            # the source states its own reliability as the probability that it is right
            scores.append(Score(reliability if claim else 1 - reliability, "probability",
                                basis=f"source{index % len(reliabilities)}@known"))
        combined = combine(scores)
        if isinstance(combined, Unknown):
            continue
        believed = combined.value >= 0.5
        pooled_right += int(believed == truth)
        last_right += int(said[-1] == truth)
        pooled_pairs.append((combined.value if believed else 1 - combined.value, believed == truth))
        contested += int(disagreement(scores) > 0.3)
    calibrated = calibration(pooled_pairs)
    return {
        "facts": facts,
        "reports_per_fact": reports,
        "source_reliabilities": list(reliabilities),
        "pooled_accuracy": round(pooled_right / facts, 4),
        "last_writer_wins_accuracy": round(last_right / facts, 4),
        "contested_share": round(contested / facts, 4),
        "pooled_ece": calibrated.ece if not isinstance(calibrated, Unknown) else None,
        "note": "pooling assumes independent sources, which is stated in the score's basis; "
                "dependent sources would make the pooled number overconfident",
    }


def refusals() -> dict[str, Any]:
    """What the machinery declines to do, which is half of what makes a number meaningful."""
    mixed = combine([Score(0.8, "probability", basis="a"), Score(0.9, "similarity")])
    empty = combine([])
    unsupported = combine([Score(0.8, "probability", basis="a")], assume="dependent")
    return {
        "mixing_a_similarity_into_a_probability": getattr(mixed, "reason", "ACCEPTED"),
        "combining_nothing": getattr(empty, "reason", "ACCEPTED"),
        "an_assumption_it_cannot_honour": getattr(unsupported, "reason", "ACCEPTED"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=400)
    ap.add_argument("--reports", type=int, default=4)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    started = time.perf_counter()
    report = {
        "calibration_on_screen_prediction": calibration_on_screen_prediction(args.steps, args.seed),
        "updating_vs_last_writer_wins": updating(args.facts, args.reports, args.seed),
        "refusals": refusals(),
        "seconds": round(time.perf_counter() - started, 2),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
