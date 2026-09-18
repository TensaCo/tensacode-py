"""Measurement 3: does intervention buy anything over co-occurrence?

The engine can fork a checkpoint, so the same starting state can be run with an action and
without it. That is a real ``do(·)``, and it makes the difference between "these happen
together" and "this brings that about" measurable rather than arguable.

Three arms, and the third is the one that matters:

* **correlational** — what co-occurred with the action in a plain trajectory, no control;
* **passive control** — the same state, with the action against doing nothing;
* **active control** — the same state, with the action against *a different* action (a click
  on empty desktop). Anything caused by merely acting — a logical clock tick, a step
  counter — happens in both branches and cancels, leaving what this action specifically did.

A shuffle control re-runs the scoring with the action labels permuted: if effect sizes stay
high when the labels are wrong, the measurement is picking up structure that is not there.

    python -m eval.structures.causal_cw --trials 2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from tensorcode.causal import Contrast, correlations, distinguish, experiment, learn, moved, tell_causal
from tensorcode.records import Ref, Store

from examples.browser_agents.worlds import desktop
from examples.browser_agents.worlds.runtime import CwWorld

from .expectation_cw import LAUNCHERS, observe, perform, point_of

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "structures_causal.json"

ASPECTS = ("title", "has_terminal", "interactions", "nodes", "clock")


def _with_clock(surface: Any) -> dict[str, Any]:
    """The observation, plus a value that moves on its own whenever anything happens.

    The engine's clock is the natural confound in this world: it advances when the world
    steps, so it co-occurs with every action and a correlational reader will call every
    action its cause.
    """
    out = observe(surface)
    scene = surface.scene()
    clock = next((str(n.get("text")) for n in scene.get("nodes", ()) if n.get("role") == "clock" or "clock" in str(n.get("id", ""))), None)
    out["clock"] = clock if clock is not None else scene.get("revision")
    return {k: out[k] for k in ASPECTS}


def empty_point(surface: Any) -> tuple[int, int]:
    """A point with nothing interactive under it, for the active control."""
    return (surface.width - 4, surface.height // 2)


def run(trials: int, seed: int) -> dict[str, Any]:
    world = CwWorld(desktop.note_world("set up a project called demo", seed), seed)
    start = world.snapshot()
    mind = Store()

    def fresh() -> Any:
        branch = world.fork(start)
        return branch.actor()

    contrasts: dict[str, list[Contrast]] = {"passive": [], "active": []}
    for action in [f"click:{name}" for name in LAUNCHERS]:
        contrasts["passive"] += experiment(
            prepare=fresh, act=lambda s, a=action: perform(s, a), observe=_with_clock, cause=action, trials=trials,
        )
        contrasts["active"] += experiment(
            prepare=fresh, act=lambda s, a=action: perform(s, a), observe=_with_clock, cause=action,
            trials=trials, control=_click_empty,  # the control acts too, so "acting at all" cancels
        )

    # correlational baseline: one plain trajectory, no control at all
    surface = world.fork(start).actor()
    observations: list[tuple[set[str], dict[str, Any]]] = []
    for step in range(len(LAUNCHERS) * trials * 2):
        action = f"click:{LAUNCHERS[step % len(LAUNCHERS)]}"
        before = _with_clock(surface)
        if not perform(surface, action):
            continue
        after = _with_clock(surface)
        observations.append(({action}, moved(before, after)))

    report: dict[str, Any] = {"seed": seed, "trials": trials, "aspects": list(ASPECTS), "arms": {}}
    for arm, rows in contrasts.items():
        links = learn(rows, least_effect=0.5, basis=f"intervention/{arm}")
        for link in links:
            tell_causal(mind, link, source=Ref("obs:experiment"))
        report["arms"][arm] = {
            "contrasts": len(rows),
            "causal_links": len(links),
            "by_aspect": _aspect_effects(rows),
            "examples": [link.describe() for link in links[:6]],
        }

    correlated: list[Any] = []
    for action in [f"click:{name}" for name in LAUNCHERS]:
        correlated += correlations(observations, cause=action, least=0.5)
    report["correlational"] = {
        "observations": len(observations),
        "links": len(correlated),
        "by_aspect": _count_by_aspect(correlated),
        "examples": [link.describe() for link in correlated[:6]],
    }
    report["verdicts_vs_correlation"] = distinguish(contrasts["active"], correlated)
    report["shuffle_control"] = _shuffled(contrasts["active"], seed)
    report["clock_case"] = _clock_case(contrasts, correlated)
    return report


def _click_empty(surface: Any) -> None:
    """The control action: a click where nothing interactive sits."""
    x, y = empty_point(surface)
    surface.act("pointer.v1", "click", {"width": surface.width, "height": surface.height, "x": x, "y": y})


def _aspect_effects(rows: list[Contrast]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        best[row.aspect] = max(best.get(row.aspect, 0.0), row.effect)
    return {k: round(v, 3) for k, v in sorted(best.items())}


def _count_by_aspect(links: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for link in links:
        out[link.aspect] = out.get(link.aspect, 0) + 1
    return dict(sorted(out.items()))


def _shuffled(rows: list[Contrast], seed: int) -> dict[str, Any]:
    """Re-score with the action labels permuted: effect sizes should collapse."""
    rng = random.Random(seed)
    causes = [row.cause for row in rows]
    rng.shuffle(causes)
    swapped = [Contrast(cause, row.aspect, row.with_act, row.without_act) for cause, row in zip(causes, rows)]
    # the contrast is between branches, so permuting labels cannot change an effect size;
    # what it changes is *which cause* each effect is attributed to. Count how many links
    # survive with a cause that never produced them.
    real = {(row.cause, row.aspect) for row in rows if row.effect >= 0.5}
    shuffled = {(row.cause, row.aspect) for row in swapped if row.effect >= 0.5}
    return {
        "links_real": len(real),
        "links_shuffled": len(shuffled),
        "misattributed": len(shuffled - real),
        "note": "permuting labels leaves effect sizes intact but misattributes them; the "
                "count of pairs that appear only in the shuffled set is the damage a "
                "correlational reader cannot detect",
    }


def _clock_case(contrasts: dict[str, list[Contrast]], correlated: list[Any]) -> dict[str, Any]:
    """The correlation-vs-cause case: the clock moves whenever anything happens."""
    def effect(arm: str) -> float:
        return max((row.effect for row in contrasts[arm] if row.aspect == "clock"), default=0.0)

    return {
        "clock_in_correlational_links": any(link.aspect == "clock" for link in correlated),
        "clock_effect_passive_control": round(effect("passive"), 3),
        "clock_effect_active_control": round(effect("active"), 3),
        "reading": "a passive control credits the action with the clock; an active control, "
                   "where both branches act, shows whether this action specifically moved it",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    started = time.perf_counter()
    report = run(args.trials, args.seed)
    again = run(args.trials, args.seed)
    report["identical_across_runs"] = report["arms"] == again["arms"]
    report["seconds"] = round(time.perf_counter() - started, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1)[:3000])


if __name__ == "__main__":
    main()
