"""Measurement 2: does an agent that predicts the next screen get better at predicting it?

The engine is deterministic and renders in milliseconds, so ground truth is free and exact:
act, look, compare. This runs a long trajectory, forms an expectation before every action
from nothing but its own record of what happened last time, and scores it.

Two cue shapes are compared, because the difference is the interesting part:

* ``action`` — "clicking Terminal does X", which ignores the state it was clicked in;
* ``action|state`` — the same action conditioned on whether a terminal is already open.

If conditioning wins, the flat cue was collapsing distinct situations, which is exactly the
mistake a bag of rules makes when it has no notion of context.

    python -m eval.structures.expectation_cw --steps 240
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tensacode.expectation import Expectation, Predictor, check, expect
from tensacode.outcomes import Unknown
from tensacode.records import Ref, Store

from examples.browser_agents.worlds import desktop
from examples.browser_agents.worlds.runtime import CwWorld

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "structures_expectation.json"

#: absolute-value aspects: what the screen *is* after the action
ABSOLUTE = ("title", "interactions", "nodes", "has_terminal", "lines")
#: change aspects: what the action *did*. A count that drifts (a clock, a panel) has no
#: stable absolute value to predict, but "opening a window adds nodes" is stable — so the
#: two encodings are measured side by side rather than one being assumed.
CHANGE = ("title_changed", "interactions_changed", "nodes_direction", "has_terminal", "lines_direction")
LAUNCHERS = ("shell:launch:terminal", "shell:launch:editor", "shell:launch:files", "shell:launcher", "shell:panel:calendar")


def point_of(scene: dict, interaction: str) -> tuple[int, int] | None:
    """Screen coordinates of an interactive node, through its own transform."""
    for node in reversed(scene.get("nodes", ())):
        if node.get("interaction") != interaction:
            continue
        bounds, t = node.get("bounds") or {}, node.get("transform") or {"a": 1024, "b": 0, "c": 0, "d": 1024, "tx": 0, "ty": 0}
        x = bounds.get("x", 0) + bounds.get("width", 0) // 2
        y = bounds.get("y", 0) + bounds.get("height", 0) // 2
        return int((t["a"] * x + t["c"] * y) / 1024 + t["tx"]), int((t["b"] * x + t["d"] * y) / 1024 + t["ty"])
    return None


def observe(surface: Any) -> dict[str, Any]:
    """The aspects of the screen an expectation may be about."""
    scene = surface.scene()
    interactions = tuple(sorted({n.get("interaction") for n in scene.get("nodes", ()) if n.get("interaction")}))
    return {
        "title": surface.title(),
        "interactions": interactions,
        "nodes": len(scene.get("nodes", ())),
        "has_terminal": surface.has_terminal(),
        "lines": len(surface.terminal_lines()),
    }


def _sign(after: float, before: float) -> str:
    return "up" if after > before else "down" if after < before else "same"


def changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The same observation expressed as what the action changed."""
    return {
        "title_changed": after["title"] != before["title"],
        "interactions_changed": after["interactions"] != before["interactions"],
        "nodes_direction": _sign(after["nodes"], before["nodes"]),
        "has_terminal": after["has_terminal"],
        "lines_direction": _sign(after["lines"], before["lines"]),
    }


def available(surface: Any) -> list[str]:
    """The actions worth trying on this screen, as stable cue strings."""
    scene = surface.scene()
    present = {n.get("interaction") for n in scene.get("nodes", ()) if n.get("interaction")}
    acts = [f"click:{name}" for name in LAUNCHERS if name in present]
    if surface.has_terminal():
        acts += ["type:ls", "key:Enter", "type:pwd"]
    return acts


def perform(surface: Any, action: str) -> bool:
    kind, _, argument = action.partition(":")
    if kind == "click":
        point = point_of(surface.scene(), argument)
        if point is None:
            return False
        ok, _ = surface.act("pointer.v1", "click", {"width": surface.width, "height": surface.height, "x": point[0], "y": point[1]})
        return bool(ok)
    if kind == "type":
        ok, _ = surface.act("keyboard.v1", "type", {"text": argument})
        return bool(ok)
    if kind == "key":
        ok, _ = surface.act("keyboard.v1", "key", {"key": argument})
        return bool(ok)
    return False


@dataclass
class Trial:
    step: int
    action: str
    cue: str
    predicted: dict[str, Any] | None
    observed: dict[str, Any]
    hit: bool | None  # None when it refused to predict
    wrong_aspects: tuple[str, ...] = ()


@dataclass
class Arm:
    """One cue shape and encoding, with its own predictor and record."""

    name: str
    with_state: bool
    encoding: str  # "absolute" | "change"
    predictor: Predictor = field(default_factory=Predictor)
    trials: list[Trial] = field(default_factory=list)

    @property
    def aspects(self) -> tuple[str, ...]:
        return ABSOLUTE if self.encoding == "absolute" else CHANGE

    def cue(self, action: str, before: dict[str, Any]) -> str:
        return f"{action}|terminal={before['has_terminal']}" if self.with_state else action


def run(steps: int, seed: int) -> tuple[dict[str, Any], str]:
    world = CwWorld(desktop.note_world("set up a project called demo", seed), seed)
    surface = world.actor()
    mind, source = Store(), Ref("obs:screen")
    arms = [
        Arm("absolute/action", False, "absolute"),
        Arm("absolute/action+state", True, "absolute"),
        Arm("change/action", False, "change"),
        Arm("change/action+state", True, "change"),
    ]

    for step in range(steps):
        before = observe(surface)
        actions = available(surface)
        if not actions:
            break
        action = actions[step % len(actions)]
        predictions: dict[str, Expectation | Unknown] = {}
        for arm in arms:
            predictions[arm.name] = arm.predictor.expectation(arm.cue(action, before), arm.aspects)
        if not perform(surface, action):
            continue
        after = observe(surface)
        delta = changes(before, after)
        for arm in arms:
            cue = arm.cue(action, before)
            truth = after if arm.encoding == "absolute" else delta
            prediction = predictions[arm.name]
            if isinstance(prediction, Unknown):
                arm.trials.append(Trial(step, action, cue, None, truth, None))
            else:
                expect(mind, prediction, source=source)
                verdict, violations = check(mind, prediction, truth, source=source)
                arm.trials.append(Trial(step, action, cue, dict(prediction.predicted), truth,
                                        verdict.status == "holds", tuple(v.aspect for v in violations)))
            for aspect in arm.aspects:
                arm.predictor.observe(cue, aspect, truth[aspect])
    return {arm.name: _score(arm) for arm in arms}, world.state_hash()


def _score(arm: Arm) -> dict[str, Any]:
    attempted = [t for t in arm.trials if t.hit is not None]
    refused = [t for t in arm.trials if t.hit is None]
    hits = [t for t in attempted if t.hit]
    half = len(attempted) // 2
    per_action: dict[str, dict[str, int]] = {}
    for trial in attempted:
        bucket = per_action.setdefault(trial.action, {"n": 0, "hit": 0})
        bucket["n"] += 1
        bucket["hit"] += 1 if trial.hit else 0
    per_aspect: dict[str, int] = {}
    for trial in attempted:
        for aspect in trial.wrong_aspects:
            per_aspect[aspect] = per_aspect.get(aspect, 0) + 1
    right_by_aspect = {
        aspect: round(sum(1 for t in attempted if aspect not in t.wrong_aspects) / len(attempted), 4)
        for aspect in arm.aspects
    } if attempted else {}
    return {
        "trials": len(arm.trials),
        "refused_too_few_trials": len(refused),
        "attempted": len(attempted),
        "accuracy_all_aspects": round(len(hits) / len(attempted), 4) if attempted else 0.0,
        "accuracy_first_half": round(sum(1 for t in attempted[:half] if t.hit) / half, 4) if half else 0.0,
        "accuracy_second_half": round(sum(1 for t in attempted[half:] if t.hit) / (len(attempted) - half), 4) if attempted[half:] else 0.0,
        "wrong_by_aspect": dict(sorted(per_aspect.items(), key=lambda kv: -kv[1])),
        "accuracy_by_aspect": right_by_aspect,
        "by_action": {k: {**v, "accuracy": round(v["hit"] / v["n"], 3)} for k, v in sorted(per_action.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    started = time.perf_counter()
    arms, state_hash = run(args.steps, args.seed)
    again, again_hash = run(args.steps, args.seed)
    report = {
        "steps": args.steps,
        "seed": args.seed,
        "aspects": {"absolute": list(ABSOLUTE), "change": list(CHANGE)},
        "arms": arms,
        "deterministic": state_hash == again_hash,
        "identical_across_runs": arms == again,
        "seconds": round(time.perf_counter() - started, 2),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
