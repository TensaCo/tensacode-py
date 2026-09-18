"""Record every decision an agent faced, not only the one it took.

    PYTHONPATH=src:. python eval/training/rollout_decisions.py --out FILE [--episodes 60]

The recordings we already have keep the chosen intention, which is not enough to learn a
chooser: a ranker needs the options that were rejected. So this replays episodes with a
``choose`` implementation that writes down the whole candidate set, delegates to the
hand-written objective, and then stamps every decision of an episode with that episode's
outcome as scored by the app itself. That makes the file a behaviour-cloning set with a
real reward attached, mined from a chooser we already trust.

One deliberate omission: ``priority`` is recorded but is NOT offered as a feature to the
learned ranker. The hand-written objective *is* priority, so a model given it learns
nothing except to copy a number. Everything else an intention carries is fair game.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path[:0] = [str(Path(__file__).parents[2] / "src"), str(Path(__file__).parents[2])]

import tensacode as tc  # noqa: E402
from examples.browser_agents import harness  # noqa: E402
from examples.browser_agents.browser import LabelMatcher, classify_feedback  # noqa: E402
from tensacode.backends.builtin import UtilityChooser  # noqa: E402

TASKS = ("access", "shop", "recon", "chart")


def describe_intention(i: object) -> dict:
    """What a chooser could look at, without peeking at the hand-written priority."""
    kind = type(i).__name__
    out = {
        "kind": kind,
        "why": str(getattr(i, "why", "") or getattr(i, "reason", ""))[:160],
        "priority": float(getattr(i, "priority", 0.0)),  # recorded for audit, not offered as a feature
        "records": len(getattr(i, "records", ()) or ()),
        "submits": bool(getattr(i, "submit", False)),
        "text_len": len(str(getattr(i, "text", "") or "")),
        "has_text": bool(getattr(i, "text", "")),
        "control": str(getattr(getattr(i, "control", None), "id", "") or "")[:120],
        "claims": len(getattr(i, "claims", ()) or ()),
        "wait_ms": float(getattr(i, "ms", 0.0) or 0.0),
    }
    return out


class RecordingChooser:
    """Delegates to the hand-written objective and records the decision it was asked to make."""

    name = "recording-utility-argmax"
    version = "1"
    op = "choose"

    def __init__(self, sink: list[dict]) -> None:
        self.sink = sink
        self.inner = UtilityChooser()
        self.traits = self.inner.traits
        self.profile = self.inner.profile
        self.episode: dict = {}

    def accepts(self, request: tc.Request) -> bool:
        return self.inner.accepts(request)

    def run(self, requests: Sequence[tc.Request]) -> list:
        outs = self.inner.run(requests)
        for request, out in zip(requests, outs):
            options = list(request.subject)
            chosen = out.value
            index = next((i for i, o in enumerate(options) if o is chosen or o == chosen), None)
            self.sink.append({
                **self.episode,
                "candidates": [describe_intention(o) for o in options],
                "chosen": index,
                "n": len(options),
            })
        return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--first-seed", type=int, default=1)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    rows: list[dict] = []
    t0 = time.perf_counter()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        base, _ = harness.serve()
        for name in args.tasks.split(","):
            task = harness.tasks([name])[name]
            task.bindings()
            context = browser.new_context(viewport={"width": task.viewport[0], "height": task.viewport[1]})
            harness.block_outside_network(context, base)
            page = context.new_page()
            for seed in range(args.first_seed, args.first_seed + args.episodes):
                sink: list[dict] = []
                chooser = RecordingChooser(sink)
                chooser.episode = {"task": name, "seed": seed}
                runtime = tc.Runtime(
                    [LabelMatcher(), chooser, classify_feedback, *task.bindings()],
                    policy=tc.Policy(localities=frozenset({"in_process"}), allow_egress=False, available=frozenset({"sklearn"}), cache=False),
                )
                ui_stats = {}
                with tc.use(runtime):
                    from examples.browser_agents.browser import Browser
                    from examples.browser_agents.mind import run_mind

                    if task.setup:
                        task.setup(seed)
                    page.goto(task.url(base, seed))
                    ui = Browser(page, episode=f"{name}-{seed}")
                    try:
                        outcome = run_mind(ui, task.spec)
                        error = None
                    except Exception as exc:  # noqa: BLE001
                        outcome, error = None, f"{type(exc).__name__}: {exc}"
                    ui_stats = {"actions": ui.stats.actions, "observations": ui.stats.observations}
                score = task.score(page, seed) if task.score else (page.evaluate("() => window.__score ? window.__score() : null") or {})
                items, correct = score.get("items", 0), score.get("correct", 0)
                for row in sink:
                    row.update({
                        "items": items, "correct": correct,
                        "episode_ok": bool(items and correct == items),
                        "status": getattr(outcome, "status", "error"),
                        "actions": ui_stats.get("actions", 0),
                        "error": error,
                    })
                rows += sink
            context.close()
        browser.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    tasks = {}
    for row in rows:
        t = tasks.setdefault(row["task"], {"decisions": 0, "episodes": set(), "ok": set()})
        t["decisions"] += 1
        t["episodes"].add(row["seed"])
        if row["episode_ok"]:
            t["ok"].add(row["seed"])
    print(json.dumps({
        "out": str(args.out), "rows": len(rows), "seconds": round(time.perf_counter() - t0, 1),
        "per_task": {k: {"decisions": v["decisions"], "episodes": len(v["episodes"]), "episodes_fully_correct": len(v["ok"])} for k, v in tasks.items()},
        "multi_option_decisions": sum(1 for r in rows if r["n"] > 1),
    }, indent=1))
    _ = dataclasses


if __name__ == "__main__":
    main()
