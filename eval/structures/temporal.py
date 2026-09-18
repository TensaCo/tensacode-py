"""Measurement 5: can the graph answer *when* and *in what order*?

An agent session is recorded — each action with the moment it happened, each belief with the
moment it came on record — and then queried. Ground truth is the recording itself, so the
questions ("what did you do before the commit", "what changed since my last message") have
exact answers and a wrong one cannot hide.

Undated events are mixed in deliberately. An agent that orders them anyway is inventing
history, so the measurement counts refusals separately from mistakes.

    python -m eval.structures.temporal --actions 40
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from tensorcode.outcomes import Unknown
from tensorcode.records import Claim, Evidence, Ref, Store
from tensorcode.temporal import (after, before, changed_since, during, event_time, events, order,
                                ordered_by_claims, relate, since, tell_event, tell_order)

from .expectation_cw import available, observe, perform
from examples.browser_agents.worlds import desktop
from examples.browser_agents.worlds.runtime import CwWorld

OUT = Path(__file__).resolve().parents[2] / "eval" / "results" / "structures_temporal.json"
START = datetime(2026, 9, 17, 9, 0, tzinfo=timezone.utc)


def record(actions: int, seed: int) -> tuple[Store, list[tuple[Ref, datetime]], list[Ref], datetime]:
    """Drive a session, writing an event per action and a claim per observed change."""
    world = CwWorld(desktop.note_world("set up a project called demo", seed), seed)
    surface = world.actor()
    mind = Store()
    timeline: list[tuple[Ref, datetime]] = []
    undated: list[Ref] = []
    midpoint = START
    for step in range(actions):
        options = available(surface)
        if not options:
            break
        action = options[step % len(options)]
        before_state = observe(surface)
        if not perform(surface, action):
            continue
        at = START + timedelta(minutes=step)
        ref = Ref(f"event:step{step}")
        tell_event(mind, ref, at=at, kind="act", source=Ref("obs:session"))
        timeline.append((ref, at))
        if step == actions // 2:
            midpoint = at
        after_state = observe(surface)
        for aspect, value in after_state.items():
            if value != before_state.get(aspect):
                mind.tell(Claim(Ref(f"screen:{aspect}"), "became", str(value)[:80]),
                          Evidence(Ref("obs:session"), at, method="session"))
        if step % 7 == 3:  # an event someone mentioned but never dated
            vague = Ref(f"event:mentioned{step}")
            mind.tell(Claim(vague, "is_a", "act"), Evidence(Ref("obs:session"), at, method="hearsay"))
            undated.append(vague)
    return mind, timeline, undated, midpoint


def ask(mind: Store, timeline: list[tuple[Ref, datetime]], undated: list[Ref], midpoint: datetime) -> dict[str, Any]:
    """Every question type, scored against the recorded order."""
    refs = [ref for ref, _ in timeline]
    times = dict(timeline)
    results: dict[str, dict[str, int]] = {}

    def tally(name: str, right: bool, refused: bool = False) -> None:
        row = results.setdefault(name, {"asked": 0, "right": 0, "wrong": 0, "refused": 0})
        row["asked"] += 1
        row["refused" if refused else ("right" if right else "wrong")] += 1

    for index, ref in enumerate(refs):
        expected_before = {r.id for r in refs[:index]}
        got_before = {e.ref.id for e in before(mind, ref, kind="act")}
        tally("before", got_before == expected_before)

        expected_after = {r.id for r in refs[index + 1:]}
        got_after = {e.ref.id for e in after(mind, ref, kind="act")}
        tally("after", got_after == expected_after)

        got_since = since(mind, ref, kind="act")
        tally("since", isinstance(got_since, list) and {e.ref.id for e in got_since} == expected_after)

        for other in refs[max(0, index - 3):index + 4]:
            if other == ref:
                continue
            expected = "before" if times[ref] < times[other] else "after"
            tally("relate", relate(mind, ref, other) == expected)

    window_start, window_end = START + timedelta(minutes=2), midpoint
    expected_window = {r.id for r, t in timeline if window_start <= t <= window_end}
    tally("during", {e.ref.id for e in during(mind, window_start, window_end, kind="act")} == expected_window)

    changed = changed_since(mind, midpoint)
    expected_changed = all(
        min(e.observed_at for e in mind.claim(c.id).evidence) > midpoint for c in changed
    )
    tally("changed_since", bool(changed) and expected_changed)

    for ref in undated:
        got = event_time(mind, ref)
        tally("undated_event_time", isinstance(got, Unknown), refused=isinstance(got, Unknown))
        placed, unplaced = order(mind, [ref, refs[0]])
        tally("undated_kept_out_of_order", [r.id for r in unplaced] == [ref.id])

    # an ordering asserted by language, with no clock on either side
    tell_order(mind, Ref("event:grain"), Ref("event:snow"), source=Ref("obs:speech"))
    tally("language_ordering", ordered_by_claims(mind, Ref("event:grain"), Ref("event:snow")) == "before")
    tally("language_ordering_unknown", isinstance(ordered_by_claims(mind, Ref("event:grain"), Ref("event:other")), Unknown),
          refused=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    started = time.perf_counter()
    mind, timeline, undated, midpoint = record(args.actions, args.seed)
    results = ask(mind, timeline, undated, midpoint)
    asked = sum(row["asked"] for row in results.values())
    right = sum(row["right"] for row in results.values())
    wrong = sum(row["wrong"] for row in results.values())
    refused = sum(row["refused"] for row in results.values())
    report = {
        "events_recorded": len(timeline),
        "undated_events": len(undated),
        "claims": len(mind.claims()),
        "queries": asked,
        "right": right,
        "wrong": wrong,
        "refused_because_undated": refused,
        "accuracy_where_answerable": round(right / (asked - refused), 4) if asked - refused else 0.0,
        "by_query": results,
        "seconds": round(time.perf_counter() - started, 2),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
