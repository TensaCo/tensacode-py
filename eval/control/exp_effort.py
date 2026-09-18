"""FACULTY 3 — persistence versus flexibility (how long to keep trying).

Prediction (stated before the measurement): retrying while the *measured* chance of success
times the value exceeds the cost beats a fixed retry count — more items correct where the fixed
cap gives up too early, with duplicate effects still zero.

The environment is the real access app, headless, unedited on disk. Two things are varied:

* the retry rule. ``fixed`` is the shipped rule (give up at 6 submits, or 3 that may have had an
  effect). ``expectation`` replaces only the *overall* budget with
  :func:`tensorcode.control.should_try_again` over the attempts made so far, using a
  :class:`tensorcode.expectation.Predictor` that has watched every attempt this arm has made.
  The safety rails are untouched in every arm: at most 3 attempts that may have had an effect,
  and never resubmit a request already listed in Recent submissions (the task's own constraint).
* the app's failure rate. As shipped it is 15% "nothing was saved" and an 8-point band where the
  page cannot confirm — at that rate a 6-attempt budget almost never binds, so the retry rule
  cannot matter and the honest result is parity. To find where it does matter, the *same* app is
  served with those two probability constants rewritten at serve time (nothing on disk changes,
  and the rewrite is printed in the results); `served_failure_rate` labels every row.

Two ways of reading an unconfirmed submit are measured, because which is right is not obvious:
``strict`` treats it as may-already-have-applied and stops, ``looked`` trusts the agent's reading
of Recent submissions — if the request is not listed, nothing was saved, so retrying is safe.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import tensorcode as tc
from tensorcode import control as C
from tensorcode.expectation import Predictor
from tensorcode.outcomes import Score

from examples.browser_agents import harness
from examples.browser_agents.mind import Note, objects, one
from examples.browser_agents.tasks import access

WEB = pathlib.Path(__file__).parents[2] / "examples" / "browser_agents" / "web"
ACCESS_HTML = (WEB / "access.html").read_text()
SHIPPED_TRANSIENT, SHIPPED_CONFIRMED = "roll < 0.15", "roll >= 0.23"


def html_at(rate: float) -> str:
    """The app's own page with only its two outcome probabilities rewritten."""
    assert SHIPPED_TRANSIENT in ACCESS_HTML and ACCESS_HTML.count(SHIPPED_CONFIRMED) == 2, "access.html changed shape"
    return ACCESS_HTML.replace(SHIPPED_TRANSIENT, f"roll < {rate}").replace(SHIPPED_CONFIRMED, f"roll >= {rate + 0.08:.2f}")


# ------------------------------------------------------------------ the two retry rules


class Effortful:
    """The shipped intentions, with the overall retry budget decided by expected value."""

    def __init__(self, *, resolve_by_looking: bool, effort: C.Effort) -> None:
        self.predictor = Predictor()
        self.effort = effort
        self.resolve_by_looking = resolve_by_looking
        self.seen: set[tuple[str, str]] = set()
        self.decisions: list[dict] = []
        self.gave_up: list[dict] = []

    def outcome_of(self, mind: tc.Store, ticket: tc.Ref, feedback: str) -> str:
        if feedback == "confirmed":
            return C.SUCCEEDED
        if feedback == "transient_failure":  # the page said nothing was saved
            return C.TRANSIENT
        if feedback == "unconfirmed":
            if one(mind, ticket, "listed_in_recent"):
                return C.SUCCEEDED  # it is in Recent submissions: it did save
            return C.TRANSIENT if self.resolve_by_looking else C.AMBIGUOUS
        return C.REFUSED if feedback == "invalid_input" else C.TRANSIENT

    def history(self, mind: tc.Store, ticket: tc.Ref) -> list[str]:
        return [self.outcome_of(mind, ticket, v) for _, v in objects(mind, ticket, "feedback")]

    def learn(self, mind: tc.Store, ticket: tc.Ref) -> None:
        for attempt, value in objects(mind, ticket, "feedback"):
            key = (str(attempt), str(value))
            if key in self.seen:
                continue
            self.seen.add(key)
            C.observe_attempt(self.predictor, self.outcome_of(mind, ticket, value))

    def intentions(self, mind: tc.Store) -> list[object]:
        out = access.intentions(mind)
        ticket = one(mind, access.ME, "working_on")
        if ticket is not None:
            self.learn(mind, ticket)
        submits = [i for i in out if getattr(i, "records", ()) and i.records[0].predicate == "attempt"]
        if not submits or ticket is None:
            return out
        history = self.history(mind, ticket)
        verdict = C.should_try_again(history, effort=self.effort, predictor=self.predictor)
        self.decisions.append({"ticket": ticket.id, "attempts": len(history), "history": history,
                               "verdict": verdict.status, "why": list(verdict.reasons)})
        if verdict.status == "holds":
            return out
        if verdict.status == "unknown":
            # this should not happen once a prior is stated, and giving up here would be giving up
            # for lack of *evidence about itself*, which is not a reason to stop
            self.decisions[-1]["unknown_treated_as"] = "try again"
            return out
        why = f"expected value says stop after {len(history)} attempts ({'; '.join(verdict.reasons)})"
        self.gave_up.append({"ticket": ticket.id, "attempts": len(history), "why": why})
        return [Note((tc.Claim(ticket, "gave_up", why),), (), f"give up on {ticket.id}: {why}", priority=35)]


ARMS = {
    "fixed (shipped: 6 submits, 3 that may have applied)": None,
    "expectation, unconfirmed = may have applied": dict(resolve_by_looking=False),
    "expectation, unconfirmed resolved by looking at Recent": dict(resolve_by_looking=True),
}


def run_arm(name: str, arm: dict | None, rate: float | None, seeds: range, *, effort: C.Effort) -> dict:
    from playwright.sync_api import sync_playwright

    task = harness.tasks(["access"])["access"]
    task.bindings()
    policy = Effortful(effort=effort, **arm) if arm is not None else None
    original_intentions, original_max = access.intentions, access.MAX_SUBMITS
    if policy is not None:
        # the overall budget is now the expectation's business; the rails (3 possibly-effectful
        # attempts, never resubmit a listed request) are left exactly as the task set them
        access.MAX_SUBMITS = 10**6
        task.spec.intentions = policy.intentions
    rows: list[dict] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport={"width": task.viewport[0], "height": task.viewport[1]})
            base, _ = harness.serve()
            body = html_at(rate) if rate is not None else None

            def route(r: object) -> None:
                url = r.request.url
                if body is not None and url.startswith(base) and "access.html" in url:
                    r.fulfill(status=200, content_type="text/html; charset=utf-8", body=body)
                elif url.startswith((base, "data:")):
                    r.continue_()
                else:
                    r.abort()

            context.route("**/*", route)
            page = context.new_page()
            for seed in seeds:
                started = time.perf_counter()
                r = harness.run_episode(context, base, task, seed, page=page)
                rows.append({"seed": seed, "items": r.score.get("items", 0), "correct": r.score.get("correct", 0),
                             "duplicates": r.score.get("duplicates", 0), "actions": r.actions,
                             "status": getattr(r.outcome, "status", "error"), "reason": getattr(r.outcome, "reason", r.error),
                             "seconds": round(time.perf_counter() - started, 2), "error": r.error,
                             "model_calls": r.model_calls})
                print(f"    seed {seed}: {rows[-1]['correct']}/{rows[-1]['items']} correct, "
                      f"{rows[-1]['duplicates']} duplicates, {rows[-1]['actions']} actions, {rows[-1]['status']}")
            browser.close()
    finally:
        access.intentions, access.MAX_SUBMITS = original_intentions, original_max
        task.spec.intentions = original_intentions
    items, correct = sum(r["items"] for r in rows), sum(r["correct"] for r in rows)
    return {
        "arm": name, "served_failure_rate": rate if rate is not None else 0.15,
        "served": "app as shipped" if rate is None else f"app with roll<{rate} transient, {rate + 0.08:.2f} confirm floor",
        "episodes": len(rows), "items": items, "correct": correct,
        "accuracy": round(correct / items, 4) if items else None,
        "duplicates": sum(r["duplicates"] for r in rows),
        "actions_per_episode": round(sum(r["actions"] for r in rows) / max(1, len(rows)), 1),
        "escalated": sum(1 for r in rows if r["status"] == "escalated"),
        "model_calls": sum(r["model_calls"] for r in rows),
        "gave_up": (policy.gave_up if policy else []),
        "decisions": (policy.decisions[-8:] if policy else []),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--rates", default="shipped,0.45,0.65")
    ap.add_argument("--cost", type=float, default=0.2)
    ap.add_argument("--prior", type=float, default=0.5,
                    help="the named prior used before there are three observations to measure")
    ap.add_argument("--out", default="eval/results/control_effort.json")
    args = ap.parse_args()
    effort = C.Effort(value=1.0, cost=args.cost, hard_cap=12,
                      prior=Score(args.prior, "probability", basis="stated: even odds before the first three attempts"))
    out = {"faculty": "persistence vs flexibility",
           "prediction": "expectation-driven retries beat a fixed count where the cap binds, duplicates stay 0",
           "provenance": {"environment": "the repo's own access app, headless; probability constants rewritten "
                                         "at serve time for the raised-failure arms (nothing on disk edited)",
                          "grader": "the app's own window.__score (environment author's grader)",
                          "held_out": "seeds are the same in every arm; no arm was tuned on them"},
           "effort": {"value": effort.value, "cost": effort.cost, "hard_cap": effort.hard_cap,
                      "prior": f"{args.prior} ({effort.prior.basis})"},
           "arms": []}
    for rate_text in args.rates.split(","):
        rate = None if rate_text == "shipped" else float(rate_text)
        for name, arm in ARMS.items():
            print(f"\n=== {name}  |  failure rate {rate_text}")
            row = run_arm(name, arm, rate, range(1, args.episodes + 1), effort=effort)
            out["arms"].append(row)
            print(f"  → {row['correct']}/{row['items']} = {row['accuracy']}, duplicates {row['duplicates']}, "
                  f"{row['actions_per_episode']} actions/episode, gave up {len(row['gave_up'])}")
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
