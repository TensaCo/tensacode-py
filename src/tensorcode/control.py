"""Control and volition: what to pursue, what to set aside, and when to stop trying.

A mind that can only run the oldest request to completion has no control, only a queue. Three
things are missing from a queue, and each is a claim here rather than a Python variable:

* **a task set with suspended goals.** Interrupting a goal should not destroy it. Everything a
  half-finished goal needs is already in the store (its frame, its position, its bindings), so
  suspension is a *state change*, not a save: :func:`suspend` marks it and :func:`resume` marks
  it back, and the work already done is still there. Dropping a goal throws that away; the only
  reason to drop is that nobody will ever want it again.
* **arbitration among live goals.** A static priority number cannot express that one goal is
  nearly finished, that another has been waiting since three turns ago, or that switching away
  from the one in progress costs something. :func:`arbitrate` scores goals on value, urgency
  (which *ages*, so nothing starves) and cost-to-go, with hysteresis for the goal in hand, and
  records why it chose — so a choice can be argued with.
* **effort allocation.** A retry cap is a constant standing in for a judgement: is another
  attempt worth its cost? :func:`should_try_again` makes that judgement from the frequencies a
  :class:`~tensorcode.expectation.Predictor` actually observed, and refuses to invent a number
  when it has too few — an unmeasured prior is stated with its basis, never smuggled in as 3.

What this module will not do: guess. A goal with no declared value is worth
:attr:`Stance.value_default` and says so; an effort decision with no evidence reports the prior
it used; an attempt that *may already have taken effect* stops, because the cost of repeating an
effect is not comparable to the cost of one more try.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from .expectation import Predictor
from .outcomes import Score, Unknown, Verdict
from .records import Claim, Evidence, Patch, Ref, Retract, Store, Tell

#: a goal is in exactly one of these; the first two are live
ACTIVE, SUSPENDED, DONE, FAILED, ABANDONED = "active", "suspended", "done", "failed", "abandoned"
LIVE = (ACTIVE, SUSPENDED)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _source(source: str) -> Ref:
    """A source name as a Ref. Callers name themselves ("arbitrate"); refs need a kind."""
    return Ref(source if ":" in source else f"decision:{source}")


def _set(mind: Store, subject: Ref, predicate: str, value: object, source: str) -> None:
    """Replace a subject's single current value, keeping the retraction visible."""
    old = [r for r in mind.claims(subject, predicate)]
    if len(old) == 1 and old[0].claim.object == value:
        return
    edits: list = [Retract(r.id, "superseded") for r in old]
    edits.append(Tell(Claim(subject, predicate, value), (Evidence(_source(source), _now(), method="control"),)))
    commit = mind.apply(Patch(tuple(edits), mind.revision))
    mind.forget(commit.retracted)


def _one(mind: Store, subject: Ref, predicate: str, default: object = None) -> object:
    found = [r.claim.object for r in mind.claims(subject, predicate)]
    return found[0] if found else default


# ------------------------------------------------------------------ the task set


def declare(mind: Store, goal: Ref, *, what: str, source: str, value: float | None = None,
            cost_to_go: float | None = None, deadline: float | None = None) -> None:
    """Enter a goal in the task set. ``what`` is how it will be described when arbitrating."""
    _set(mind, goal, "goal:what", what, source)
    _set(mind, goal, "goal:state", ACTIVE, source)
    _set(mind, goal, "goal:since", 0.0, source)
    if value is not None:
        _set(mind, goal, "goal:value", float(value), source)
    if cost_to_go is not None:
        _set(mind, goal, "goal:cost_to_go", float(cost_to_go), source)
    if deadline is not None:
        _set(mind, goal, "goal:deadline", float(deadline), source)


def weigh(mind: Store, goal: Ref, *, source: str, what: str | None = None, value: float | None = None,
          urgency: float | None = None, cost_to_go: float | None = None, deadline: float | None = None) -> None:
    """Give a goal the weights arbitration needs, without claiming its lifecycle.

    A mind that already tracks whether a task is running (as a request status, a plan state,
    anything) should use this rather than :func:`declare`: two sources of truth for one goal's
    state is how a suspended task ends up both asleep and running.
    """
    for predicate, given in (("goal:what", what), ("goal:value", value), ("goal:urgency", urgency),
                             ("goal:cost_to_go", cost_to_go), ("goal:deadline", deadline)):
        if given is not None:
            _set(mind, goal, predicate, given if predicate == "goal:what" else float(given), source)


def waiting_since(mind: Store, goal: Ref, at: float, reason: str, *, source: str) -> None:
    """Record that a goal has been waiting, and why — the input to aging, without owning state."""
    _set(mind, goal, "goal:suspended_because", reason, source)
    _set(mind, goal, "goal:suspended_at", float(at), source)


def suspend(mind: Store, goal: Ref, reason: str, *, source: str, at: float = 0.0) -> None:
    """Set a goal aside without losing it. Its progress stays exactly where it was."""
    _set(mind, goal, "goal:state", SUSPENDED, source)
    waiting_since(mind, goal, at, reason, source=source)


def resume(mind: Store, goal: Ref, *, source: str) -> None:
    _set(mind, goal, "goal:state", ACTIVE, source)
    _set(mind, goal, "goal:resumed", True, source)


def settle(mind: Store, goal: Ref, state: str, *, source: str) -> None:
    """Finish a goal: done, failed, or abandoned (the only state that discards progress)."""
    if state not in (DONE, FAILED, ABANDONED):
        raise ValueError(f"not an ending state: {state!r}")
    _set(mind, goal, "goal:state", state, source)


def state_of(mind: Store, goal: Ref) -> str:
    return str(_one(mind, goal, "goal:state", ACTIVE))


def task_set(mind: Store, *, state: str | tuple[str, ...] = LIVE) -> list[Ref]:
    """Goals in the given state(s), oldest declaration first."""
    want = (state,) if isinstance(state, str) else tuple(state)
    return [r.claim.subject for r in mind.claims(predicate="goal:state") if r.claim.object in want]


def progress_kept(mind: Store, goal: Ref, predicates: Iterable[str]) -> dict:
    """What a suspended goal still holds — the evidence that resuming is not restarting."""
    return {p: _one(mind, goal, p) for p in predicates if _one(mind, goal, p) is not None}


# ------------------------------------------------------------------ arbitration


@dataclass(frozen=True)
class Stance:
    """How this mind weighs its goals against each other.

    ``stickiness`` is hysteresis: the goal in hand keeps an advantage, so a near-tie does not
    make the mind thrash between two goals and finish neither. ``aging`` is what stops a
    cheap-first rule from starving an expensive goal forever.
    """

    stickiness: float = 0.35
    aging: float = 0.05
    cost_weight: float = 0.15
    value_default: float = 1.0
    urgency_default: float = 0.0


@dataclass(frozen=True)
class Choice:
    """Which goal to pursue, what it scored, and what it beat."""

    goal: Ref | None
    why: str
    score: float = 0.0
    alternatives: tuple[tuple[Ref, float], ...] = ()

    def describe(self) -> str:
        if self.goal is None:
            return f"nothing to pursue: {self.why}"
        others = ", ".join(f"{g.id}@{s:.2f}" for g, s in self.alternatives)
        return f"{self.goal.id}@{self.score:.2f} — {self.why}" + (f" (over {others})" if others else "")


def urgency(mind: Store, goal: Ref, *, stance: Stance, now: float = 0.0) -> float:
    """How pressing this goal is: what was declared, plus what waiting has added.

    Aging is the honest part. Without it, ordering by cost finishes short goals first and a long
    goal can wait forever; with it, a goal's urgency rises for as long as it is passed over, so
    the worst case is bounded by how long you are willing to let something wait.
    """
    base = float(_one(mind, goal, "goal:urgency", stance.urgency_default))
    waited = max(0.0, now - float(_one(mind, goal, "goal:suspended_at", now)))
    deadline = _one(mind, goal, "goal:deadline")
    pressure = 0.0
    if isinstance(deadline, (int, float)) and deadline > now:
        pressure = 1.0 / max(1e-6, float(deadline) - now)
    return base + stance.aging * waited + pressure


def score_goal(mind: Store, goal: Ref, *, stance: Stance, now: float = 0.0, current: Ref | None = None) -> float:
    value = float(_one(mind, goal, "goal:value", stance.value_default))
    cost = float(_one(mind, goal, "goal:cost_to_go", 0.0))
    score = value + urgency(mind, goal, stance=stance, now=now) - stance.cost_weight * cost
    if current is not None and goal == current:
        score += stance.stickiness
    return score


def arbitrate(mind: Store, *, stance: Stance = Stance(), now: float = 0.0, current: Ref | None = None,
              among: Sequence[Ref] | None = None, source: str = "arbitrate") -> Choice:
    """Choose the goal to pursue now, and record the choice with its reasons."""
    goals = list(among) if among is not None else task_set(mind, state=ACTIVE)
    if not goals:
        return Choice(None, "the task set holds no active goal")
    scored = sorted(((score_goal(mind, g, stance=stance, now=now, current=current), g) for g in goals), key=lambda p: (-p[0], p[1].id))
    best_score, best = scored[0]
    parts = [f"value {float(_one(mind, best, 'goal:value', stance.value_default)):.2f}"]
    if (u := urgency(mind, best, stance=stance, now=now)):
        parts.append(f"urgency {u:.2f}")
    if (c := float(_one(mind, best, "goal:cost_to_go", 0.0))):
        parts.append(f"cost {c:.0f}")
    if current is not None and best == current:
        parts.append("in progress")
    why = ", ".join(parts)
    _set(mind, best, "goal:chosen_because", why, source)
    return Choice(best, why, best_score, tuple((g, s) for s, g in scored[1:]))


# ------------------------------------------------------------------ effort


#: how an attempt turned out, for deciding whether to make another
SUCCEEDED, TRANSIENT, AMBIGUOUS, REFUSED = "succeeded", "transient", "ambiguous", "refused"


@dataclass(frozen=True)
class Effort:
    """What a goal is worth against what another attempt costs.

    ``irreversible_cost`` is not a large number standing in for caution: an attempt whose effect
    *may already have happened* is a different kind of act, because repeating it can double the
    effect. Such an attempt is refused outright rather than priced.
    """

    value: float = 1.0
    cost: float = 0.1
    prior: Score | None = None  # used, and named, when there is too little evidence to measure
    hard_cap: int = 12  # a bound on pathology, not the policy
    irreversible_cost: float = float("inf")


def should_try_again(history: Sequence[str], *, effort: Effort = Effort(), predictor: Predictor | None = None,
                     cue: str = "attempt", aspect: str = "outcome") -> Verdict:
    """Is one more attempt worth making, given how the previous ones went?

    The verdict ``holds`` to try again, ``fails`` to stop, and carries the reasoning: an
    expectation of success, where it came from, and what it was weighed against.
    """
    attempts = len(history)
    if any(h == SUCCEEDED for h in history):
        return Verdict("fails", ("already succeeded; another attempt would repeat the effect",))
    if history and history[-1] == AMBIGUOUS:
        return Verdict("fails", ("the last attempt may already have taken effect; repeating it could double it",))
    if any(h == REFUSED for h in history):
        return Verdict("fails", ("the attempt was refused, so trying again unchanged will be refused too",))
    if attempts >= effort.hard_cap:
        return Verdict("fails", (f"{attempts} attempts reached the hard cap {effort.hard_cap}",))
    # the question is "will the next attempt succeed", so what is predicted is that one aspect.
    # Predicting the *most likely outcome* instead would read a 60% chance of a transient failure
    # as a 40% chance of success, which is only true when there are two outcomes; here there are
    # three (succeeded, transient, ambiguous), so the binary aspect is what is recorded and read.
    measured = predictor.predict(cue, f"{aspect}:succeeded") if predictor is not None else Unknown("no_predictor", "nothing observed")
    if isinstance(measured, Unknown):
        if effort.prior is None:
            return Verdict("unknown", (f"no measured chance of success ({measured.reason}) and no prior was stated",))
        p, basis = effort.prior.value, f"stated prior ({effort.prior.basis or 'unnamed'})"
    else:
        value, score = measured
        p = score.value if value is True else max(0.0, 1.0 - score.value)
        basis = f"measured {score.basis}"
    worth = p * effort.value
    if worth > effort.cost:
        return Verdict("holds", (f"chance of success {p:.2f} ({basis}) × value {effort.value:.2f} = {worth:.2f} > cost {effort.cost:.2f}",))
    return Verdict("fails", (f"chance of success {p:.2f} ({basis}) × value {effort.value:.2f} = {worth:.2f} ≤ cost {effort.cost:.2f}",))


def observe_attempt(predictor: Predictor, outcome: str, *, cue: str = "attempt", aspect: str = "outcome") -> None:
    """Record how an attempt went, so the next decision is made on frequencies rather than a constant.

    Two things are recorded: the outcome itself, which is what a person reading the record wants,
    and whether it succeeded, which is what :func:`should_try_again` asks about.
    """
    predictor.observe(cue, aspect, outcome)
    predictor.observe(cue, f"{aspect}:succeeded", outcome == SUCCEEDED)


__all__ = [
    "ACTIVE", "SUSPENDED", "DONE", "FAILED", "ABANDONED", "LIVE",
    "SUCCEEDED", "TRANSIENT", "AMBIGUOUS", "REFUSED",
    "Stance", "Choice", "Effort",
    "declare", "suspend", "resume", "settle", "state_of", "task_set", "progress_kept",
    "urgency", "score_goal", "arbitrate", "should_try_again", "observe_attempt",
]
