"""The control library: a task set, arbitration between goals, and how long to keep trying."""

from __future__ import annotations

import tensacode as tc
from tensacode import control as C
from tensacode.expectation import Predictor
from tensacode.outcomes import Score

A, B, LONG = tc.Ref("goal:a"), tc.Ref("goal:b"), tc.Ref("goal:long")


def a_mind() -> tc.Store:
    mind = tc.Store()
    C.declare(mind, A, what="answer the question", source="test", value=1.0, cost_to_go=3.0)
    C.declare(mind, B, what="tidy the desktop", source="test", value=1.0, cost_to_go=20.0)
    return mind


# --------------------------------------------------------------------- the task set


def test_a_bare_source_name_is_accepted_as_provenance():
    mind = tc.Store()
    C.declare(mind, A, what="x", source="arbitrate")  # not "kind:name", and must still record
    assert [r.evidence[-1].source.id for r in mind.claims(A, "goal:what")] == ["decision:arbitrate"]


def test_declaring_and_settling_moves_a_goal_through_the_task_set():
    mind = a_mind()
    assert set(C.task_set(mind)) == {A, B}
    assert C.state_of(mind, A) == C.ACTIVE
    C.suspend(mind, A, "you said something else", source="test", at=2.0)
    assert C.state_of(mind, A) == C.SUSPENDED
    assert A in C.task_set(mind)  # suspended is still in the task set: that is the whole point
    assert C.task_set(mind, state=C.ACTIVE) == [B]
    C.resume(mind, A, source="test")
    assert C.state_of(mind, A) == C.ACTIVE
    C.settle(mind, A, C.DONE, source="test")
    assert C.task_set(mind) == [B]


def test_a_suspended_goal_keeps_what_it_had():
    mind = a_mind()
    C._set(mind, A, "pc", 7, "test")  # stand-in for whatever progress the goal was holding
    C.suspend(mind, A, "interrupted", source="test", at=1.0)
    assert C.progress_kept(mind, A, ["pc", "goal:suspended_because"]) == {"pc": 7, "goal:suspended_because": "interrupted"}


def test_settle_refuses_a_state_that_is_not_an_ending():
    mind = a_mind()
    try:
        C.settle(mind, A, "running", source="test")
    except ValueError as exc:
        assert "not an ending state" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("settle accepted a non-ending state")


# --------------------------------------------------------------------- arbitration


def test_waiting_raises_urgency_and_a_deadline_presses():
    mind = a_mind()
    C.waiting_since(mind, A, 0.0, "waiting", source="test")
    stance = C.Stance(aging=0.1)
    assert C.urgency(mind, A, stance=stance, now=0.0) == 0.0
    assert C.urgency(mind, A, stance=stance, now=10.0) == 1.0
    C.weigh(mind, A, source="test", deadline=12.0)
    assert C.urgency(mind, A, stance=stance, now=10.0) > 1.0  # the deadline adds pressure


def test_arbitration_prefers_the_cheaper_goal_and_records_why():
    mind = a_mind()
    choice = C.arbitrate(mind, stance=C.Stance(), now=0.0, among=[A, B], source="test")
    assert choice.goal == A
    assert "cost 3" in choice.why
    assert [r.claim.object for r in mind.claims(A, "goal:chosen_because")] == [choice.why]
    assert dict(choice.alternatives)[B] < choice.score
    assert "goal:a@" in choice.describe()


def test_stickiness_keeps_the_goal_in_hand_on_a_near_tie():
    mind = tc.Store()
    C.declare(mind, A, what="a", source="test", cost_to_go=3.0)
    C.declare(mind, B, what="b", source="test", cost_to_go=2.0)  # slightly better on cost
    assert C.arbitrate(mind, among=[A, B], current=None).goal == B
    assert C.arbitrate(mind, among=[A, B], current=A, stance=C.Stance(stickiness=0.35)).goal == A
    assert C.arbitrate(mind, among=[A, B], current=A, stance=C.Stance(stickiness=0.0)).goal == B


def test_aging_eventually_beats_a_cost_first_ordering():
    mind = tc.Store()
    C.declare(mind, LONG, what="long", source="test", cost_to_go=20.0)
    C.declare(mind, A, what="short", source="test", cost_to_go=1.0)
    C.waiting_since(mind, LONG, 0.0, "passed over", source="test")
    C.waiting_since(mind, A, 0.0, "just arrived", source="test")
    stance = C.Stance(aging=0.2, cost_weight=0.15)
    assert C.arbitrate(mind, stance=stance, now=0.0, among=[LONG, A]).goal == A
    # LONG has been waiting since 0 and A is re-declared as waiting now, so only LONG ages
    C.waiting_since(mind, A, 20.0, "just arrived", source="test")
    assert C.arbitrate(mind, stance=stance, now=20.0, among=[LONG, A]).goal == LONG


def test_arbitration_with_nothing_to_choose_says_so():
    choice = C.arbitrate(tc.Store(), among=[])
    assert choice.goal is None and "no active goal" in choice.why
    assert "nothing to pursue" in choice.describe()


# --------------------------------------------------------------------- effort


def test_no_evidence_and_no_prior_is_unknown_not_a_guess():
    v = C.should_try_again([])
    assert v.status == "unknown" and "no prior was stated" in v.reasons[0]


def test_a_stated_prior_is_used_and_named():
    effort = C.Effort(value=1.0, cost=0.2, prior=Score(0.8, "probability", basis="the app says retry"))
    v = C.should_try_again([C.TRANSIENT], effort=effort)
    assert v.status == "holds" and "the app says retry" in v.reasons[0]
    cautious = C.Effort(value=1.0, cost=0.9, prior=Score(0.8, "probability", basis="the app says retry"))
    assert C.should_try_again([C.TRANSIENT], effort=cautious).status == "fails"


def test_measured_frequency_decides_once_there_is_enough_of_it():
    p = Predictor()
    for outcome in (C.SUCCEEDED, C.SUCCEEDED, C.TRANSIENT, C.SUCCEEDED):
        C.observe_attempt(p, outcome)
    effort = C.Effort(value=1.0, cost=0.3)
    good = C.should_try_again([C.TRANSIENT], effort=effort, predictor=p)
    assert good.status == "holds" and "measured" in good.reasons[0]
    for _ in range(20):
        C.observe_attempt(p, C.TRANSIENT)
    poor = C.should_try_again([C.TRANSIENT], effort=effort, predictor=p)
    assert poor.status == "fails", poor.reasons


def test_the_probability_read_is_of_success_not_of_the_likeliest_outcome():
    """Three outcomes: 1 - p(transient) is not p(success), and the decision must not use it."""
    p = Predictor()
    for _ in range(6):
        C.observe_attempt(p, C.TRANSIENT)
    for _ in range(3):
        C.observe_attempt(p, C.AMBIGUOUS)
    for _ in range(1):
        C.observe_attempt(p, C.SUCCEEDED)
    v = C.should_try_again([C.TRANSIENT], effort=C.Effort(value=1.0, cost=0.15), predictor=p)
    # p(success) = (1+1)/(10+2) = 0.17, not 1 - p(transient) = 0.4
    assert "0.17" in v.reasons[0], v.reasons


def test_an_attempt_that_may_already_have_applied_is_refused_not_priced():
    p = Predictor()
    for _ in range(9):
        C.observe_attempt(p, C.SUCCEEDED)
    v = C.should_try_again([C.TRANSIENT, C.AMBIGUOUS], predictor=p, effort=C.Effort(value=100.0, cost=0.01))
    assert v.status == "fails" and "double" in v.reasons[0]


def test_success_and_refusal_end_the_attempts():
    assert C.should_try_again([C.SUCCEEDED]).status == "fails"
    assert "refused" in C.should_try_again([C.REFUSED, C.TRANSIENT]).reasons[0]


def test_the_hard_cap_bounds_pathology_even_with_a_hopeful_prior():
    effort = C.Effort(value=10.0, cost=0.01, prior=Score(0.99, "probability", basis="optimism"), hard_cap=4)
    assert C.should_try_again([C.TRANSIENT] * 3, effort=effort).status == "holds"
    assert "hard cap" in C.should_try_again([C.TRANSIENT] * 4, effort=effort).reasons[0]
