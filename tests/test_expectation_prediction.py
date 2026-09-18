"""Expectations: a prediction that can be wrong, and a record of when it was."""

from tensorcode.expectation import MIN_TRIALS, Expectation, Predictor, check, expect, surprises
from tensorcode.outcomes import Score, Unknown
from tensorcode.records import Ref, Store

SCREEN = Ref("obs:screen")


def held(p: float = 0.9) -> Expectation:
    return Expectation("click:Send", (("announcement", "Saved"),), Score(p, "probability", basis="test@n=10"))


def test_a_met_expectation_holds_and_records_no_error():
    mind = Store()
    expectation = held()
    expect(mind, expectation, source=SCREEN)
    verdict, violations = check(mind, expectation, {"announcement": "Saved"}, source=SCREEN)
    assert verdict.status == "holds" and violations == ()
    assert surprises(mind) == []


def test_a_broken_expectation_records_both_sides_and_its_surprise():
    mind = Store()
    expectation = held(0.9)
    expect(mind, expectation, source=SCREEN)
    verdict, (violation,) = check(mind, expectation, {"announcement": "Error 503"}, source=SCREEN)
    assert verdict.status == "fails"
    assert (violation.expected, violation.observed) == ("Saved", "Error 503")
    assert violation.surprise > 3.0  # -log2(1 - 0.9): a confident prediction failing is surprising
    recorded = {r.claim.predicate: r.claim.object for r in mind.claims(violation.ref)}
    assert recorded["expected"] == "Saved" and recorded["observed"] == "Error 503"
    assert recorded["of"] == expectation.ref  # the error points back at what predicted it


def test_surprise_scales_with_how_sure_the_expectation_was():
    mind = Store()
    confident = check(mind, held(0.99), {"announcement": "no"}, source=SCREEN)[1][0]
    hedged = check(mind, held(0.5), {"announcement": "no"}, source=SCREEN)[1][0]
    assert confident.surprise > hedged.surprise


def test_an_unobserved_aspect_is_unknown_not_a_violation():
    """Unseen is not disconfirmed: an expectation nothing bore on stays open."""
    mind = Store()
    verdict, violations = check(mind, held(), {"something_else": 1}, source=SCREEN)
    assert verdict.status == "unknown" and violations == ()


def test_violations_come_back_most_surprising_first():
    mind = Store()
    check(mind, held(0.55), {"announcement": "a"}, source=SCREEN)
    check(mind, Expectation("click:Save", (("announcement", "Saved"),), Score(0.99, "probability", basis="t@n=9")),
          {"announcement": "b"}, source=SCREEN)
    ranked = [c.object for c in surprises(mind)]
    assert ranked == sorted(ranked, reverse=True) and len(ranked) == 2


def test_a_predictor_refuses_before_it_has_seen_enough():
    predictor = Predictor()
    for _ in range(MIN_TRIALS - 1):
        predictor.observe("click:Send", "announcement", "Saved")
    refused = predictor.predict("click:Send", "announcement")
    assert isinstance(refused, Unknown) and refused.reason == "too_few_trials"
    predictor.observe("click:Send", "announcement", "Saved")
    value, score = predictor.predict("click:Send", "announcement")
    assert value == "Saved" and score.kind == "probability" and "click:Send" in score.basis


def test_a_run_of_successes_is_not_certainty():
    predictor = Predictor()
    for _ in range(20):
        predictor.observe("click:Send", "announcement", "Saved")
    _, score = predictor.predict("click:Send", "announcement")
    assert score.value < 1.0  # Laplace: twenty hits do not license "always"


def test_a_learned_expectation_takes_the_weakest_aspect_and_names_its_basis():
    predictor = Predictor()
    for _ in range(9):
        predictor.observe("click:Send", "announcement", "Saved")
        predictor.observe("click:Send", "rows", 3)
    predictor.observe("click:Send", "rows", 4)  # rows is less reliable than announcement
    learned = predictor.expectation("click:Send", ["announcement", "rows"])
    assert learned.source == "learned"
    assert learned.p.value == min(predictor.predict("click:Send", a)[1].value for a in ("announcement", "rows"))


def test_an_expectation_over_an_unseen_cue_refuses_rather_than_inventing_one():
    assert isinstance(Predictor().expectation("click:Never", ["announcement"]), Unknown)
