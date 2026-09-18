"""The decision layer: typed answers, gating that refuses the wrong kind of number,
citable refund reasons, and replay.

These are contract tests, not accuracy tests — accuracy lives in eval/decisions/measure.py
against public labels. Each test here pins a property the example claims.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import tensorcode as tc
from tensorcode.outcomes import Score, Unknown, Verdict

from examples.decisions import decisions, tiers
from examples.decisions.audit import Audit, replay
from examples.decisions.domain import Charge, Department, Passage, Ticket, Urgency
from examples.decisions.gating import Gate, Thresholds, gates
from examples.support_router.domain import Intent

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def ticket(text: str, *, amount: float = 12.5, days: int = 3, status: str = "settled", duplicate: str | None = "C-0") -> Ticket:
    return Ticket(
        "T-1",
        "someone@example.com",
        text,
        NOW,
        (Charge("C-1", "Acme Ltd", amount, NOW - timedelta(days=days), status, duplicate),),
    )


@pytest.fixture
def rules_runtime() -> tc.Runtime:
    return tiers.runtime(tiers.rules_only())


# -------------------------------------------------------------- gating


def test_a_gate_refuses_to_threshold_a_score_that_is_not_a_probability():
    """A relevance or similarity score is not P(correct); thresholding it is a type error."""
    gate = Gate(Thresholds("money", auto=0.95, confirm=0.70))
    for kind in ("relevance", "similarity", "uncalibrated", "utility", "vote_share"):
        decided = gate.decide(True, Score(0.99, kind))
        assert decided.action == "escalate"
        assert kind in decided.why
    allowed = gate.decide(True, Score(0.99, "probability", "some/validation@2026"))
    assert allowed.action == "auto"


def test_unknown_escalates_and_is_not_treated_as_zero_confidence():
    gate = Gate(Thresholds("routing", auto=0.60, confirm=0.30))
    decided = gate.decide(Unknown("below_threshold", "p=0.4"), None)
    assert decided.action == "escalate" and decided.confidence is None
    assert "below_threshold" in decided.why


def test_an_answer_with_no_confidence_escalates_rather_than_being_trusted():
    """The measured failure that ScoredKeywords exists to fix."""
    gate = Gate(Thresholds("routing", auto=0.60, confirm=0.30))
    assert gate.decide(Intent.pin_blocked, None).action == "escalate"


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        Thresholds("bad", auto=0.4, confirm=0.9)


def test_a_gate_built_from_measurement_disables_auto_when_nothing_reaches_the_target():
    curve = [(0.0, 0.5, 100), (0.5, 0.7, 50), (0.9, 0.8, 10)]
    gate = Gate.from_measurement("money", curve, target_accuracy=0.99, confirm_at=0.3, basis="test curve")
    assert gate.t.auto == 1.0 and "NO measured threshold" in gate.t.basis
    assert gate.decide(True, Score(0.999, "probability", "b")).action == "confirm"


def test_a_gate_built_from_measurement_picks_the_lowest_threshold_reaching_the_target():
    curve = [(0.0, 0.80, 100), (0.5, 0.94, 60), (0.7, 0.96, 40), (0.9, 0.99, 10)]
    gate = Gate.from_measurement("routing", curve, target_accuracy=0.95, confirm_at=0.3, basis="test curve")
    assert gate.t.auto == 0.7


# ------------------------------------------------------------- triage


def test_triage_leaves_everything_unknown_when_the_intent_is_unknown(rules_runtime):
    """No default department: an unclassified ticket is not quietly 'general'."""
    with tc.use(rules_runtime):
        out = decisions.triage(ticket("zzzz qqqq vvvv"))
    assert isinstance(out.intent, Unknown)
    assert isinstance(out.department, Unknown) and isinstance(out.urgency, Unknown)
    assert out.refund_asked.status == "unknown"
    assert out.routing.action == "escalate"


def test_triage_derives_department_urgency_and_refund_from_a_known_intent(rules_runtime):
    with tc.use(rules_runtime):
        out = decisions.triage(ticket("my pin is blocked after too many tries"))
    assert out.intent is Intent.pin_blocked
    assert out.department is Department.cards
    assert out.urgency is Urgency.critical
    assert out.refund_asked.status == "fails"  # a blocked pin does not ask for money back


def test_classify_with_confidence_keeps_the_score_the_facade_drops():
    """Documents the library gap: tc.classify returns the value only."""
    runtime = tiers.runtime(tiers.rules_only())
    with tc.use(runtime):
        plain = tc.classify("my pin is blocked after too many tries", Intent)
        answer, score, span = decisions.classify_with_confidence("my pin is blocked after too many tries", Intent)
    assert plain is Intent.pin_blocked and answer is Intent.pin_blocked
    assert span is not None and span.answered_by == "intent-keyword-rules@2"
    assert score is None  # the keyword tier reports no confidence; see ScoredKeywords


# -------------------------------------------------------- refund policy


def test_a_duplicate_charge_within_policy_is_allowed_and_cites_its_clause(rules_runtime):
    with tc.use(rules_runtime):
        result = decisions.refund_eligibility(ticket("charged twice"), now=NOW)
    assert result.allowed and result.clauses == ("R1",)
    assert "C-1" in result.reasons[0] and "C-0" in result.reasons[0]


@pytest.mark.parametrize(
    "kwargs, clause",
    [
        ({"status": "pending"}, "R2"),
        ({"status": "reversed"}, "R3"),
        ({"days": 200}, "R4"),
        ({"amount": 500.0}, "R5"),
    ],
)
def test_each_refusal_names_the_clause_that_refused_it(rules_runtime, kwargs, clause):
    with tc.use(rules_runtime):
        result = decisions.refund_eligibility(ticket("charged twice", **kwargs), now=NOW)
    assert not result.allowed and clause in result.clauses


def test_no_duplicate_charge_is_unknown_not_a_refusal(rules_runtime):
    """'I cannot tell' is not 'no'. A ticket with no duplicate needs a human, not a denial."""
    with tc.use(rules_runtime):
        result = decisions.refund_eligibility(ticket("charged twice", duplicate=None), now=NOW)
    assert isinstance(result, Unknown) and result.reason == "no_duplicate_charge"


# -------------------------------------------------------------- rank


def test_rerank_orders_by_relevance_and_labels_the_score_kind(rules_runtime):
    passages = [
        Passage("P1", "Transfers", "Transfers arrive within three working days."),
        Passage("P2", "Refunds", "A charge billed twice may be refunded up to fifty pounds."),
    ]
    with tc.use(rules_runtime):
        ranked = decisions.rerank("refund a duplicate charge", passages)
    assert [p.id for p, _ in ranked] == ["P2", "P1"]
    assert all(score.kind == "relevance" for _, score in ranked)


def test_support_has_three_values_including_an_explicit_unknown(rules_runtime):
    passage = Passage("P1", "Refunds", "A charge billed twice may be refunded up to fifty pounds.")
    with tc.use(rules_runtime):
        assert decisions.supports("a charge billed twice may be refunded", passage).status == "holds"
        assert decisions.supports("penguins live in antarctica", passage).status == "fails"
        middling = decisions.supports("refunded pounds antarctica penguins colony", passage)
    assert middling.status == "unknown"


# ------------------------------------------------------- audit and replay


def test_a_decision_records_its_citations_and_their_sources(rules_runtime):
    audit = Audit()
    with tc.use(rules_runtime):
        decisions.triage(ticket("my pin is blocked after too many tries"), audit=audit)
        decisions.refund_eligibility(ticket("charged twice"), now=NOW, audit=audit)
    why = audit.why("T-1")
    assert why["clauses"] == ["R1"] and why["charges"] == ["C-1"]
    assert why["sources"]["intent"] == "impl:intent-keyword-rules@2"
    assert why["sources"]["refund_allowed"] == "policy:refund"
    assert any("refund_allowed" in line for line in audit.explain("T-1"))


def test_replay_of_stored_state_reproduces_the_decision(rules_runtime):
    audit = Audit()
    with tc.use(rules_runtime):
        first = decisions.handle(ticket("my pin is blocked after too many tries"), audit=audit)
    again = replay(audit.inputs["T-1"], runtime=tiers.runtime(tiers.rules_only()))
    assert again["intent"] == first["intent"] and again["department"] == first["department"]
    assert again["refund_asked"] == first["refund_asked"]


def test_the_handler_reports_which_tier_answered_and_counts_no_model_calls(rules_runtime):
    with tc.use(rules_runtime):
        out = decisions.handle(ticket("my pin is blocked after too many tries"))
        _, routed = decisions.route(ticket("my pin is blocked after too many tries"))
    assert out["tier"] == "intent-keyword-rules@2"
    assert routed.answered_by == "intent-keyword-rules@2"
    assert not any(impl.startswith("chat:") for impl, _, _ in routed.attempts)


def test_scored_keywords_turn_an_ungateable_answer_into_a_gateable_one():
    """The measured fix: same rules, now reporting precision measured on a holdout."""
    from tensorcode.backends.builtin import KeywordClassifier

    rules = KeywordClassifier(Intent, {Intent.pin_blocked: [r"\bpin\b.*\bblocked\b"]}, name="tiny-rules", version="1")
    validation = [("my pin is blocked", Intent.pin_blocked), ("pin is blocked again", Intent.pin_blocked)]
    scored = tiers.ScoredKeywords.measure(rules, validation, basis="test holdout")
    runtime = tiers.runtime([scored])
    gate = Gate(Thresholds("routing", auto=0.60, confirm=0.30))
    with tc.use(runtime):
        answer, score, _ = decisions.classify_with_confidence("my pin is blocked", Intent)
    assert answer is Intent.pin_blocked
    assert score is not None and score.kind == "probability" and "test holdout" in score.basis
    assert gate.decide(answer, score).action == "auto"


def test_a_rule_never_seen_on_validation_abstains_rather_than_claiming_certainty():
    from tensorcode.backends.builtin import KeywordClassifier

    rules = KeywordClassifier(Intent, {Intent.age_limit: [r"\bhow old\b"]}, name="tiny-rules", version="1")
    scored = tiers.ScoredKeywords.measure(rules, [("unrelated text", Intent.pin_blocked)], basis="empty holdout")
    with tc.use(tiers.runtime([scored])):
        answer, score, _ = decisions.classify_with_confidence("how old must i be", Intent)
    assert isinstance(answer, Unknown) and answer.reason == "unmeasured_rule"


def test_default_gates_are_ordered_by_consequence():
    table = gates()
    assert table["money"].t.auto > table["reply"].t.auto > table["routing"].t.auto
