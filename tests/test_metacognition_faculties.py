"""The four metacognitive faculties: competence, decomposed confidence, repair, agency."""

import pytest

from tensacode.metacognition import (
    Belief, Confidences, Monitor, SelfModel, agreement, attribute, surprising,
)
from tensacode.outcomes import Score, Unknown


# ------------------------------------------------------------- competence


def test_competence_claims_the_conservative_end_of_the_interval():
    model = SelfModel()
    for _ in range(10):
        model.record(["read"], True)
    got = model.competence(["read"])
    assert got.rate == 1.0
    assert got.lower < 0.8, "ten successes is not certainty"
    assert got.score().kind == "probability" and "@10" in got.score().basis


def test_competence_backs_off_to_a_coarser_kind_it_has_history_for():
    model = SelfModel()
    for i in range(20):
        model.record([f"squad2/how_many", "squad2", "extractive_qa"], i < 4)
    # a question type never attempted still inherits the dataset's record
    got = model.competence(["squad2/where", "squad2", "extractive_qa"])
    assert got.kind == "squad2" and got.backed_off_from == "squad2/where"
    assert got.attempts == 20


def test_no_history_is_unknown_not_a_refusal():
    model = SelfModel()
    verdict = model.worth_attempting(["never_tried"], floor=0.5)
    assert verdict.status == "unknown", "an untried kind must not be reported as beyond me"
    for _ in range(10):
        model.record(["hard"], False)
    assert model.worth_attempting(["hard"], floor=0.5).status == "fails"
    for _ in range(40):
        model.record(["easy"], True)
    assert model.worth_attempting(["easy"], floor=0.5).status == "holds"


def test_spread_says_whether_a_competence_prior_could_help_at_all():
    alike = SelfModel()
    varied = SelfModel()
    for i in range(20):
        alike.record(["a"], i % 2 == 0)
        alike.record(["b"], i % 2 == 0)
        varied.record(["a"], True)
        varied.record(["b"], False)
    assert alike.spread(["a", "b"]) == pytest.approx(0.0), "identical kinds: a prior buys nothing"
    assert varied.spread(["a", "b"]) == pytest.approx(1.0)


# ------------------------------------------------- decomposed confidence


def _parts(act: float, slot: float) -> Confidences:
    return Confidences((
        Belief("speech_act", "mood", "command", Score(0.99, "vote_share", basis="t")),
        Belief("act", "delete", "delete", Score(act, "vote_share", basis="t")),
        Belief("slot", "target", "report.txt", Score(slot, "vote_share", basis="t")),
    ))


def test_a_weak_slot_becomes_a_question_while_a_weak_act_does_not():
    assert _parts(0.99, 0.99).gate(act_at=0.8, ask_below=0.2).decision == "act"
    asked = _parts(0.99, 0.5).gate(act_at=0.8, ask_below=0.2)
    assert asked.decision == "ask" and asked.about == "target"
    # too weak even to form a question about
    assert _parts(0.99, 0.1).gate(act_at=0.8, ask_below=0.2).decision == "refuse"


def test_an_unreadable_speech_act_refuses_rather_than_asking():
    weak_mood = Confidences((
        Belief("speech_act", "mood", "?", Score(0.3, "vote_share", basis="t")),
        Belief("act", "read", "read", Score(0.9, "vote_share", basis="t")),
    ))
    gate = weak_mood.gate(act_at=0.8, ask_below=0.2)
    assert gate.decision == "refuse" and "speech_act" in gate.why


def test_an_unreported_confidence_is_not_treated_as_certain():
    silent = Confidences((Belief("act", "list", "list", None),))
    assert silent.weakest().strength == 0.0
    assert silent.gate(act_at=0.5, ask_below=0.1).decision != "act"


def test_agreement_measures_readers_not_one_readers_posterior():
    assert agreement(["list", "list", "list"], basis="tiers").value == pytest.approx(1.0)
    assert agreement(["list", "read", "list"], basis="tiers").value == pytest.approx(2 / 3)
    assert agreement([None, None, None], basis="tiers").value == 0.0


# ------------------------------------------------------- error and repair


def test_a_failed_action_is_never_proposed_again_unchanged():
    m = Monitor()
    action = "echo hi | nc 10.0.0.1 80"
    seen = []
    for _ in range(4):
        m.note(action, "error_output", "command not found: nc")
        seen.append(m.repair(action, "error_output").kind)
    assert seen[0] == "retry_differently"
    assert "retry" not in seen[1:], f"a plain retry of a known-bad action was proposed: {seen}"
    assert seen[-1] == "give_up"
    assert action in m.repair(action, "error_output").avoid


def test_repair_is_chosen_by_what_kind_of_failure_it_was():
    m = Monitor()
    assert m.repair("click Send", "stale_reference").kind == "reperceive"
    assert m.repair("open missing.txt", "target_missing").kind == "ask"
    assert m.repair("wait", "timeout").kind == "retry"
    assert m.repair("anything", "crashed").kind == "give_up"


def test_a_timeout_retried_once_becomes_a_different_attempt():
    m = Monitor()
    m.note("slow command", "timeout")
    assert m.repair("slow command", "timeout").kind == "retry_differently"


# ---------------------------------------------------------------- agency


def test_a_predicted_change_is_mine_and_an_unpredicted_one_is_the_worlds():
    before = {"cwd": "~", "clock": "10:00", "files": 2}
    after = {"cwd": "~/Projects", "clock": "10:01", "files": 2}
    got = attribute(after, before=before, predicted={"cwd": "~/Projects"}, acted=True)
    assert got == {"cwd": "self", "clock": "world"}
    assert surprising(got) == ("clock",)


def test_a_change_i_touched_but_got_wrong_is_attributed_to_both():
    got = attribute({"path": "/tmp/other"}, before={"path": "/tmp/a"}, predicted={"path": "/tmp/b"})
    assert got["path"] == "both" and "path" in surprising(got)


def test_changes_while_i_did_nothing_are_the_worlds():
    got = attribute({"files": 3}, before={"files": 2}, predicted={"files": 3}, acted=False)
    assert got == {"files": "world"}, "crediting myself for a change I did not act on is the failure here"


def test_an_unchanged_aspect_is_not_attributed_at_all():
    assert attribute({"a": 1}, before={"a": 1}, predicted={"a": 1}) == {}
