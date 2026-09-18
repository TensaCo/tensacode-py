"""The four social structures, tested as structures: what they represent, not how they read.

Each test names the cognitive claim it holds to. The point of a test here is that the claim is
falsifiable in isolation — if common ground makes no behavioural difference it is bookkeeping,
so the behavioural difference is asserted in tests/test_social_faculties.py instead.
"""

from __future__ import annotations

import tensorcode as tc
from tensorcode.cognition import Fragment, integrate
from tensorcode.social import (
    BOTH_SAW,
    I_SAID,
    YOU_SAID,
    CommonGround,
    Implication,
    Indirect,
    Reading,
    OtherMind,
    Unknown,
    ask_or_act,
    indirect_reading,
    near_names,
    uncertainty,
)


def _mind_with_fact() -> tuple[tc.Store, str]:
    mind = tc.Store()
    frag = Fragment(tc.Ref("utterance:1"), ((tc.Claim(tc.Ref("person:user"), "name", "Jacob"), None),),
                    method="told")
    integrate(mind, frag)
    return mind, mind.claims(tc.Ref("person:user"), "name")[0].id


# ------------------------------------------------------------- common ground


def test_a_fact_i_hold_is_not_yet_shared():
    """Holding a belief and sharing it are different states; storage alone is not ground."""
    mind, about = _mind_with_fact()
    ground = CommonGround(mind)
    assert ground.is_shared(about) is False
    assert ground.new_to_other(about) is True


def test_saying_it_makes_it_shared_and_the_repeat_is_countable():
    mind, about = _mind_with_fact()
    ground = CommonGround(mind)
    ground.add(about, I_SAID, 2, source="reply:2")
    assert ground.is_shared(about)
    assert ground.mentions(about, I_SAID) == 1
    ground.add(about, I_SAID, 3, source="reply:3")
    assert ground.mentions(about, I_SAID) == 2
    assert ground.status(about).turn == 3  # the repeat supersedes rather than piling up


def test_again_asks_my_own_ground_not_the_strongest_one():
    """That you told me a thing is no reason for me to say "as I mentioned"."""
    mind, about = _mind_with_fact()
    ground = CommonGround(mind)
    ground.add(about, YOU_SAID, 1, source="utterance:1")
    assert ground.status(about).how == YOU_SAID
    assert ground.again(about) is False  # you said it; I have not
    ground.add(about, I_SAID, 2, source="reply:2")
    assert ground.again(about) is True


def test_jointly_seen_outranks_said():
    mind, about = _mind_with_fact()
    ground = CommonGround(mind)
    ground.add(about, I_SAID, 1, source="reply:1")
    ground.add(about, BOTH_SAW, 2, source="screen:2")
    assert ground.status(about).how == BOTH_SAW


def test_told_me_is_ordered_by_the_exchange():
    """"What did I tell you" is a question about the exchange, so the answer follows it."""
    mind = tc.Store()
    ids = []
    for turn, (pred, value) in enumerate([("name", "Jacob"), ("colour", "green")], start=1):
        integrate(mind, Fragment(tc.Ref(f"utterance:{turn}"),
                                 ((tc.Claim(tc.Ref("person:user"), pred, value), None),), method="told"))
        ids.append(mind.claims(tc.Ref("person:user"), pred)[0].id)
    ground = CommonGround(mind)
    ground.add(ids[1], YOU_SAID, 4, source="utterance:4")  # told second, out of storage order
    ground.add(ids[0], YOU_SAID, 1, source="utterance:1")
    assert [s.about for s in ground.told_me()] == ids


def test_grounding_is_claims_so_it_can_be_explained():
    mind, about = _mind_with_fact()
    CommonGround(mind).add(about, YOU_SAID, 1, source="utterance:1")
    assert mind.claims(predicate="shared_via", object=YOU_SAID)
    assert mind.claims(predicate="grounds", object=about)


# --------------------------------------------------------------- near misses


def test_a_wrong_name_finds_its_near_miss():
    pool = ["notes.txt", "recipes/", "reports.txt", "photo.png"]
    assert near_names("report.txt", pool) == ["reports.txt"]
    assert near_names("notse.txt", pool) == ["notes.txt"]
    assert near_names("recipe", pool) == ["recipes/"]


def test_nothing_near_stays_nothing():
    """The structure has to be able to say "no": otherwise a correction is a guess."""
    assert near_names("zzzqqq.txt", ["notes.txt", "photo.png"]) == []


def test_the_name_you_asked_for_is_not_its_own_near_miss():
    assert near_names("notes.txt", ["notes.txt"]) == []


# ------------------------------------------------------- asking as a decision


def test_uncertainty_is_zero_when_one_reading_is_certain():
    assert uncertainty([Reading("a", 1.0), Reading("b", 0.0)]) == 0.0
    assert uncertainty([Reading("a", 0.5), Reading("b", 0.5)]) == 1.0


def test_asking_is_priced_not_reflexive():
    """A dominant reading is acted on; a split one is worth a question."""
    split = [Reading("by type", 0.45), Reading("by date", 0.35), Reading("just list", 0.20)]
    decision = ask_or_act(split, question="How would you like it organized?", about="~/Desktop")
    assert decision.choose == "ask" and decision.expected_gain > 0
    assert decision.clarification is not None and len(decision.clarification.options) == 3

    clear = [Reading("read it", 0.95), Reading("delete it", 0.05)]
    assert ask_or_act(clear).choose == "act"


def test_a_question_that_costs_more_than_it_saves_is_not_asked():
    split = [Reading("a", 0.5), Reading("b", 0.5)]
    assert ask_or_act(split, ask_cost=5.0).choose == "act"


# ------------------------------------------------- intention behind the words


def _implications() -> list[Implication]:
    return [Implication(r"\b(?:i )?can'?t find (?:my |the )?(?P<obj>[\w. -]{2,40})", "complaint", "find", 0.8, "pattern")]


def test_the_same_form_reads_differently_by_what_i_can_act_on():
    """Form cannot separate these two: the discriminator is whether the object is mine to act on."""
    mine = indirect_reading("I can't find my invoice", _implications(), in_domain=lambda w: w == "invoice")
    assert isinstance(mine, Indirect) and mine.act == "find" and mine.slots == {"pattern": "invoice"}

    theirs = indirect_reading("I can't find my keys", _implications(), in_domain=lambda w: w == "invoice")
    assert isinstance(theirs, Unknown) and theirs.reason == "not_my_domain"


def test_no_implication_matches_means_silence_not_a_guess():
    assert isinstance(indirect_reading("the weather is awful", _implications(), in_domain=lambda w: True), Unknown)


# ------------------------------------------------------ what a request assumes


def test_a_request_carries_presuppositions_that_can_be_checked():
    other = OtherMind(tc.Store())
    presupposed = other.presupposes({"target": "report.txt", "place": "~/Desktop"},
                                    {"target": "exists", "place": "location"})
    assert any(p.kind == "exists" and p.subject == "report.txt" for p in presupposed)


def test_a_false_presupposition_becomes_a_correction_with_the_near_miss():
    other = OtherMind(tc.Store())
    presupposed = other.presupposes({"target": "report.txt"}, {"target": "exists"})[0]
    false = other.check(presupposed, exists=lambda _n: False, neighbours=lambda _n: ["reports.txt"])
    assert false is not None
    assert "did you mean" in false.correction().lower() and "reports.txt" in false.correction()


def test_a_presupposition_that_holds_produces_no_correction():
    other = OtherMind(tc.Store())
    presupposed = other.presupposes({"target": "notes.txt"}, {"target": "exists"})[0]
    assert other.check(presupposed, exists=lambda _n: True) is None
