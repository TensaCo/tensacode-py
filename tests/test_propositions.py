"""What a claim has to be able to say, and what it must never say.

The store used to hold binary subject-predicate-object claims. Anything with more than
two participants was *reified*: "the meeting is on Tuesday" became an invented
``event:…`` node with a ``subject`` claim and a ``time`` claim hung off it. Retrieval
then had to hop through those nodes, and the hop is what answered "how many properties
does the meeting have?" with a list of adverbs — any event has an object, so hopping on
one returns whatever happens to be stored.

These tests fix the replacement: one proposition per predication, with the sentence's own
roles, matched by pattern. They are about the *shape* of what can be said, so they are
written against the store and the converter directly, not through a particular reader.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

import tensorcode as tc
from agent_test_support import selected_agent as Agent
from tensorcode.agent.understand import LearnedReader
from tensorcode.records import Proposition, Var, matches


def _grounded_turn(agent, text, roles, *, speech_label):
    """Authored occurrence bindings isolate downstream mechanisms, not inference."""
    from tensorcode.agent.core import InterpretationDecision
    from tensorcode.agent.grounding import MentionBinding, propose_grounding
    from tensorcode.records import Ref
    learned = isinstance(agent.reader, LearnedReader)
    if learned:
        from dependency_meaning_fixtures import teach_speech_family
        teach_speech_family(agent, text, speech_label)

    evidence = agent.interpretations.add_source(
        "Test fixture explicitly supplies occurrence identities", provider="test-fixture")

    def select(group):
        if learned and speech_label.kind == 'question':
            for _ in range(32):
                if not agent.interpretations.continuation_status(group.id).pending:
                    break
                agent.expand_interpretation(group.id, max_expansions=1000000, max_candidates=256)
            assert not agent.interpretations.continuation_status(group.id).pending
            group = agent.interpretations.get(group.id)
        parents = [item for item in group.candidates if not learned or (
            item.payload.provenance == 'learned-speech-acts' and item.payload.metadata.get('speech_act_complete'))]
        assert parents, 'explicitly taught meaning must be present before grounding'
        candidate = propose_grounding(agent.interpretations, group.id, parents[0].id, [
            MentionBinding(("acts", 0, "frame", "roles", role), Ref(identity),
                           (evidence.id,), "authored binding for this test occurrence")
            for role, identity in roles.items()
        ])
        compared = agent.interpretations.get(group.id)
        return InterpretationDecision(candidate.id, "test supplies intended grounded reading", (evidence.id,),
            compared_revision=compared.revision,
            compared_candidate_ids=tuple(item.id for item in compared.candidates))
    agent.interpretation_selector = select
    return agent.turn(text)


def at(hour: int) -> datetime:
    return datetime(2026, 9, 18, hour, tzinfo=timezone.utc)


def ev(hour: int = 8) -> tc.Evidence:
    return tc.Evidence(tc.Ref("obs:x"), at(hour))


# ------------------------------------------------------------------ what can be said


def test_a_predication_keeps_all_its_participants_in_one_claim():
    """Three participants, one claim. This is the case binary claims could not hold."""
    store = tc.Store(tc.TypeRegistry())
    give = Proposition("give", {"subject": tc.Ref("person:ana"), "object": tc.Ref("thing:book"),
                                "recipient": tc.Ref("person:bo"), "time": tc.Ref("day:tuesday")})
    store.assert_(give, ev())
    [match] = store.find(Proposition("give", {"subject": tc.Ref("person:ana"), "recipient": Var("who")}))
    assert match.bindings["who"] == tc.Ref("person:bo")
    assert len(match.record.proposition.roles) == 4  # nothing was split off into a node


def test_a_proposition_can_fill_a_role_of_another():
    """Reported speech nests instead of flattening: the inner claim is not asserted."""
    store = tc.Store(tc.TypeRegistry())
    failed = Proposition("fail", {"subject": tc.Ref("thing:field")})
    store.assert_(Proposition("say", {"subject": tc.Ref("person:anem"), "content": failed}), ev())
    assert store.find(Proposition("fail", {"subject": tc.Ref("thing:field")})) == []
    [match] = store.find(Proposition("say", {"content": Proposition("fail", {"subject": Var("what")})}))
    assert match.bindings["what"] == tc.Ref("thing:field")


def test_polarity_and_modality_are_part_of_what_was_said():
    """A denial and an assertion are different claims, and neither answers for the other."""
    store = tc.Store(tc.TypeRegistry())
    store.assert_(Proposition("attend", {"subject": tc.Ref("person:ana")}, polarity=False), ev())
    store.assert_(Proposition("attend", {"subject": tc.Ref("person:bo")}, modality="desired"), ev())
    assert store.find(Proposition("attend", {"subject": Var("who")})) == []
    [denied] = store.find(Proposition("attend", {"subject": Var("who")}, polarity=False))
    assert denied.bindings["who"] == tc.Ref("person:ana")
    [wanted] = store.find(Proposition("attend", {"subject": Var("who")}, modality="desired"))
    assert wanted.bindings["who"] == tc.Ref("person:bo")


def test_how_sure_a_source_is_belongs_to_the_evidence_not_the_claim():
    """Two sources saying the same thing is one claim with two pieces of evidence.

    If confidence were part of the claim, the second source would either split it in two
    or be quietly dropped — and an unstated confidence would read as 1.0.
    """
    store = tc.Store(tc.TypeRegistry())
    said = Proposition("be", {"subject": tc.Ref("a:1"), "object": "x"})
    store.assert_(said, tc.Evidence(tc.Ref("person:ana"), at(8)))
    store.assert_(said, tc.Evidence(tc.Ref("person:bo"), at(9), confidence=tc.Score(0.6, "uncalibrated")))
    [record] = store.propositions()
    assert [e.confidence for e in record.evidence] == [None, tc.Score(0.6, "uncalibrated")]


def test_a_pattern_never_matches_a_fact_missing_the_role_it_asks_about():
    """The question is answered by what is stored, not by what is absent from it."""
    fact = Proposition("be", {"subject": tc.Ref("m:1"), "time": tc.Ref("day:tuesday")})
    assert matches(Proposition("be", {"subject": tc.Ref("m:1"), "location": Var("where")}), fact) is None
    assert matches(Proposition("be", {"subject": tc.Ref("m:1"), "time": Var("when")}), fact) is not None


# ------------------------------------------------------------------ retrieval


@pytest.mark.parametrize("reader", [None, "learned"], ids=["grammar", "learned"])
def test_an_answer_never_comes_from_a_side_the_question_supplied(reader):
    """The property behind "what is my name?" answering "name".

    Whatever the agent replies, it must not be something the question itself stated. This
    is checked across every told fact and every question about it, both readers.
    """
    told = ["my name is Jacob.", "I live in Austin.", "the meeting is on Tuesday."]
    asked = ["what is my name?", "where do I live?", "when is the meeting?",
             "where is the meeting?", "how many properties does the meeting have?",
             "what is my favourite colour?", "who is the meeting?"]
    agent = Agent([], reader=LearnedReader() if reader else None)
    for text in told:
        agent.turn(text)
    for question in asked:
        turn = agent.turn(question)
        answers = [outcome.answer for outcome in turn.outcomes if outcome.status == "answered"]
        for supplied in ("name", "meeting", "propert", "colour"):
            if supplied in question.lower():
                # An unresolved reading may quote the question in its explanation.
                # Quoted input is not a retrieved answer.
                assert all(supplied not in str(answer).lower() for answer in answers), (question, answers)


@pytest.mark.parametrize("reader", [None, "learned"], ids=["grammar", "learned"])
def test_what_it_was_told_comes_back_without_a_hop_through_an_invented_node(reader):
    from tensorcode.learning.speech_act import SpeechActLabel
    agent = Agent([], reader=LearnedReader() if reader else None)
    _grounded_turn(agent, "my name is Jacob.", {"subject": "fixture:name", "object": "fixture:Jacob"}, speech_label=SpeechActLabel('statement'))
    _grounded_turn(agent, "the meeting is red.", {"subject": "fixture:meeting"}, speech_label=SpeechActLabel('statement'))
    _grounded_turn(agent, "I live in Austin.", {"subject": "fixture:speaker", "location": "fixture:Austin"}, speech_label=SpeechActLabel('statement'))
    name_role = "subject" if reader else "object"  # Explicit expected reader structure.
    assert "Jacob" in _grounded_turn(agent, "what is my name?", {name_role: "fixture:name"},
        speech_label=SpeechActLabel('question', 'object', (), (0, 1, 2, 3, 4))).reply
    assert "Austin" in _grounded_turn(agent, "where do I live?", {"subject": "fixture:speaker"},
        speech_label=SpeechActLabel('question', 'location', ('roles', 'manner'), (0,))).reply
    stored = [r.proposition for r in agent.store.propositions()]
    assert stored and not any(str(f).startswith("event:") for p in stored for f in p.roles.values())


@pytest.mark.parametrize("reader", [None, "learned"], ids=["grammar", "learned"])
def test_a_question_about_a_property_that_was_never_stated_is_unanswered(reader):
    """The failure the event hop caused: a "how many" answered with whatever was around."""
    agent = Agent([], reader=LearnedReader() if reader else None)
    agent.turn("the meeting is red.")
    for question in ("how many properties does the meeting have?", "where is the meeting?",
                     "why is the meeting?"):
        turn = agent.turn(question)
        assert not any(outcome.status in ("answered", "done") for outcome in turn.outcomes), question
        assert "don't know" in turn.reply or "didn't fully follow" in turn.reply, question


# ------------------------------------------------------------------ persistence


def test_competing_claims_both_survive_a_round_trip():
    """Two sources disagreeing is a state the store must be able to hold and reload.

    Nothing resolves the disagreement on the way in or on the way out: both propositions
    come back, each with the evidence that brought it.
    """
    reg = tc.TypeRegistry()
    store = tc.Store(reg)
    here = Proposition("has_location", {"undergoer": tc.Ref("thing:key"), "goal": tc.Ref("room:a")})
    there = Proposition("has_location", {"undergoer": tc.Ref("thing:key"), "goal": tc.Ref("room:b")})
    store.assert_(here, tc.Evidence(tc.Ref("person:ana"), at(8), confidence=tc.Score(0.6, "uncalibrated")))
    store.assert_(there, tc.Evidence(tc.Ref("person:bo"), at(9)))

    restored, report = tc.Store.from_json(json.loads(json.dumps(store.to_json())), reg)
    assert report.lossless
    found = restored.find(Proposition("has_location", {"undergoer": tc.Ref("thing:key"), "goal": Var("where")}))
    assert {m.bindings["where"] for m in found} == {tc.Ref("room:a"), tc.Ref("room:b")}
    assert {r.proposition for r in restored.propositions()} == {here, there}
    assert [e.source for r in restored.propositions() for e in r.evidence] == [tc.Ref("person:ana"), tc.Ref("person:bo")] \
        or [e.source for r in restored.propositions() for e in r.evidence] == [tc.Ref("person:bo"), tc.Ref("person:ana")]


def test_a_nested_proposition_survives_a_round_trip():
    reg = tc.TypeRegistry()
    store = tc.Store(reg)
    inner = Proposition("do", {"subject": tc.Ref("person:lara"), "object": tc.Ref("exam:17")})
    store.assert_(Proposition("believe", {"subject": tc.Ref("person:casey"), "content": inner}), ev())
    restored, report = tc.Store.from_json(json.loads(json.dumps(store.to_json())), reg)
    assert report.lossless
    [match] = restored.find(Proposition("believe", {"content": Proposition("do", {"subject": Var("who")})}))
    assert match.bindings["who"] == tc.Ref("person:lara")


def test_a_superseded_claim_stops_answering_but_is_not_erased():
    store = tc.Store(tc.TypeRegistry())
    old = Proposition("has_location", {"undergoer": tc.Ref("thing:key"), "goal": tc.Ref("room:a")})
    store.assert_(old, ev(8))
    assert store.supersede(Proposition("has_location", {"undergoer": tc.Ref("thing:key")}), "it moved") == [old.id]
    store.assert_(Proposition("has_location", {"undergoer": tc.Ref("thing:key"), "goal": tc.Ref("room:b")}), ev(9))
    [match] = store.find(Proposition("has_location", {"undergoer": tc.Ref("thing:key"), "goal": Var("where")}))
    assert match.bindings["where"] == tc.Ref("room:b")
    assert store._props[old.id].retracted is not None  # still there, with why and when


# ------------------------------------------------------------------ the converter


def test_the_converter_reports_what_it_could_not_carry():
    """A converter that silently drops what it cannot represent looks like a full reading."""
    from tensorcode.language.semantics import to_propositions
    from tensorcode.language import Frame

    class Odd:
        def __repr__(self) -> str:
            return "<odd>"

    frame = Frame("be", {"subject": Odd()}, {"mood": "declarative"})
    got, dropped = to_propositions(frame, source=tc.Ref("agent:user"))
    assert got and dropped == ["role filler of type Odd"]
