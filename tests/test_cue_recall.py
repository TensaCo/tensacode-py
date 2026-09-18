"""Cue recall by structure: roles, morphology, and the mind's own links."""

from datetime import datetime, timezone

import tensacode as tc
from tensacode.cues import Cues, content, exact_find, lemma

T0 = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
USER = tc.Ref("person:user")


def store(*claims):
    mind = tc.Store()
    for subject, predicate, obj in claims:
        mind.tell(tc.Claim(subject, predicate, obj), tc.Evidence(tc.Ref("utterance:1"), T0, method="told"))
    return mind


def test_lemmas_make_an_inflected_cue_the_same_cue():
    assert lemma("notes") == "note" and lemma("bikes") == "bike" and lemma("reading") == "read"
    assert "deadline" in content("my deadlines")


def test_stop_words_are_dropped_but_compounds_are_kept():
    assert content("what is my time zone") >= {"time", "zone", "timezone"}


def test_the_predicate_is_the_strongest_role():
    mind = store((USER, "deadline", "the 30th"), (tc.Ref("ui:col#1"), "label", "Date Modified"))
    (top, *_) = Cues(mind).find("what is my deadline", k=3)
    assert top.record.claim.predicate == "deadline"


def test_a_question_about_me_prefers_what_i_said_over_what_is_on_screen():
    """"my time zone" is a question about the asker; a column header reading "Time" is not an answer."""
    mind = store((USER, "timezone", "UTC+1"), (tc.Ref("ui:col#3"), "label", "Time"))
    (top, *_) = Cues(mind).find("my time zone", k=3)
    assert top.record.claim.subject == USER


def test_a_paraphrase_with_no_shared_word_still_finds_it_through_the_object():
    mind = store((USER, "cat", "Mackerel"))
    hits = Cues(mind).find("who is Mackerel", k=3)
    assert hits and hits[0].record.claim.predicate == "cat"


def test_the_stores_own_links_bridge_words_it_has_been_told_are_the_same():
    mind = store((USER, "bike", "a Brompton"))
    assert not Cues(mind).find("the make of my bicycle", k=3)
    mind.tell(tc.Claim(tc.Ref("word:bicycle"), "same_as", tc.Ref("word:bike")),
              tc.Evidence(tc.Ref("utterance:2"), T0, method="told"))
    hits = Cues(mind).find("the make of my bicycle", k=3)
    assert hits and hits[0].record.claim.predicate == "bike"
    assert "bicycle" in hits[0].why()


def test_a_direct_hit_outranks_one_reached_through_a_link():
    mind = store((USER, "bike", "a Brompton"), (USER, "bicycle", "a Moulton"))
    mind.tell(tc.Claim(tc.Ref("word:bicycle"), "same_as", tc.Ref("word:bike")),
              tc.Evidence(tc.Ref("utterance:2"), T0, method="told"))
    (top, *_) = Cues(mind).find("what is my bicycle", k=3)
    assert top.record.claim.predicate == "bicycle"


def test_the_exact_baseline_needs_the_predicate_verbatim():
    mind = store((USER, "deadline", "the 30th"))
    assert exact_find(mind, "what is my deadline")
    assert not exact_find(mind, "my deadlines")


def test_a_cue_of_nothing_but_stop_words_recalls_nothing():
    mind = store((USER, "deadline", "the 30th"))
    assert Cues(mind).find("what about it") == []


def test_skipped_predicates_stay_out_of_the_way():
    """The assistant stores the raw utterance too; recall should not answer with it."""
    mind = store((USER, "deadline", "the 30th"), (tc.Ref("request:1"), "words", "my deadline is the 30th"))
    hits = Cues(mind, skip_predicates=("words",)).find("what is my deadline", k=5)
    assert all(h.record.claim.predicate != "words" for h in hits)
