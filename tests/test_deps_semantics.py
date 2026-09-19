"""Dependency trees to meanings, with the trees written out by hand (no model needed)."""

from __future__ import annotations

from tensorcode.language.deps_semantics import Reader
from tensorcode.language.semantics import Entity, Frame, Question, Request

# "in" is a place and "to" a goal in STREUSLE's counts; the tests state what they need so
# they do not depend on that data being installed.
# Empty marker is an authored role for the no-case oblique in the wh fixture.
PREPOSITIONS = {"": [("location", 0.0)], "in": [("location", -0.1)], "on": [("location", -0.1)], "to": [("destination", -0.2)],
                "from": [("source", -0.1)], "with": [("instrument", -0.1)]}


def read(words, tags, lemmas, heads, labels):
    return Reader(PREPOSITIONS).read(words, tags, lemmas, {i + 1: h for i, h in enumerate(heads)},
                                     {i + 1: dep for i, dep in enumerate(labels)})


def test_an_imperative_with_no_subject_is_a_request():
    got = read(["make", "a", "folder"], ["VERB", "DET", "NOUN"], ["make", "a", "folder"], [0, 3, 1],
               ["root", "det", "obj"])
    assert isinstance(got[0], Request)
    assert got[0].frame.predicate == "make"
    assert got[0].frame.roles["object"].features["noun"] == "folder"


def test_a_prepositional_phrase_takes_the_role_its_preposition_marks():
    got = read(["put", "it", "in", "documents"], ["VERB", "PRON", "ADP", "NOUN"], ["put", "it", "in", "document"],
               [0, 1, 4, 1], ["root", "obj", "case", "obl"])
    assert isinstance(got[0], Request)
    assert "location" in got[0].frame.roles


def test_a_wh_question_asks_about_the_role_its_word_names():
    got = read(["where", "is", "the", "meeting"], ["PRON", "AUX", "DET", "NOUN"], ["where", "be", "the", "meeting"],
               [4, 4, 4, 0], ["obl", "cop", "det", "root"])
    assert isinstance(got[0], (Question, Entity, Frame))
    if isinstance(got[0], Question):
        assert got[0].asked == "location"


def test_a_statement_with_a_subject_is_a_statement():
    got = read(["I", "live", "in", "austin"], ["PRON", "VERB", "ADP", "PROPN"], ["i", "live", "in", "austin"],
               [2, 0, 4, 2], ["nsubj", "root", "case", "obl"])
    assert isinstance(got[0], Frame) and got[0].mood == "declarative"
    assert got[0].roles["subject"].kind == "pronoun"
    assert got[0].roles["location"].text.endswith("austin")


def test_negation_and_modality_reach_the_frame():
    got = read(["you", "can", "not", "design", "it"], ["PRON", "AUX", "PART", "VERB", "PRON"],
               ["you", "can", "not", "design", "it"], [4, 4, 4, 0, 4], ["nsubj", "aux", "advmod", "root", "obj"])
    assert got[0].features.get("modality") == "can"
    assert got[0].features.get("polarity") == "negative"
