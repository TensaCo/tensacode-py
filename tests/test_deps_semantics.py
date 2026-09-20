"""Dependency trees to meanings, with the trees written out by hand (no model needed)."""

from __future__ import annotations

from tensorcode.language.deps_semantics import ProvisionalMeaning, Reader
from tensorcode.language.semantics import Entity, Frame, Question, Request

# "in" is a place and "to" a goal in STREUSLE's counts; the tests state what they need so
# they do not depend on that data being installed.
# Empty marker is an authored role for the no-case oblique in the wh fixture.
PREPOSITIONS = {"": [("location", 0.0)], "in": [("location", -0.1)], "on": [("location", -0.1)], "to": [("destination", -0.2)],
                "from": [("source", -0.1)], "with": [("instrument", -0.1)]}


def read(words, tags, lemmas, heads, labels):
    return Reader(PREPOSITIONS).read(words, tags, lemmas, {i + 1: h for i, h in enumerate(heads)},
                                     {i + 1: dep for i, dep in enumerate(labels)})


def test_subjectless_clause_is_provisional_not_an_authorized_request():
    got = read(["make", "a", "folder"], ["VERB", "DET", "NOUN"], ["make", "a", "folder"], [0, 3, 1],
               ["root", "det", "obj"])
    assert isinstance(got[0], ProvisionalMeaning)
    assert got[0].frame.predicate == "make"
    assert got[0].frame.roles["object"].features["noun"] == "folder"


def test_a_prepositional_phrase_takes_the_role_its_preposition_marks():
    got = read(["put", "it", "in", "documents"], ["VERB", "PRON", "ADP", "NOUN"], ["put", "it", "in", "document"],
               [0, 1, 4, 1], ["root", "obj", "case", "obl"])
    assert isinstance(got[0], ProvisionalMeaning)
    assert "location" in got[0].frame.roles


def test_wh_input_preserves_role_content_without_guessing_queried_slot():
    got = read(["where", "is", "the", "meeting"], ["PRON", "AUX", "DET", "NOUN"], ["where", "be", "the", "meeting"],
               [4, 4, 4, 0], ["obl", "cop", "det", "root"])
    assert isinstance(got[0], ProvisionalMeaning)
    assert got[0].frame.roles["location"].text == "where"
    assert "mood" not in got[0].frame.features


def test_subject_present_does_not_authorize_a_statement():
    got = read(["I", "live", "in", "austin"], ["PRON", "VERB", "ADP", "PROPN"], ["i", "live", "in", "austin"],
               [2, 0, 4, 2], ["nsubj", "root", "case", "obl"])
    assert isinstance(got[0], ProvisionalMeaning)
    assert "mood" not in got[0].frame.features
    assert got[0].frame.roles["subject"].kind == "pronoun"
    assert got[0].frame.roles["location"].text.endswith("austin")


def test_negation_and_modality_reach_the_frame():
    got = read(["you", "can", "not", "design", "it"], ["PRON", "AUX", "PART", "VERB", "PRON"],
               ["you", "can", "not", "design", "it"], [4, 4, 4, 0, 4], ["nsubj", "aux", "advmod", "root", "obj"])
    assert got[0].frame.features.get("modality") == "can"
    assert got[0].frame.features.get("polarity") == "negative"


def test_removed_speech_act_shortcuts_do_not_survive_under_reader_api():
    assert not hasattr(Reader, 'speech_act')
    for words, tags, lemmas, heads, labels in (
        (['Running'], ['VERB'], ['run'], [0], ['root']),
        (['Run', '?'], ['VERB', 'PUNCT'], ['run', '?'], [0, 1], ['root', 'punct']),
        (['Whoever', 'runs'], ['PRON', 'VERB'], ['whoever', 'run'], [2, 0], ['nsubj', 'root']),
    ):
        meaning, = read(words, tags, lemmas, heads, labels)
        assert isinstance(meaning, ProvisionalMeaning)
        assert not isinstance(meaning, (Frame, Request, Question))
        assert 'mood' not in meaning.frame.features
        if words[0] == 'Whoever':
            assert meaning.frame.roles['subject'].text == 'Whoever'


def test_provisional_meaning_retains_detached_syntax_and_tree_anchor():
    words, tags, lemmas = ['make', 'folder'], ['VERB', 'NOUN'], ['make', 'folder']
    heads, labels = [0, 1], ['root', 'obj']
    meaning, = read(words, tags, lemmas, heads, labels)
    words.clear()
    tags.clear()
    lemmas.clear()
    assert meaning.words == ('make', 'folder')
    assert meaning.tags == ('VERB', 'NOUN')
    assert meaning.lemmas == ('make', 'folder')
    assert meaning.heads == ((1, 0), (2, 1))
    assert meaning.labels == ((1, 'root'), (2, 'obj'))
    assert meaning.root == 1 and meaning.frame_index == 0


def test_entity_fragment_does_not_gain_a_communicative_act():
    meaning, = read(['folder'], ['NOUN'], ['folder'], [0], ['root'])
    assert isinstance(meaning, Entity)
    assert not isinstance(meaning, (Request, Question, Frame))
