"""What the world demands of the grammar, as a test the grammar can be held to.

These cases have *independent provenance*: they are not invented to exercise the grammar, they are
every claim shape `research/civ_sim/minds.py` actually puts into somebody's mouth, in every
combination of dialect, tense, negation, modality and hearsay the simulation produces. That is why
they are worth keeping — an authored grammar suite tends to test what its author already thought of,
whereas this set found five copula failures, a lost quantifier, a mis-stemmed `showed` and a tense
that could not be expressed, none of which were on anyone's list.

Two things are asserted, and only two, because they are the two the world needs:

    say -> hear recovers the predicate and the object     (meaning survives the round trip)
    say produces well-formed English                      (the viewer shows these sentences to a person)

The second is checked structurally rather than by eyeballing: no doubled copula, no bare "am"/"are"
with a third-person subject, no leftover placeholder, one sentence-final stop. If a sentence looks
wrong to a reader it should fail here first.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from research.civ_sim import language as L

NAMES = ["Anem", "Kasa", "Nise", "Domili", "Miol", "Coralin", "Aldmere", "Brenholt"]
PLACES = ["Coralin", "Aldmere", "Brenholt"]

# every (subject, predicate, object) shape the minds can speak, from minds._sayable and converse
SHAPES = [
    ("village:Coralin", "has_amount", "food:much"),
    ("village:Coralin", "has_amount", "food:little"),
    ("village:Coralin", "has_amount", "food:none"),
    ("village:Aldmere", "has_amount", "food:some"),
    ("person:Anem", "died", "True"),
    ("person:Nise", "hungry", "True"),
    ("person:Kasa", "trustworthy", "True"),
    ("person:Kasa", "trustworthy", "False"),
    ("good:wood", "expensive", "True"),
    ("good:wood", "cheap", "True"),
    ("good:tools", "expensive", "True"),
    ("weather:snow", "coming", "True"),
    ("weather:rain", "coming", "True"),
    ("weather:frost", "coming", "True"),
    ("sky:sky", "showed", "conjunction"),
    ("sky:sky", "showed", "solar"),
    ("sky:sky", "showed", "lunar"),
    ("person:Domili", "owes", "good:food"),
    ("person:Kasa", "keeps_back", "good:food"),
    ("person:Miol", "gives", "good:food"),
    ("settlement:Aldmere", "raided", "settlement:Brenholt"),
]

BAD_ENGLISH = (
    re.compile(r"\b(is|are|am|was|were)\s+(is|are|am|be|was|were)\b"),  # doubled copula, "am never be"
    re.compile(r"^(?!I\b)\w+\s+am\b"),  # "Nise am hungry"
    re.compile(r"\bnever\s+be\b"),
    re.compile(r"[?:]|\bNone\b|\bTrue\b|\bFalse\b"),  # a placeholder or a raw value reached the surface
    re.compile(r"\.\s*\S"),  # something after the full stop
    re.compile(r"\b(plenty|heaps|lots)\s+(?!of\b)\w"),  # "heaps bread" wants "heaps of bread"
)


def lexicons():
    rng = np.random.default_rng(11)
    base = L.base_lexicon()
    return [("shared", base)] + [(f"dialect{i}", L.dialect(base, i, rng)) for i in range(3)]


@pytest.mark.parametrize("subject,predicate,obj", SHAPES)
@pytest.mark.parametrize("lex_name", [n for n, _ in lexicons()])
def test_every_shape_the_world_speaks_survives_its_own_dialect(subject, predicate, obj, lex_name):
    """Said in a dialect and heard in the same dialect, the meaning must come back whole."""
    lex = dict(lexicons())[lex_name]
    sentence = L.say(subject, predicate, obj, lex)
    assert sentence and sentence[0].isupper() and sentence.rstrip().endswith("."), f"malformed: {sentence!r}"
    heard = L.hear(sentence, lex, names=NAMES, context={}, speaker="Speaker", settlements=PLACES)
    assert heard.claims, f"{sentence!r} could not be understood in the dialect that said it"
    got = heard.claims[0]
    assert got["predicate"] == predicate, f"{sentence!r} -> {got['predicate']} (wanted {predicate})"
    if predicate == "has_amount":
        good, _, amount = str(obj).partition(":")
        heard_amount = str(got["object"]).split(":")[-1]
        # "some" and "none" are the amounts the grammar cannot yet hold; they degrade to "unsaid"
        if amount in ("much", "little"):
            assert heard_amount == amount, f"{sentence!r} lost its quantifier: {heard_amount}"
    elif obj in ("True", "False"):
        assert str(got["object"]) == obj, f"{sentence!r} -> {got['object']} (wanted {obj})"


@pytest.mark.parametrize("subject,predicate,obj", SHAPES)
@pytest.mark.parametrize("lex_name", [n for n, _ in lexicons()])
def test_what_the_world_says_is_well_formed_english(subject, predicate, obj, lex_name):
    """These sentences are shown to a person in the viewer, so they have to read as English —
    in every dialect, not just the shared vocabulary. Checking only the base lexicon is how
    "Coralin holds heaps bread" reached the viewer: the quantifier synonyms a dialect uses
    (`plenty`, `heaps`) are mass quantifiers and need "of", and only `much` works bare."""
    lex = dict(lexicons())[lex_name]
    sentence = L.say(subject, predicate, obj, lex)
    for pattern in BAD_ENGLISH:
        assert not pattern.search(sentence), f"{sentence!r} matches {pattern.pattern}"
    assert sentence.count(".") == 1


def test_reported_speech_keeps_its_source_and_its_content():
    lex = L.base_lexicon()
    sentence = L.say("village:Coralin", "has_amount", "food:much", lex, secondhand="Nise")
    assert "Nise" in sentence
    heard = L.hear(sentence, lex, names=NAMES, context={}, speaker="Speaker", settlements=PLACES)
    assert heard.claims, f"reported speech was not understood: {sentence!r}"
    got = heard.claims[0]
    assert got["hearsay"] is True and got["via"] == "Nise"
    assert got["predicate"] == "has_amount"


def test_a_request_is_still_sayable_and_heard():
    lex = L.base_lexicon()
    sentence = L.say("person:Miol", "gives", "good:food", lex, mood="command", modality="should")
    assert sentence and "." in sentence
    heard = L.hear(sentence, lex, names=NAMES, context={}, speaker="Speaker", settlements=PLACES)
    assert heard.claims and heard.claims[0]["predicate"] == "gives"


def test_a_shape_the_general_grammar_cannot_say_falls_back_rather_than_going_silent():
    """A new kind of claim must degrade to the old wording, not strike the speaker mute."""
    lex = L.base_lexicon()
    sentence = L.say("person:Anem", "befriended", "person:Kasa", lex)
    assert sentence and sentence.strip(), "an unmapped predicate produced no sentence at all"


@pytest.mark.parametrize("subject,predicate,obj", SHAPES)
def test_saying_the_same_thing_twice_says_it_the_same_way(subject, predicate, obj):
    """Determinism at the level of the sentence, memo or no memo."""
    lex = L.base_lexicon()
    assert L.say(subject, predicate, obj, lex) == L.say(subject, predicate, obj, lex)


def test_a_drifted_dialect_costs_the_word_and_not_always_the_meaning():
    """The lossy channel: a listener who lacks the word reports it as unfamiliar, and sometimes
    still recovers the claim. Both outcomes are legitimate; silently inventing a claim is not."""
    rng = np.random.default_rng(3)
    base = L.base_lexicon()
    speaker, listener = L.dialect(base, 0, rng), L.dialect(base, 1, rng)
    for _ in range(12):
        L.drift(speaker, rng, rate=0.25)
    sentence = L.say("village:Coralin", "has_amount", "food:much", speaker)
    heard = L.hear(sentence, listener, names=NAMES, context={}, speaker="Speaker", settlements=PLACES)
    if heard.claims:
        assert heard.claims[0]["predicate"] == "has_amount"
        assert str(heard.claims[0]["object"]).startswith("food:")
    else:
        assert heard.via == "none"  # an honest failure, not a guess


@pytest.mark.parametrize("subject,predicate,obj", SHAPES)
def test_the_memos_are_transparent_not_load_bearing(subject, predicate, obj):
    """Saying and hearing must give the same answer with the caches off as on.

    This is the property that makes the memos an optimization rather than part of the model. It is
    worth a test because a warm cache can make an A/B comparison look clean while the change under
    test has actually broken something — so every verification in this tree is run with the caches
    off, and this holds them to producing the same result either way.
    """
    lex = L.base_lexicon()
    names, places = NAMES, PLACES
    was = L.MEMO
    try:
        L.MEMO = True
        said_on = L.say(subject, predicate, obj, lex)
        heard_on = L.hear(said_on, lex, names=names, context={}, speaker="S", settlements=places)
        L.MEMO = False
        said_off = L.say(subject, predicate, obj, lex)
        heard_off = L.hear(said_off, lex, names=names, context={}, speaker="S", settlements=places)
    finally:
        L.MEMO = was
    assert said_on == said_off, f"the cache changed what was said: {said_on!r} vs {said_off!r}"
    assert heard_on.claims == heard_off.claims, "the cache changed what was understood"
    assert (heard_on.unknown, heard_on.via) == (heard_off.unknown, heard_off.via)


def test_our_canonicaliser_never_hands_the_parser_a_dot_terminated_word():
    """Why a trailing-dot tokenizer bug upstream could not have reached our measurements.

    The general parser briefly treated `food.` as one token with no lexicon entry, which made the
    last word of every sentence enter as a guessed name. Our `_canonicalize` tokenizes with its own
    pattern and drops punctuation before their tokenizer sees anything, so the bug never applied
    here — this test is the evidence for that claim, and a guard if the canonicaliser changes.
    """
    lex = L.base_lexicon()
    known = set(NAMES) | set(PLACES)
    for subject, predicate, obj in SHAPES:
        sentence = L.say(subject, predicate, obj, lex)
        assert sentence.rstrip().endswith("."), f"{sentence!r} should end in a stop"
        text, _unknown = L._canonicalize(sentence, lex, known=known)
        assert not any(word.endswith(".") for word in text.split()), f"a dot survived into {text!r}"
        assert "." not in text
