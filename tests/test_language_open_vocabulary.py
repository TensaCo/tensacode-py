"""Reading sentences whose content words the lexicon has never seen.

This is the realistic case for the simulation: villagers talk about grain, fields,
wells and each other's names, and no fixed lexicon holds those words. The grammar
below is bare ``ENGLISH`` — every content word in every test here is unknown — so a
failure shows up as a vanished clause rather than as a wrong slot.

The bug these tests exist for: unknown words could only be *names*, so a clause with
an unknown verb had no verb, and "Anem said the north field failed" lost its entire
embedded clause (coverage 0.33) while "the north field did not fail" was read as a
polarity question about a thing called "fail".
"""

import pytest

import tensorcode as tc
from tensorcode.language import ENGLISH, Frame, Question, Request, realize, to_claims, understand


def read(text, grammar=ENGLISH):
    got = understand(grammar, text)
    assert got.best is not None, text
    return got.meanings[0], got


# --------------------------------------------------------- the fallback itself


def test_an_unknown_word_takes_a_category_from_its_morphology():
    entries = {(e.cat, e.features.get("tense"), e.features.get("number"), e.sem)
               for e in ENGLISH.entries_for("failed")}
    assert ("V", "past", None, "fail") in entries, entries      # -ed marks a past verb
    assert any(cat == "N" for cat, *_ in entries)               # and a noun reading survives
    plural = {(e.cat, e.features.get("number"), e.sem) for e in ENGLISH.entries_for("fields")}
    assert ("N", "plural", "field") in plural


def test_a_guessed_word_is_marked_and_lowers_confidence():
    frame, got = read("the north field failed")
    assert got.coverage == 1.0
    guessed = dict(got.guessed)
    assert guessed.get("failed") == "V" and "field" in guessed
    assert got.confidence.kind == "uncalibrated" and 0.0 < got.confidence.value < 1.0
    # a sentence of known words is not marked at all
    known = understand(ENGLISH, "who am i")
    assert known.guessed == () and known.confidence.value == 1.0
    _ = frame


def test_a_known_word_keeps_its_own_reading_rather_than_being_guessed_at():
    assert all(not e.features.get("guessed") for e in ENGLISH.entries_for("not"))
    assert all(not e.features.get("guessed") for e in ENGLISH.entries_for("the"))


def test_skipping_still_happens_for_genuinely_unattachable_material():
    got = understand(ENGLISH, "the north field failed !! ?? ,,")
    assert got.skipped, "punctuation salad should be skipped, not guessed at"
    assert got.meanings and isinstance(got.meanings[0], Frame)


# ------------------------------------------------------------------- negation


def test_negation_with_an_out_of_lexicon_verb_is_a_negated_claim_not_a_question():
    frame, got = read("the north field did not fail")
    assert isinstance(frame, Frame), frame
    assert frame.predicate == "fail" and frame.negated
    assert frame.feature("tense") == "past"
    assert frame.mood == "declarative"
    assert got.coverage == 1.0
    assert not isinstance(frame, Question)


def test_a_negated_out_of_lexicon_claim_is_reified_not_flattened():
    frame, _ = read("the grain did not arrive")
    pairs = to_claims(frame, source=tc.Ref("obs:overheard"))
    assert not isinstance(pairs, tc.Unknown)
    predicates = {c.predicate for c, _ in pairs}
    assert "polarity" in predicates and "is_a" in predicates
    # the bare proposition is never asserted
    assert not any(c.predicate == "arrive" for c, _ in pairs)


@pytest.mark.parametrize("text", [
    "the north field did not fail",
    "the grain did not arrive",
    "the well did not flood",
])
def test_negated_clauses_are_never_read_as_polarity_questions(text):
    meaning, _ = read(text)
    assert not isinstance(meaning, Question), f"{text!r} read as a question: {meaning}"


# --------------------------------------------------------- reported speech


@pytest.mark.parametrize("text,inner,negated", [
    ("Anem said the north field failed", "fail", False),
    ("Anem said that the north field failed", "fail", False),
    ("Anem said the grain did not arrive", "arrive", True),
    ("Bera told me the grain arrived", "arrive", False),
    ("Anem told Bera the well flooded", "flood", False),
])
def test_a_sentential_complement_survives_out_of_lexicon_content_words(text, inner, negated):
    frame, got = read(text)
    assert got.coverage == 1.0, f"{text!r} lost words: {got.skipped}"
    assert isinstance(frame, Frame) and frame.predicate in ("say", "tell")
    content = frame.role("content")
    assert isinstance(content, Frame), f"no embedded clause in {frame.describe()}"
    assert content.predicate == inner
    assert content.negated is negated


def test_reported_out_of_lexicon_speech_stays_in_the_speakers_scope():
    frame, _ = read("Anem said the north field failed")
    pairs = to_claims(frame, source=tc.Ref("obs:overheard"))
    assert not isinstance(pairs, tc.Unknown)
    scoped = [(c, e) for c, e in pairs if c.scope is not None]
    assert scoped, "the reported clause must not enter the shared world"
    assert any(e.source.id.endswith("Anem") for _, e in scoped)


def test_the_hearer_of_a_report_is_kept_apart_from_its_content():
    frame, _ = read("Bera told me the grain arrived")
    assert frame.role("recipient") is not None
    assert isinstance(frame.role("content"), Frame)


# ----------------------------------------------------- other constructions


@pytest.mark.parametrize("text,predicate,feature,value", [
    ("the grain has arrived", "arrive", "aspect", "perfect"),
    ("the well might flood", "flood", "modality", "may"),
    ("everyone must share grain", "share", "modality", "must"),
])
def test_tense_aspect_and_modality_survive_unknown_verbs(text, predicate, feature, value):
    frame, _ = read(text)
    assert frame.predicate == predicate and frame.feature(feature) == value


def test_a_noun_can_modify_a_noun_without_either_being_known():
    frame, _ = read("the grain store is empty")
    subject = frame.role("subject")
    assert subject is not None and subject.text == "grain store"


def test_an_unknown_verb_can_be_said_again():
    """The suffix table that lets "failed" in also gets it back out."""
    frame, _ = read("the north field failed")
    said = realize(ENGLISH, frame)
    assert said == "the north field failed"
    back, _ = read(said)
    assert repr(back) == repr(frame)
