"""What the English grammar must read, and what it must refuse to assert."""

import pytest

import tensacode as tc
from tensacode.language.semantics import Entity
from tensacode.language import ENGLISH, Context, Frame, Question, Request, realize, resolve, to_claims, understand, unresolved, words

VILLAGE = ENGLISH.extend(entries=[
    *words("field", cat="N", sem="field"),
    *words("grain", cat="N", sem="grain"),
    *words("harvest", cat="N", sem="harvest"),
    *words("fail", cat="V", sem="fail"),
    *words("arrive", cat="V", sem="arrive"),
    *words("share", cat="V", sem="share"),
    *words("north", "south", cat="Adj"),
    *words("good", cat="Adj"),
    *words("better", cat="Adj", sem="good", degree="comparative"),
])


def read(text):
    got = understand(VILLAGE, text)
    assert got.best is not None, text
    return got.meanings[0], got


def test_a_declarative_is_a_frame_with_its_tense():
    frame, got = read("the north field failed")
    assert isinstance(frame, Frame) and frame.predicate == "fail"
    assert frame.feature("tense") == "past" and frame.mood == "declarative"
    assert frame.role("subject").features["noun"] == "field"
    assert got.coverage == 1.0


def test_negation_is_a_feature_not_a_lost_word():
    frame, _ = read("the grain did not arrive")
    assert frame.predicate == "arrive" and frame.negated and frame.feature("tense") == "past"


def test_modality_and_quantifiers_are_read():
    frame, _ = read("everyone must share grain")
    assert frame.feature("modality") == "must"
    assert frame.role("subject").features["quantifier"] == "all"


def test_a_comparative_keeps_its_standard():
    frame, _ = read("the north field is better than the south field")
    assert frame.predicate == "good" and frame.feature("degree") == "comparative"
    standards = [f.role("standard") for f in frame.walk() if f.role("standard") is not None]
    entities = [e for e in frame.entities()]
    assert standards or any(e.features.get("standard") for e in entities)  # verb- or noun-attached


def test_reported_speech_nests_rather_than_splitting_into_two_assertions():
    frame, _ = read("Anem said the north field failed")
    assert frame.predicate == "say"
    inner = frame.role("content")
    assert isinstance(inner, Frame) and inner.predicate == "fail"


def test_reported_speech_lands_in_its_own_scope_sourced_to_the_speaker():
    frame, _ = read("Anem said the north field failed")
    pairs = to_claims(frame, source=tc.Ref("obs:overheard"))
    assert not isinstance(pairs, tc.Unknown)
    scopes = {claim.scope for claim, _ in pairs if claim.predicate == "is_a" or claim.predicate == "fail"}
    inner = [(c, e) for c, e in pairs if c.scope is not None]
    assert inner, "the reported content must be scoped, not asserted in the shared world"
    assert any(e.source.id.startswith("entity:Anem") for _, e in inner)
    assert any(c.predicate == "say" and c.scope is None for c, _ in pairs)
    _ = scopes


def test_an_imperative_is_a_request_and_asserts_nothing():
    got = understand(VILLAGE, "share grain")
    request = next(m for m in got.meanings if isinstance(m, Request))
    assert request.act == "share"
    assert isinstance(to_claims(request.frame, source=tc.Ref("obs:chat")), tc.Unknown)


def test_a_question_is_a_question_and_asserts_nothing():
    got = understand(VILLAGE, "did the north field fail")
    question = next(m for m in got.meanings if isinstance(m, Question))
    assert question.asked == "polarity" and question.frame.predicate == "fail"
    assert isinstance(to_claims(question.frame, source=tc.Ref("obs:chat")), tc.Unknown)


def test_a_negated_claim_is_reified_rather_than_flattened():
    frame, _ = read("the grain did not arrive")
    pairs = to_claims(frame, source=tc.Ref("obs:x"))
    predicates = {c.predicate for c, _ in pairs}
    assert "polarity" in predicates and "is_a" in predicates
    # crucially: no bare triple that would state the opposite of what was said
    assert not any(c.predicate == "arrive" for c, _ in pairs)


def test_a_simple_claim_stays_a_triple():
    got = understand(VILLAGE.extend(entries=[*words("own", cat="V", sem="owns")]), "Anem owns the north field")
    pairs = to_claims(got.meanings[0], source=tc.Ref("obs:x"))
    assert len(pairs) == 1 and pairs[0][0].predicate == "owns"


def test_pronouns_resolve_against_the_conversation():
    first, _ = read("the north field failed")
    context = Context()
    context.observe(first)
    second, _ = read("it failed")
    resolved = resolve(second, context)
    assert not unresolved(resolved)
    assert resolved.role("subject").features.get("via") == "it"


def test_two_equally_good_antecedents_stay_unresolved():
    context = Context()
    for text in ("the north field failed", "the south field failed"):
        frame, _ = read(text)
        context.observe(frame)
    frame, _ = read("it failed")
    resolved = resolve(frame, context)
    left = unresolved(resolved)
    assert left and len(left[0].candidates) >= 2
    assert isinstance(to_claims(resolved, source=tc.Ref("obs:x")), tc.Unknown)


@pytest.mark.parametrize("text", [
    "the north field failed",
    "the grain did not arrive",
    "everyone must share grain",
    "Anem said the north field failed",
])
def test_saying_a_meaning_and_reading_it_back_gives_the_same_meaning(text):
    frame, _ = read(text)
    said = realize(VILLAGE, frame)
    assert said is not None, f"cannot say {frame.describe()}"
    back = understand(VILLAGE, said)
    assert back.meanings, said
    assert repr(back.meanings[0]) == repr(frame), f"{text!r} -> {said!r} -> {back.describe()}"


# --------------------------------------------- saying it, in the coordinator's cases


@pytest.mark.parametrize("text,said", [
    ("snow came", "snow came"),                       # was "snow cames"
    ("the elder died", "the elder died"),             # was "the elder di"
    ("Miol owes Anem grain", "Miol owes Anem grain"),  # was "Miol ows ..."
    ("Anem carried the grain", "Anem carried the grain"),
    ("they shared the grain", "they shared the grain"),
    ("the field will fail", "the field will fail"),   # the future needs its auxiliary
    ("it could fail", "it could fail"),
])
def test_generation_says_the_morphology_it_read(text, said):
    got = understand(ENGLISH, text)
    assert realize(ENGLISH, got.meanings[0]) == said


@pytest.mark.parametrize("text", ["Miol ought to give food", "Miol should give food"])
def test_a_modal_takes_a_bare_complement_in_both_directions(text):
    """"ought gave" was two bugs: a missing "to", and a tensed complement."""
    got = understand(ENGLISH, text)
    frame = got.meanings[0]
    assert frame.predicate == "give" and frame.feature("modality") == "should"
    assert realize(ENGLISH, frame) == "Miol should give food"


def test_a_modal_frame_that_carries_tense_is_said_by_the_modal_not_the_verb():
    """The sim builds this frame directly; it used to come out "Miol ought gave food"."""
    frame = Frame("give", {"subject": Entity("name", "Miol"), "object": Entity("name", "food")},
                  {"modality": "should", "tense": "past", "mood": "declarative"})
    said = realize(ENGLISH, frame)
    assert said == "Miol should give food"
    assert "gave" not in said and "ought gave" not in said


def test_a_tenseless_frame_is_not_said_with_a_past_form():
    """Saying "gave" for a tenseless meaning adds a past that nobody said.

    It is said in the present ("Miol gives food") because English has no tenseless
    finite form to say it in — so this pins the part that is a defect, the past, and
    not the part that is the language.
    """
    frame = Frame("give", {"subject": Entity("name", "Miol"), "object": Entity("name", "food")},
                  {"mood": "declarative"})
    said = realize(ENGLISH, frame)
    assert said == "Miol gives food" and "gave" not in said
    assert understand(ENGLISH, said).meanings[0].feature("tense") != "past"


def test_a_modal_cannot_be_read_with_a_tensed_complement():
    """"ought gave" is not English, so the grammar must not find it either."""
    got = understand(ENGLISH, "Miol should gave food")
    assert all(m.feature("modality") is None or m.predicate != "give" for m in got.meanings
               if hasattr(m, "feature"))


# ------------------------------------------------- the copula, from the civ-sim fork
#
# Five frames that world speaks constantly, all of which came out wrong: the copula
# was chosen alphabetically ("Nise am hungry") because nothing carried the subject's
# agreement into the verb, and a negated copula had to borrow a clause negation
# ("Kasa am never be trustworthy").


def _be(subject, complement, **features):
    return Frame("be", {"subject": subject, "object": complement},
                 {"tense": "present", "mood": "declarative", **features})


@pytest.mark.parametrize("frame,said", [
    (_be(Entity("name", "Nise"), Entity("name", "hungry")), "Nise is hungry"),
    (_be(Entity("description", "wood", {"noun": "wood"}), Entity("name", "dear")), "wood is dear"),
    (_be(Entity("description", "wood", {"noun": "wood"}), Entity("name", "cheap")), "wood is cheap"),
    (_be(Entity("name", "Kasa"), Entity("name", "trustworthy")), "Kasa is trustworthy"),
    (_be(Entity("name", "Kasa"), Entity("name", "trustworthy"), polarity="negative"),
     "Kasa is not trustworthy"),
])
def test_a_copula_agrees_with_its_subject_and_negates_in_place(frame, said):
    assert realize(ENGLISH, frame) == said


@pytest.mark.parametrize("text", ["the fields are empty", "the wells are dry",
                                  "the elders were hungry", "all the fields failed",
                                  "Nise is hungry", "wood is dear"])
def test_a_plural_or_copular_clause_round_trips(text):
    """A guessed noun claimed to be singular, which made every plural unsayable."""
    meaning = understand(ENGLISH, text).meanings[0]
    said = realize(ENGLISH, meaning)
    assert said == text
    assert understand(ENGLISH, said).meanings[0].describe() == meaning.describe()


def test_a_past_copula_is_said_in_the_past():
    frame = _be(Entity("name", "Nise"), Entity("name", "hungry"), tense="past")
    assert realize(ENGLISH, frame) == "Nise was hungry"


# --------------------------------------------- quantifiers, and the partitive "of"
#
# The civ-sim fork's dialects use synonyms for *much* — `plenty`, `heaps` — and those
# are partitives: they cannot stand in front of a noun without "of". `realize` returned
# None for them, so that world printed "Coralin holds heaps bread" from its fallback.
# Canonicalising a dialect back to the shared vocabulary produces the other half of the
# problem, "much of food": a bare quantifier with the partitive's "of" left behind.


def _holds(quantifier, noun="food", **features):
    number = "plural" if quantifier in ("many", "few") else "singular"
    thing = Entity("description", noun, {"noun": noun, "number": number,
                                         "quantifier": quantifier, **features})
    return Frame("hold", {"subject": Entity("name", "Coralin"), "object": thing},
                 {"tense": "present", "mood": "declarative"})


@pytest.mark.parametrize("quantifier,said", [
    ("much", "Coralin holds much food"),
    ("little", "Coralin holds little food"),
    ("some", "Coralin holds some food"),      # not "any food": that wants a question
    ("none", "Coralin holds no food"),
    ("many", "Coralin holds many foods"),
    ("few", "Coralin holds few foods"),
    # the dialect synonyms, which is where it broke: each needs its "of"
    ("plenty", "Coralin holds plenty of food"),
    ("heaps", "Coralin holds heaps of food"),
    ("lots", "Coralin holds lots of food"),
    ("loads", "Coralin holds loads of food"),
])
def test_a_quantifier_is_said_the_way_english_says_it(quantifier, said):
    assert realize(ENGLISH, _holds(quantifier)) == said


@pytest.mark.parametrize("quantifier", ["much", "little", "some", "many", "plenty", "heaps", "lots"])
def test_a_quantifier_survives_being_said_and_read_back(quantifier):
    said = realize(ENGLISH, _holds(quantifier))
    thing = understand(ENGLISH, said).meanings[0].role("object")
    assert thing.features.get("quantifier") == quantifier
    # the word stays in the text too: a caller that reads the text to find the amount
    # (the civ sim does) must keep finding it there
    assert quantifier in thing.text.split()


def test_an_amount_reaches_a_determiner_through_of():
    """"much of the food", never "much the food" — though "all the food" is fine."""
    assert realize(ENGLISH, _holds("much", definite=True)) == "Coralin holds much of the food"
    thing = understand(ENGLISH, "Coralin holds much of the food").meanings[0].role("object")
    assert thing.features.get("quantifier") == "much" and thing.features.get("definite")
    assert not thing.features.get("nonstandard")


@pytest.mark.parametrize("text,quantifier", [
    ("Coralin holds plenty food", "plenty"),    # a dropped "of"
    ("Coralin holds heaps bread", "heaps"),
    ("Coralin holds much of food", "much"),     # an "of" a canonicaliser left behind
    ("Coralin holds little of food", "little"),
])
def test_a_misplaced_of_is_understood_and_marked_rather_than_refused(text, quantifier):
    """Tolerant in, strict out: understood, marked repaired, and never said that way."""
    thing = understand(ENGLISH, text).meanings[0].role("object")
    assert thing.features.get("quantifier") == quantifier
    assert thing.features.get("nonstandard") is True
    assert realize(ENGLISH, _holds(quantifier)) != text
