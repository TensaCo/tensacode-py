"""The parsing machinery: unification, morphology both ways, and honest partial parses."""

import pytest

from tensacode.language import ENGLISH, Entry, Grammar, Lexicon, understand, words
from tensacode.language.chart import tokenize
from tensacode.language.features import FVar, ground, unify
from tensacode.language.grammar import Build, Head, Merge, OpenClass, guess_entries, inflect, production
from tensacode.language.semantics import Entity
from tensacode.language.semantics import Entity, Frame

TOY = Grammar(
    productions=(
        production("S -> NP[number=?n] VP[number=?n]", Merge(1, roles=(("subject", 0),), features=(("mood", "declarative"),))),
        production("NP[number=?n] -> N[number=?n]", Head(0)),
        production("VP[number=?n] -> V[number=?n]", Build(predicate_from=0, lift=(("tense", 0, "tense"),))),
    ),
    lexicon=Lexicon.of({
        "field": [Entry("field", "N", {"number": "singular"}, Entity("description", "field"))],
        "fields": [Entry("fields", "N", {"number": "plural"}, Entity("description", "fields"))],
        "fail": [Entry("fail", "V", {}, "fail")],
        "fails": [Entry("fails", "V", {"number": "singular"}, "fail")],
    }),
    start=("S",),
)


def test_unification_binds_variables_and_refuses_disagreement():
    assert unify({"number": FVar("n")}, {"number": "plural"}) == {"n": "plural"}
    assert unify({"number": "singular"}, {"number": "plural"}) is None
    assert ground({"number": FVar("n")}, {"n": "plural"}) == {"number": "plural"}


def test_agreement_is_enforced_by_the_same_mechanism():
    assert understand(TOY, "fields fail").best.complete
    mismatched = understand(TOY, "fields fails")
    assert not mismatched.best.complete  # no S spans it, so the cover reports the words


@pytest.mark.parametrize("word,cat,features,expected", [
    ("fail", "V", {"tense": "past"}, "failed"),      # not "faild"
    ("share", "V", {"tense": "past"}, "shared"),     # not "shareed"
    ("carry", "V", {"tense": "past"}, "carried"),
    ("open", "V", {"aspect": "progressive"}, "opening"),
    ("folder", "N", {"number": "plural"}, "folders"),  # not "folderes"
    ("box", "N", {"number": "plural"}, "boxes"),
])
def test_inflection_picks_the_rule_that_fits_the_stem(word, cat, features, expected):
    assert inflect(words(word, cat=cat, sem=word)[0], features) == expected


def test_morphology_runs_both_directions_from_one_table():
    entries = ENGLISH.lexicon.extend(*words("fail", cat="V", sem="fail")).lookup("failed")
    assert any(e.features.get("tense") == "past" for e in entries)
    assert inflect(words("fail", cat="V", sem="fail")[0], {"tense": "past"}) == "failed"
    assert inflect(words("share", cat="V", sem="share")[0], {"tense": "past"}) == "shared"
    assert inflect(words("folder", cat="N", sem="folder")[0], {"number": "plural"}) == "folders"


def test_a_nouns_s_is_a_plural_and_a_verbs_is_not():
    lexicon = ENGLISH.lexicon.extend(*words("field", cat="N", sem="field"), *words("fail", cat="V", sem="fail"))
    noun = [e for e in lexicon.lookup("fields") if e.cat == "N"]
    verb = [e for e in lexicon.lookup("fails") if e.cat == "V"]
    assert noun and noun[0].features["number"] == "plural"
    assert verb and verb[0].features["number"] == "singular" and verb[0].features["person"] == 3


def test_clitics_are_separate_words():
    assert tokenize("what's in it") == ["what", "'s", "in", "it"]
    assert tokenize("it didn't arrive") == ["it", "did", "n't", "arrive"]
    assert tokenize("~/Desktop/notes.txt") == ["~/Desktop/notes.txt"]
    assert tokenize("say 'don't forget'") == ["say", "'don't forget'"]


def test_unparsed_words_are_reported_not_guessed():
    got = understand(TOY, "the field failed xyzzy")
    assert "xyzzy" in got.skipped and got.coverage < 1.0
    assert got.best is not None  # a partial reading still beats no reading


def test_a_partial_reading_is_scored_against_a_full_one_on_one_scale():
    full = understand(TOY, "fields fail")
    partial = understand(TOY, "fields fail zzz")
    assert full.best.score > partial.best.score  # skipping is priced, not free
    assert full.coverage == 1.0 > partial.coverage


def test_genuine_ambiguity_survives_as_several_readings():
    got = understand(ENGLISH.extend(entries=[*words("field", cat="N", sem="field"), *words("good", cat="Adj")]),
                     "the field is good in the north")
    assert len(got.readings) > 1  # PP attachment is ambiguous, and both readings are kept


def test_a_literal_feature_demand_requires_the_feature_to_be_present():
    """Subcategorisation is a demand, not merely an absence of contradiction.

    Unification alone treats a missing feature as compatible, which made
    ``V[ditrans=true]`` match every verb: "make me a sandwich" then parsed as a
    double-object verb whose object was "me".
    """
    lexicon = Lexicon.of({
        "give": [Entry("give", "V", {"ditrans": True}, "give")],
        "make": [Entry("make", "V", {}, "make")],
        "me": [Entry("me", "Pron", {"person": 1}, Entity("pronoun", "me", {"person": 1}))],
        "cake": [Entry("cake", "N", {"number": "singular"}, Entity("description", "cake"))],
    })
    grammar = Grammar(
        productions=(
            production("IMP -> VP", Head(0)),
            production("VP -> V[ditrans=true] NP NP", Build(predicate_from=0, roles=(("object", 1), ("destination", 2)))),
            production("NP -> Pron", Head(0)),
            production("NP -> N", Head(0)),
        ),
        lexicon=lexicon,
        start=("IMP",),
    )
    assert understand(grammar, "give me cake").best.complete
    assert not understand(grammar, "make me cake").best.complete  # "make" is not ditransitive


def test_the_chart_finds_fragments_anywhere_not_only_from_the_start():
    got = understand(TOY, "zzz fields fail")
    assert got.best.nodes and got.best.nodes[0].start == 1


# --------------------------------------------------- morphology, both directions
#
# The coordinator's report: generation said "snow cames", "Miol ought gave food",
# and stemmed "owes" to "ow" and "died" to "di". Each case is pinned below, in the
# direction it went wrong.


@pytest.mark.parametrize("surface,stem,features", [
    ("died", "die", {"tense": "past"}),          # not "di"
    ("owes", "owe", {"number": "singular", "person": 3, "tense": "present"}),  # not "ow"
    ("carried", "carry", {"tense": "past"}),     # not "carri"
    ("shared", "share", {"tense": "past"}),      # not "shar"
    ("arrived", "arrive", {"tense": "past"}),    # not "arriv"
    ("failed", "fail", {"tense": "past"}),
    ("opened", "open", {"tense": "past"}),       # two vowel groups: no "e" comes back
    ("snows", "snow", {"number": "singular", "person": 3, "tense": "present"}),
    ("showed", "show", {"tense": "past"}),       # not "showe": "ow" is a diphthong
    ("followed", "follow", {"tense": "past"}),
])
def test_a_guessed_stem_is_one_the_same_table_inflects_back(surface, stem, features):
    spec = OpenClass(r".*", "V", sem="word", morphology=True)
    guessed = [e for e in guess_entries(surface, spec) if e.features.get("tense") or e.features.get("person")]
    assert [e.sem for e in guessed] == [stem]
    assert inflect(Entry(stem, "V", {}, stem), features) == surface


@pytest.mark.parametrize("surface,stem", [("boxes", "box"), ("folders", "folder"),
                                          ("stories", "story"), ("wishes", "wish")])
def test_a_guessed_plural_stem_strips_only_the_plural(surface, stem):
    spec = OpenClass(r".*", "N", sem="word", morphology=True)
    plural = [e.sem for e in guess_entries(surface, spec) if e.features.get("number") == "plural"]
    assert plural == [stem]


@pytest.mark.parametrize("wanted,expected", [
    ({"number": "singular", "person": 3}, "came"),  # not "cames": only present verbs agree
    ({"tense": "past"}, "came"),
    ({"tense": "present"}, None),                   # contradicted, and no form of it exists
    ({"aspect": "progressive"}, None),              # not "cameing"
])
def test_an_already_inflected_form_takes_no_second_suffix(wanted, expected):
    assert inflect(Entry("came", "V", {"tense": "past"}, "come"), wanted) == expected


@pytest.mark.parametrize("wanted,expected", [
    ({"number": "plural"}, "share"),                # "they share": English marks nothing
    ({"number": "plural", "tense": "present"}, "share"),
    ({"tense": "past"}, "shared"),
    ({"tense": "future"}, None),                    # marked by "will", not by a suffix
])
def test_unmarked_is_an_answer_but_unmarkable_is_a_refusal(wanted, expected):
    assert inflect(Entry("share", "V", {}, "share"), wanted) == expected


@pytest.mark.parametrize("text,tokens", [
    ("Coralin holds much food.", ["Coralin", "holds", "much", "food", "."]),
    ("open hi.txt.", ["open", "hi.txt", "."]),          # an internal dot is part of the word
    ("it is 3.14.", ["it", "is", "3.14", "."]),
    ("what is it?", ["what", "is", "it", "?"]),
])
def test_a_sentence_final_dot_is_not_part_of_the_last_word(text, tokens):
    """"food." matched no lexicon and no open-class pattern, so every sentence's last
    word entered as an unknown name — invisible here, expensive for a caller that
    counts unknown words."""
    assert tokenize(text) == tokens
