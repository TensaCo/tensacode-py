"""The desktop domain: acts read compositionally, and unclear input that stays unclear."""

import pytest

from tensorcode.language.domains.desktop import DESKTOP, Act, read_request


def acts_for(text):
    got, _ = read_request(text)
    return [(a.act, dict(a.slots)) for a in got]


@pytest.mark.parametrize("text,act,slots", [
    ("make a folder called recipes on my desktop", "create_folder", {"name": "recipes", "place": "~/Desktop"}),
    ("delete the recipes folder", "delete", {"target": "recipes"}),
    ("read shopping.txt", "read", {"target": "shopping.txt"}),
    ("what's on my desktop?", "list", {"place": "~/Desktop"}),
    ("how many files are in downloads", "list", {"place": "~/Downloads", "count": True}),
    ("go to documents", "cd", {"target": "~/Documents"}),
    ("what time is it", "info", {"topic": "date"}),
    ("who am i", "info", {"topic": "user"}),
    ("open firefox", "open_app", {"app": "Firefox"}),
    ("hi", "greet", {}),
    ("thanks", "thanks", {}),
])
def test_requests_read_to_the_expected_act(text, act, slots):
    got = acts_for(text)
    assert got, f"nothing read from {text!r}"
    assert got[0][0] == act, got
    for key, value in slots.items():
        assert got[0][1].get(key) == value, (key, got)


def test_chained_requests_become_several_acts_without_a_clause_splitter():
    got = acts_for("make a folder called site then go to documents")
    assert [a for a, _ in got] == ["create_folder", "cd"]


DESTRUCTIVE = {"delete", "move", "rename", "run", "install", "write"}


@pytest.mark.parametrize("text", [
    "make me a sandwich",
    "organize my desktop",
    "what's the weather",
    "tell me a joke",
    "fix my wifi",
    "move on",
    "remove the background from photo.png",
])
def test_unclear_requests_never_produce_a_destructive_act(text):
    assert not {a for a, _ in acts_for(text)} & DESTRUCTIVE, text


@pytest.mark.parametrize("text", [
    "make me a sandwich",
    "could you possibly show me what the notes file says",
    "show me a sandwich",
    "get me a coffee",
])
def test_a_dative_pronoun_never_becomes_the_thing_acted_on(text):
    """"me" names the person asking, so it can never fill a name or a target.

    Both of these previously projected to an act built out of the pronoun —
    ``create_folder(name='@it')`` and ``list(place='@it')`` — which would have made a
    folder from "make me a sandwich".
    """
    got = acts_for(text)
    assert got == [], got


def test_a_reference_is_only_used_when_the_utterance_made_one():
    # licensed: the utterance really does point at something
    assert acts_for("show it") == [("list", {"place": "@it"})]
    assert acts_for("delete it") == [("delete", {"target": "@it"})]
    assert acts_for("initialize a git repo there")[0][0] == "git_init"
    # unlicensed: no pronoun, no demonstrative, so no act may carry "@it"
    for text in ("make me a sandwich", "could you possibly show me what the notes file says"):
        assert not any("@it" in str(v) for _, slots in acts_for(text) for v in slots.values()), text


def test_an_act_that_changes_something_must_say_what():
    assert acts_for("move on") == []
    assert acts_for("make a folder") == []  # no name: the regexes handle this, we abstain


def test_a_half_understood_fragment_reads_as_nothing_rather_than_a_guess():
    got, understanding = read_request("zzz qqq wwww")
    assert got == []  # three unknown words are a fragment, not an instruction
    assert understanding.best is not None  # the words are still reported, not discarded


def test_the_grammar_reads_structure_the_act_vocabulary_cannot_express():
    """Reported speech, negation and modality parse even where no act applies.

    The assistant's act list has no place for these, so ``acts`` returns nothing —
    but the *parse* is there, which is what a caller with a richer vocabulary needs.
    """
    from tensorcode.language import understand
    from tensorcode.language.semantics import Frame

    got = understand(DESKTOP, "Anem said the file did not open")
    frame = got.meanings[0]
    assert isinstance(frame, Frame) and frame.predicate == "say"
    inner = frame.role("content")
    assert isinstance(inner, Frame) and inner.negated
