"""The assistant's control layer: setting a request aside, picking it back up, automatizing it.

These drive the real ``hear``/``intentions``/decoders/interpreter/procedures with a fake shell
(``eval.control.harness``), so what is exercised is the control decisions rather than the screen.
"""

from __future__ import annotations

import pytest

from tensorcode.chunking import Chunks

from eval.control.harness import HOME, TREE, Conversation, redone, step_runs
from eval.control.legacy import as_before

AMBIGUOUS = dict(TREE, **{f"{HOME}/Desktop/report.pdf": "desktop copy\n"})


def statuses(conv: Conversation) -> dict[str, str]:
    return {r: conv.status(r) for r in conv.requests()}


def words_of(conv: Conversation, status: str) -> list[str]:
    from examples.browser_agents.assistant import agent
    import tensorcode as tc

    return [str(agent.one(conv.mind, tc.Ref(r), "words")) for r in conv.requests() if conv.status(r) == status]


# ------------------------------------------------------- interruption and resumption


def test_an_interruption_sets_the_question_aside_instead_of_dropping_it():
    conv = Conversation(TREE)
    conv.say("tidy up my desktop")
    assert words_of(conv, "awaiting") == ["tidy up my desktop"]
    turn = conv.say("make a folder called plans on the desktop")
    assert "Suspend" in turn.intentions and "Drop" not in turn.intentions
    assert not any("skipping" in r for r in turn.replies), "nothing was given up on"
    assert any(r.startswith("Back to it") for r in turn.replies), "the question came back by itself"


def test_resumption_does_not_re_run_finished_steps():
    conv = Conversation(TREE)
    for text in ("tidy up my desktop", "make a folder called plans on the desktop", "carry on", "1"):
        conv.say(text)
    assert redone(conv) == {}, "a resumed request must not redo work it had already done"
    assert max(step_runs(conv).values()) == 1
    assert [s for s in statuses(conv).values()] == ["done"] * 4


def test_the_old_layer_dropped_the_question_and_restarting_redid_its_work():
    """The before/after of the same conversation, so the improvement is not taken on faith."""
    with as_before():
        conv = Conversation(TREE)
        for text in ("tidy up my desktop", "make a folder called plans on the desktop", "carry on",
                     "tidy up my desktop", "1"):
            conv.say(text)
    assert "tidy up my desktop" in words_of(conv, "dropped")
    assert redone(conv).get("tidy up my desktop", 0) >= 8, "restating it re-ran the steps it had done"


def test_carry_on_names_no_goal_and_still_picks_the_right_one():
    conv = Conversation(AMBIGUOUS)
    conv.say("delete report.pdf")
    conv.say("how many words are in ~/Desktop/notes.txt")
    turn = conv.say("where were we")
    assert "Resume" in turn.intentions
    assert any("report.pdf" in r for r in turn.replies)
    assert words_of(conv, "new") == [], "the carry-on utterance is answered by the resumption itself"


def test_two_questions_can_be_held_at_once_and_neither_is_lost():
    conv = Conversation(AMBIGUOUS)
    for text in ("tidy up my desktop", "delete report.pdf", "how many words are in ~/Desktop/notes.txt"):
        conv.say(text)
    assert sorted(words_of(conv, "suspended") + words_of(conv, "awaiting")) == [
        "delete report.pdf", "tidy up my desktop"]
    # answering "1" to whichever comes back first works either way round, and which one that is
    # is arbitration's business, not this test's
    replies = []
    for text in ("carry on", "1", "carry on", "1"):
        replies += conv.say(text).replies
    assert any("Grouped" in r for r in replies) and any("Deleted" in r for r in replies)
    assert words_of(conv, "suspended") == [] and words_of(conv, "awaiting") == []


def test_a_question_re_asked_on_resumption_is_not_immediately_set_aside_again():
    """The live-lock this fixed: 'carry on' is a later utterance than the question it revives."""
    conv = Conversation(TREE, budget=60)
    conv.say("tidy up my desktop")
    conv.say("make a folder called plans on the desktop")
    turn = conv.say("carry on")
    assert turn.intentions.count("Resume") == 1, turn.intentions
    assert turn.intentions[-1] == "Finish"


# --------------------------------------------------------------------- automatization


def test_a_repeated_skill_stops_being_deliberated_and_says_the_same_thing():
    ask = "how many words are in ~/Desktop/notes.txt"
    plain = Conversation(TREE)
    said_plain = [plain.say(ask).replies for _ in range(8)]

    chunks = Chunks(repeats=3)
    automatic = Conversation(TREE, chunks=chunks)
    said_auto = [automatic.say(ask).replies for _ in range(8)]

    assert said_auto == said_plain, "an automatized skill that answers differently is a different skill"
    assert chunks.stats()["compiled"] == 2  # count, and the resolve it calls
    assert len(list(automatic.mind.claims(predicate="index"))) < len(list(plain.mind.claims(predicate="index")))
    assert [r.claim.object for r in automatic.mind.claims(predicate="ran_chunk")], "the graph records one act"


def test_a_chunk_is_given_up_when_a_step_goes_differently_and_the_expanded_form_still_works():
    ask = "how many words are in ~/Desktop/notes.txt"
    chunks = Chunks(repeats=3)
    conv = Conversation(TREE, chunks=chunks)
    for _ in range(6):
        conv.say(ask)
    assert chunks.chunk_for("count") is not None
    conv.shell.fail["wc -w"] = 1  # one step fails, mid-chunk
    broken = conv.say(ask)
    assert any("couldn't count" in r for r in broken.replies), broken.replies
    assert any("retired count" in h for h in chunks.history)
    after = conv.say(ask)
    assert after.replies == ["~/Desktop/notes.txt has 2 words."], "it recovers by deliberating again"


def test_chunking_off_is_the_shipped_behaviour():
    conv = Conversation(TREE, chunks=None)
    conv.say("how many words are in ~/Desktop/notes.txt")
    assert not list(conv.mind.claims(predicate="ran_chunk"))
    assert list(conv.mind.claims(predicate="index")), "every step is still recorded"


@pytest.fixture(autouse=True)
def _reset_host():
    """The assistant's host is module-level; a test that automatizes must not infect the next."""
    from examples.browser_agents.assistant import agent

    yield
    agent.HOST.chunks = None
