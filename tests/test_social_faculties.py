"""What the four faculties change in the assistant's actual replies.

The library tests (tests/test_social_ground.py) say the structures are right; these say they
make a behavioural difference, which is the only thing that keeps common ground from being
bookkeeping. Everything runs against the fake machine in eval/social/measure.py, so no server
and no display are involved.
"""

from __future__ import annotations

import pytest
import tensacode as tc

from eval.social.measure import TREE, FakeShell, read_frame, run_request


def reply_to(text: str, *, answers: list[str] | None = None, mind: tc.Store | None = None,
             turn: int = 1, shell: FakeShell | None = None) -> tuple[list[str], list[str], tc.Store]:
    frame = read_frame(text, turn)  # every reading tier, in the order the live assistant uses
    return run_request(frame, shell or FakeShell(TREE), answers or [], mind=mind, turn=turn)


# ------------------------------------------ 1. a false presupposition is corrected


@pytest.mark.parametrize("asked, near", [
    ("delete the report.txt", "reports.txt"),   # the owner's own prompt
    ("read notse.txt", "notes.txt"),
    ("delete recipe", "recipes"),
    ("read shoping.txt", "shopping.txt"),
])
def test_a_wrong_name_is_corrected_not_reported_absent(asked, near):
    said, _commands, _mind = reply_to(asked)
    assert "did you mean" in said[-1].lower() and near in said[-1]


def test_nothing_near_is_still_a_plain_absence():
    """A correction has to be able to not happen, or it is a guess dressed as help."""
    said, _c, _m = reply_to("read zzzqqq.txt")
    assert "couldn't find" in said[-1].lower() and "did you mean" not in said[-1].lower()


@pytest.mark.parametrize("remark", ["I can't find my keys", "I'm looking for a new job", "I can't find my glasses"])
def test_the_gate_applies_to_every_reading_tier_not_only_the_indirect_one(remark):
    """These three were read as searches by the symbolic grammar tier, which the gate did not
    reach until it was applied to that tier's output as well (see docs/revival/24)."""
    from examples.browser_agents.assistant.agent import _read_with_grammar
    from examples.browser_agents.assistant.language import act_is_affordable

    ungated = _read_with_grammar(remark)
    assert ungated is not None and ungated.act != "unknown"  # the tier does offer a reading
    assert act_is_affordable(remark, ungated) is False  # and the gate refuses it
    assert read_frame(remark).act == "unknown"


def test_an_order_is_never_second_guessed_by_the_gate():
    """"delete keys" is an order about a file called keys; the gate is not for orders."""
    from examples.browser_agents.assistant.language import act_is_affordable

    frame = read_frame("delete keys")
    assert frame.act == "delete" and frame.slots.get("target") == "keys"
    assert act_is_affordable("delete keys", frame) is True


def test_a_near_miss_is_found_outside_the_folder_you_named():
    """reports.txt is on the Desktop, not in the home folder the request assumed."""
    said, commands, _m = reply_to("delete the report.txt")
    assert any("-iname" in c for c in commands)  # the prefix net, after the exact name failed
    assert "reports.txt" in said[-1]


def test_a_missing_folder_is_not_called_an_empty_one():
    """"organise my videos" presupposes a Videos folder; saying it is empty would be false."""
    said, _c, _m = reply_to("organise my videos")
    assert "there's no" in said[-1].lower() and "empty" not in said[-1].lower()


def test_nothing_destructive_happens_on_a_corrected_request():
    _said, commands, _m = reply_to("delete the report.txt")
    assert not any(c.startswith(("rm", "mv", "mkdir")) for c in commands)


# --------------------------------------------- 2. clarification when the goal is open


def test_an_underdetermined_goal_asks_and_then_acts():
    said, commands, _m = reply_to("organize my desktop", answers=["1"])
    assert "?" in said[0] and "1." in said[0]  # one question, with the options priced behind it
    assert any(" && mv " in c for c in commands)
    assert "Grouped" in said[-1]


def test_the_question_offers_only_what_can_be_done_and_admits_the_rest():
    """By-date is offered but refused loudly: a listing carries names, not dates."""
    said, commands, _m = reply_to("organize my desktop", answers=["2"])
    assert not any(" && mv " in c for c in commands)
    assert "can't" in said[-1].lower() or "cannot" in said[-1].lower()


@pytest.mark.parametrize("clear", [
    "whats on my desktop", "read notes.txt", "delete notes.txt", "find all pdfs",
    "make a folder called recipes on my desktop", "what is my name", "copy notes.txt to documents",
])
def test_a_clear_request_is_never_questioned(clear):
    assert read_frame(clear).act != "clarify_goal"


def test_declining_the_question_leaves_the_folder_alone():
    said, commands, _m = reply_to("organize my desktop", answers=[])  # no answer: treated as cancel
    assert not any(" && mv " in c for c in commands)
    assert "leave it" in said[-1].lower()


# ------------------------------------------ 3. an indirect request is read as a request


@pytest.mark.parametrize("text, act", [
    ("it would be good if you deleted notes.txt", "delete"),
    ("it'd be nice if you renamed notes.txt to ideas.txt", "rename"),
    ("would you mind deleting old.log", "delete"),
    ("I can't find my invoice", "find"),
    ("where did my report.pdf go", "find"),
    ("is there a readme?", "find"),
    ("my desktop is a mess", "clarify_goal"),
])
def test_a_request_in_another_form_is_still_a_request(text, act):
    assert read_frame(text).act == act


@pytest.mark.parametrize("text", [
    "I can't find my keys", "I can't find my wallet", "I'm looking for a new job",
    "I wish I had more time", "it would be good if it stopped raining", "the weather is a mess",
    "my mood is a mess",
])
def test_a_remark_about_something_i_cannot_touch_stays_a_remark(text):
    """Through every tier: the grammar tier used to read three of these as searches."""
    read = read_frame(text).act
    if read not in ("unknown", "tell"):
        pytest.fail(f"{text!r} was read as {read}")


def test_a_complaint_about_a_folder_is_not_filed_as_a_fact_about_you():
    """"my desktop is a mess" used to be stored as a fact; the affordance decides which it is."""
    assert read_frame("my desktop is a mess").act == "clarify_goal"
    assert read_frame("my mood is a mess").act == "tell"


# ------------------------------------------------- 4. common ground marks a repeat


def test_the_second_time_i_say_it_is_marked_and_the_first_is_not():
    mind, shell = tc.Store(), FakeShell(TREE)
    reply_to("my name is Jacob", mind=mind, turn=1, shell=shell)
    first, _c, _m = reply_to("what is my name", mind=mind, turn=2, shell=shell)
    second, _c, _m = reply_to("what is my name", mind=mind, turn=3, shell=shell)
    assert "as i mentioned" not in first[-1].lower()
    assert "as i mentioned" in second[-1].lower()
    assert "Jacob" in second[-1]  # marked, not withheld


def test_what_did_i_tell_you_is_complete_and_in_telling_order():
    mind, shell = tc.Store(), FakeShell(TREE)
    reply_to("my name is Jacob", mind=mind, turn=1, shell=shell)
    reply_to("my favourite colour is green", mind=mind, turn=2, shell=shell)
    said, _c, _m = reply_to("what did I tell you", mind=mind, turn=3, shell=shell)
    listing = said[-1]
    assert "Jacob" in listing and "green" in listing
    assert listing.index("Jacob") < listing.index("green")


def test_hearing_a_fact_grounds_it_as_yours():
    from tensacode.social import YOU_SAID, CommonGround

    mind = tc.Store()
    reply_to("my name is Jacob", mind=mind, turn=1)
    told = CommonGround(mind).told_me()
    assert len(told) == 1 and told[0].how == YOU_SAID and told[0].turn == 1


def test_the_mark_survives_forgetting_because_ground_is_not_perception():
    """Both sides of the exchange outlive a perceptual sweep, not just the fact you told me.

    Regression guard for a cross-cutting bug found by measurement: `YOU_SAID` was grounded with an
    `utterance:` source and so protected, while `I_SAID` carried a `reply:` source and was swept on
    the perceptual clock. The assistant kept the fact and its provenance but silently stopped
    marking a repeat. Forgetting is for perception; an utterance of mine is not perception either.
    """
    from datetime import timedelta

    from tensacode.memory import Memory, MemoryPolicy
    from tensacode.social import CommonGround

    mind, shell = tc.Store(), FakeShell(TREE)
    reply_to("my name is Jacob", mind=mind, turn=1, shell=shell)
    reply_to("what is my name", mind=mind, turn=2, shell=shell)  # now I have said it too
    ground = CommonGround(mind)
    about = ground.told_me()[0].about
    assert ground.again(about) is True

    report = Memory(mind, MemoryPolicy(half_life=timedelta(0), min_salience=1.0)).forget_stale()
    assert report.claims_forgotten > 0  # the sweep really ran
    assert ground.again(about) is True and len(ground.told_me()) == 1

    said, _c, _m = reply_to("what is my name", mind=mind, turn=3, shell=shell)
    assert "as i mentioned" in said[-1].lower() and "Jacob" in said[-1]
