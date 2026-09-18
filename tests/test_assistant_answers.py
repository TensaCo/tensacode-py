"""Answering, not only acting: the five prompts from a real failing transcript.

ACCEPTANCE (held out): the user pasted these five after chatting with the assistant. Every
one of them either got "I don't know how to do that" or — worse — a confident listing of
the home folder. They are the truth this file exists to check, and none of them is
special-cased anywhere: each is parsed by the ordinary grammar and answered by an ordinary
procedure.

The cases further down are mine. They are a FLOOR on obvious breakage, not evidence of
coverage: three times in this project a set written beside the code passed while real use
broke, so treat anything below the acceptance block as the weaker kind of evidence.
"""

from __future__ import annotations

import tensorcode as tc
from tensorcode.cognition import Thought

from examples.browser_agents.assistant import agent as A
from examples.browser_agents.assistant import interpreter as I
from examples.browser_agents.assistant import procedures as PR
from examples.browser_agents.assistant.language import parse_message

HOME = "/home/agent"


class Control:
    """Just enough of a perceived control: a label and where it is on screen."""

    def __init__(self, role: str, name: str, box: tuple[int, int, int, int], section: str = ""):
        self.role, self.name, self.box, self.section = role, name, box, section


class Text:
    def __init__(self, text: str, box: tuple[int, int, int, int], section: str = ""):
        self.text, self.box, self.section = text, box, section


#: a desktop like the one the user was looking at: a launcher strip of 13 icons, two windows
DOCK = ["Files", "Firefox", "Chromium", "Mail", "Rhythmbox", "Terminal", "Text Editor",
        "Visual Studio Code", "Slack", "App Center", "Settings", "System Monitor", "Wireshark"]


def desktop_mind() -> tc.Store:
    mind = A.new_mind()
    claims: list[tuple[tc.Claim, None]] = []
    entities: list[tuple[tc.Ref, object]] = []
    for i, app in enumerate(DOCK):
        ref = tc.Ref(f"ui:/button/{app}#1")
        control = Control("button", app, (16, 120 + i * 48, 48, 48))
        entities.append((ref, control))
        claims += [(tc.Claim(ref, "label", app, scope=A.SPEC and tc.Ref("scope:screen")), None),
                   (tc.Claim(ref, "is_a", "button", scope=tc.Ref("scope:screen")), None)]
    for i, (window, line) in enumerate([("Files", "~/Desktop"), ("Files", "2 items"), ("Wireshark", "Capturing")]):
        ref = tc.Ref(f"text:{window}#{i}")
        entities.append((ref, Text(line, (300 + i, 200 + i * 20, 200, 18), window)))
        claims.append((tc.Claim(ref, "reads", line, scope=tc.Ref("scope:screen")), None))
    from tensorcode.cognition import Fragment, integrate

    integrate(mind, Fragment(tc.Ref("obs:frame-1"), tuple(claims), tuple(entities), snapshot_of=tc.Ref("scope:screen"), method="test"))
    return mind


class FakeBody:
    """Answers the two body steps a question can need: pixels, and clicking an app open."""

    def __init__(self, colors=((16, 16, 18, 0.7), (72, 40, 62, 0.2), (200, 60, 55, 0.1)), opens=True):
        self.colors, self.opens, self.sampled, self.opened = list(colors), opens, [], []

    def sample(self, region: str) -> dict:
        self.sampled.append(region)
        if region and region not in ("screen", "display", "desktop", "background", "terminal", "Files"):
            return {"unavailable": f"I don't know which part of the screen “{region}” is", "colors": [], "mean": None, "box": None}
        return {"colors": [list(c) for c in self.colors], "mean": [40, 30, 35], "box": [0, 0, 1280, 800], "unavailable": None}


def ask(mind: tc.Store, message: str, body: FakeBody | None = None, turn: int = 1) -> list[str]:
    """Hear one message and run every request in it to the end, as the live loop would."""
    body = body or FakeBody()
    said: list[str] = []

    def say(mind_, req_, text, cycle):
        said.append(text)
        return Thought()

    host = I.Host(say=say, find=A.find_procedure)
    A.hear(mind, message, turn)
    for req in [r for r in A.open_requests(mind)]:
        frame = A.frame_of(mind, req)
        proc = A.procedure_for(frame) or PR.BY_ID["unknown"]
        I.begin(mind, req, proc, A._env_for(mind, frame), 0)
        value = None
        for cycle in range(300):
            I.advance(mind, req, value, cycle, host)
            if I.one(mind, req, "status") in ("done", "failed"):
                break
            doing = I.one(mind, req, "doing")
            if doing is None:
                break
            kind = I.one(mind, req, "doing") and I.one(mind, doing, "kind")
            if kind == "sample":
                value = body.sample(I.one(mind, doing, "region"))
            elif kind == "open_app":
                body.opened.append(I.one(mind, doing, "app"))
                value = body.opens
            elif kind == "run":
                value = type("Out", (), {"text": "", "cwd": None, "timed_out": False})()
            else:
                value = True
        A.set_state(mind, req, "status", I.one(mind, req, "status") or "done", "test")
    return said


# --------------------------------------------------------------- acceptance


def test_hello_still_works():
    assert "Hi!" in ask(desktop_mind(), "hello")[0]


def test_being_told_a_name_and_asked_for_it_in_one_message():
    """The transcript's worst case: both halves answered "I don't know how to do that"."""
    said = ask(desktop_mind(), "my name is Jacob. what is my name?")
    assert len(said) == 2, said
    assert "Jacob" in said[0] and "name" in said[0]
    assert "Jacob" in said[1], said[1]
    assert "don't know how" not in said[1] and "didn't understand" not in said[1]


def test_a_colour_question_looks_at_pixels_and_does_not_list_a_folder():
    body = FakeBody()
    said = ask(desktop_mind(), "what color is the display", body)
    assert body.sampled == ["display"], body.sampled
    assert "black" in said[-1], said
    assert "items" not in said[-1] and "Desktop" not in said[-1]


def test_counting_the_launcher_icons_answers_from_what_was_perceived():
    said = ask(desktop_mind(), "how many icons are in the sidebar")
    assert f"{len(DOCK)} icons" in said[-1], said
    assert "Terminal" in said[-1] and "Visual Studio Code" in said[-1]
    assert "folders" not in said[-1]  # not a count of the home folder


def test_an_app_named_by_what_it_is_for_gets_opened():
    body = FakeBody()
    said = ask(desktop_mind(), "open the app that is used for writing code", body)
    assert body.opened == ["Visual Studio Code"], body.opened
    assert "Visual Studio Code" in said[-1], said


# ------------------------------------------------------------------- floor
# Mine, written beside the code: a floor on obvious breakage, not coverage.


def test_a_told_fact_can_be_corrected_and_the_old_value_is_named():
    mind = desktop_mind()
    ask(mind, "my name is Jacob", turn=1)
    said = ask(mind, "call me Jake", turn=2)
    assert "Jake" in said[-1] and "Jacob" in said[-1], said
    assert "Jake" in ask(mind, "what is my name", turn=3)[-1]


def test_asking_for_something_never_told_says_what_it_would_need():
    mind = desktop_mind()
    ask(mind, "my name is Jacob", turn=1)
    said = ask(mind, "what is my favourite colour", turn=2)
    assert "don't know" in said[-1] and "name" in said[-1], said  # names what it does have


def test_forgetting_drops_the_fact():
    mind = desktop_mind()
    ask(mind, "my name is Jacob", turn=1)
    assert "no longer" in ask(mind, "forget my name", turn=2)[-1]
    assert "haven't told me" in ask(mind, "what is my name", turn=3)[-1]


def test_what_did_i_tell_you_lists_the_facts():
    mind = desktop_mind()
    ask(mind, "my name is Jacob", turn=1)
    ask(mind, "my favourite colour is blue", turn=2)
    said = ask(mind, "what did I tell you", turn=3)
    assert "Jacob" in said[-1] and "blue" in said[-1], said


def test_which_windows_are_open_comes_from_the_screen():
    said = ask(desktop_mind(), "which apps are open")
    assert "Files" in said[-1] and "Wireshark" in said[-1], said


def test_whats_on_the_screen_summarises_windows_and_launcher():
    said = ask(desktop_mind(), "whats on the screen")
    assert "2 windows" in said[-1] and "13 icons" in said[-1], said


def test_a_region_it_cannot_find_is_refused_with_the_reason():
    body = FakeBody()
    said = ask(desktop_mind(), "what colour is the fridge", body)
    assert "can't do that" in said[-1] and "fridge" in said[-1], said


def test_an_unknown_function_says_what_it_can_see():
    said = ask(desktop_mind(), "open the app that is used for tarot readings")
    assert "don't know" in said[-1] and "Terminal" in said[-1], said


def test_asking_what_i_did_before_doing_anything_is_honest():
    assert "haven't done anything" in ask(desktop_mind(), "what did you just do")[-1]


def test_the_three_failures_are_recorded_in_the_graph():
    mind = desktop_mind()
    ask(mind, "florble the wibbit")
    kinds = {r.claim.object for r in mind.claims(predicate="failed_because")}
    assert kinds == {"not_understood"}, kinds


def test_every_new_procedure_round_trips_through_json():
    from examples.browser_agents.assistant.procedure import Procedure

    for proc in PR.PROCEDURES:
        proc.check()
        assert Procedure.from_json(proc.to_json()).steps == proc.steps
