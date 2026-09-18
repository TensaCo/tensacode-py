"""The general agent core, against an in-memory file plugin (no computerworld needed).

These pin behaviour the owner asked for, not phrasings: a request is achieved only by a
capability whose effects achieve it; part of a request is never acted on; quoted
language is mentioned, not obeyed; and the agent package contains no regular
expressions. They need WordNet and VerbNet on disk and skip without them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tensorcode.agent.core import Agent
from tensorcode.agent.plugin import Call, Capability, Effect, Informs, Param, Plugin
from tensorcode.language import Entity
from tensorcode.language import verbnet, wordnet
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Claim, Ref

pytestmark = pytest.mark.skipif(wordnet.find_wordnet() is None or verbnet.find_verbnet() is None,
                                reason="needs WordNet and VerbNet data on disk")


def ref(path: str) -> Ref:
    return Ref(f"path:{path}")


class Files(Plugin):
    """A tiny file system: directories are paths ending without a dot, files have one."""

    def __init__(self, entries: set[str]) -> None:
        super().__init__(name="files", kinds={"folder": ("directory",), "directory": ("path",), "file": ("path",),
                                              "desktop": ("directory",), "documents": ("directory",)})
        self.fs = set(entries)
        self.calls: list[str] = []

    def capabilities(self):
        return (
            Capability("make_directory", (Param("path", "directory"),), effects=(Effect("be", {"undergoer": "path"}),)),
            Capability("delete", (Param("path", "path"),), effects=(Effect("has_location", {"undergoer": "path"}, negated=True),)),
            Capability("move", (Param("path", "path"), Param("destination", "directory")),
                       effects=(Effect("has_location", {"undergoer": "path", "goal": "destination"}),
                                Effect("has_location", {"undergoer": "path"}, negated=True))),
            Capability("list_directory", (Param("directory", "directory"),),
                       informs=(Informs("has_location", "goal", "directory"),), effect_kind="read"),
        )

    places = {"desktop": "/h/Desktop", "documents": "/h/Documents"}

    def _path(self, d, creating):
        if isinstance(d, Entity) and d.kind == "path":
            hits = [p for p in self.fs if p.rsplit("/", 1)[-1] == d.text]
            return hits[0] if len(hits) == 1 else Unknown("not_found", d.text)
        if not isinstance(d, Entity):
            return Unknown("cannot_refer", repr(d))
        noun = d.features.get("noun")
        if d.text.lower() in self.places:
            return self.places[d.text.lower()]
        if noun in self.places:
            return self.places[noun]
        name = d.features.get("name")
        if name is not None:
            parent = self._path(d.features["location"], False) if d.features.get("location") is not None else "/h"
            return f"{parent}/{name.text}"
        words = [w for w in d.text.split() if w.lower() not in (noun, "the")]
        hits = [p for p in self.fs if words and p.rsplit("/", 1)[-1] == words[0]]
        return hits[0] if len(hits) == 1 else Unknown("not_found", d.text)

    def refer(self, description, param, *, context):
        got = self._path(description, creating=param.kind == "directory")
        return ref(got) if isinstance(got, str) else got

    def denote(self, description):
        got = self._path(description, creating=False)
        return ref(got) if isinstance(got, str) else got

    def display(self, r):
        return r.id.rsplit("/", 1)[-1]

    def execute(self, act: Call, *, key):
        a = {k: v.id[5:] for k, v in act.args}
        self.calls.append(act.capability)
        if act.capability == "make_directory":
            self.fs.add(a["path"])
        elif act.capability == "delete":
            self.fs = {p for p in self.fs if not (p == a["path"] or p.startswith(a["path"] + "/"))}
        elif act.capability == "move":
            name = a["path"].rsplit("/", 1)[-1]
            self.fs.discard(a["path"])
            self.fs.add(f"{a['destination']}/{name}")
        return Receipt(act, "applied", idempotency_key=key)

    def holds(self, cap, args):
        a = {k: v.id[5:] for k, v in args.items()}
        if cap.name == "make_directory":
            return a["path"] in self.fs
        if cap.name == "delete":
            return a["path"] not in self.fs
        if cap.name == "move":
            return f"{a['destination']}/{a['path'].rsplit('/', 1)[-1]}" in self.fs
        return Unknown("no_check", cap.name)

    def reveal(self, cap, args, receipt):
        d = args["directory"].id[5:]
        for p in sorted(self.fs):
            if p.rsplit("/", 1)[0] == d:
                yield Claim(ref(p), "has_location", ref(d))


@pytest.fixture
def setup():
    files = Files({"/h/Desktop", "/h/Documents", "/h/Desktop/notes.txt", "/h/Documents/handbook.txt"})
    return files, Agent([files])


def test_a_request_is_achieved_by_the_capability_whose_effect_it_needs(setup):
    files, agent = setup
    turn = agent.turn("make a folder called recipes on my desktop")
    assert files.calls == ["make_directory"]
    assert "/h/Desktop/recipes" in files.fs
    assert turn.outcomes[0].status == "done"


def test_moving_somewhere_is_never_done_by_deleting(setup):
    """Regression: 'move X to documents' once ran delete, which achieves half the goal."""
    files, agent = setup
    agent.turn("move notes.txt to documents")
    assert files.calls == ["move"]
    assert "/h/Documents/notes.txt" in files.fs


def test_a_question_is_answered_by_looking_and_an_empty_look_is_an_answer(setup):
    files, agent = setup
    assert "notes.txt" in agent.turn("what is on my desktop?").reply
    agent.turn("move notes.txt to documents")
    assert "nothing" in agent.turn("what is on my desktop?").reply.lower()


def test_a_request_no_capability_can_achieve_is_declined_without_acting(setup):
    files, agent = setup
    turn = agent.turn("design a device under 250 g.")
    assert files.calls == []
    assert turn.outcomes[0].status == "declined"


def test_quoted_language_is_mentioned_not_obeyed(setup):
    files, agent = setup
    turn = agent.turn('the spec says:\n"make a folder called robots on my desktop."')
    assert files.calls == []
    assert any(o.status == "mentioned" for o in turn.outcomes)


def test_one_reply_per_message(setup):
    _, agent = setup
    turn = agent.turn("make a folder called a1 on my desktop. make a folder called a2 on my desktop.")
    assert isinstance(turn.reply, str) and turn.reply.count("\n") == 0


def test_the_agent_package_contains_no_regular_expressions():
    """The owner's rule: no hardcoded regex matching in the agent. Tokenizing lives in language/."""
    repo = Path(__file__).parents[1]
    offenders = []
    for f in [*(repo / "src" / "tensorcode" / "agent").glob("*.py"), *(repo / "examples" / "general_agent").glob("*.py")]:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                if "re" in names or "regex" in names:
                    offenders.append(f.name)
    assert offenders == []


class Eyes(Plugin):
    """A stand-in vision plugin: 'sees' whatever label the test hands it as the image."""

    def __init__(self) -> None:
        super().__init__(name="eyes")

    def see(self, image, ref):
        if image is None:
            return
        thing = Ref(f"{ref.id}/{image}")
        yield Claim(thing, "has_location", ref)
        yield Claim(thing, "is_a", image)

    def display(self, r):
        return "a " + r.id.rsplit("/", 1)[-1] if "/" in r.id else super().display(r)


def test_what_is_in_this_picture_is_answered_from_what_was_seen():
    agent = Agent([Eyes()])
    assert "cat" in agent.turn("what is in this picture?", images=["cat"]).reply
    # a later picture replaces "this picture"
    assert "dog" in agent.turn("what is in this photo?", images=["dog"]).reply


def test_nothing_seen_is_said_as_nothing_not_a_guess():
    agent = Agent([Eyes()])
    turn = agent.turn("what is in this picture?", images=[None])
    assert "cat" not in turn.reply and "dog" not in turn.reply


def test_every_event_is_plain_json(setup):
    """The viewer streams events as JSON; a declined plan once held an Unknown object."""
    import json

    _, agent = setup
    for text in ("design a device.", "make a folder called x on my desktop", "what is on my desktop?", "hello there"):
        json.dumps(agent.turn(text).events)
