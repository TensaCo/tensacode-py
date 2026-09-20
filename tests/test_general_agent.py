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

from agent_test_support import selected_agent as Agent
from tensorcode.agent.plugin import Call, Capability, Effect, Informs, Param, Plugin
from tensorcode.language import Entity
from tensorcode.language import verbnet, wordnet
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Claim, Proposition, Ref, Var
from tensorcode.agent.scene import SceneGraph, SceneProposal


def _grounded_turn(agent, text, roles, *, projected_goal=None):
    """Supply identities and, optionally, an exact authored semantic projection.

    A projected goal is explicitly authored by each execution test. Binding an
    identity alone does not consume the selected frame's qualifications.
    """
    from tensorcode.agent.core import InterpretationDecision
    from tensorcode.agent.grounding import MentionBinding, propose_grounding
    from tensorcode.records import Ref

    evidence = agent.interpretations.add_source(
        "Test fixture explicitly supplies occurrence identities", provider="test-fixture")

    class AuthoredProjection(Plugin):
        expected = None

        def refine_goal(self, lexical):
            if projected_goal is not None and lexical.frame == self.expected:
                return projected_goal
            return Unknown("no_refinement")

    projection = AuthoredProjection("test-authored-projection")
    if projected_goal is not None:
        assert projected_goal.basis, "an authored projection must state its basis"
        agent.plugins.append(projection)

    def select(group):
        candidate = propose_grounding(agent.interpretations, group.id, group.candidates[0].id, [
            MentionBinding(("acts", 0, "frame", "roles", role), Ref(identity),
                           (evidence.id,), "authored binding for this test occurrence")
            for role, identity in roles.items()
        ])
        projection.expected = candidate.payload.acts[0].frame
        compared = agent.interpretations.get(group.id)
        return InterpretationDecision(candidate.id, "test supplies intended grounded reading", (evidence.id,),
            compared_revision=compared.revision,
            compared_candidate_ids=tuple(item.id for item in compared.candidates))
    agent.interpretation_selector = select
    try:
        return agent.turn(text)
    finally:
        if projected_goal is not None:
            agent.plugins.remove(projection)


def _supplied_move(source, destination):
    from tensorcode.goals import Condition, GoalSpec

    return GoalSpec((Condition("has_location", {"undergoer": Ref(source), "goal": Ref(destination)}),
                     Condition("has_location", {"undergoer": Ref(source)}, negated=True)),
                    basis=("authored-test:move-one-file-to-supplied-directory",))


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
                       informs=(Informs("has_location", "goal", "directory", query=Proposition(
                           "has_location", {"subject": Var("answer"), "object": Var("directory")})),), effect_kind="read"),
        )

    def display(self, r):
        return r.id.rsplit("/", 1)[-1] if r.id.startswith("path:") else None

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
    from tensorcode.goals import Condition, GoalSpec

    goal = GoalSpec((Condition("be", {"undergoer": ref("/h/Desktop/recipes")}),),
                    basis=("authored-test:folder-name-and-location-projected-to-exact-path",))
    turn = _grounded_turn(agent, "make a folder called recipes on my desktop", {"object": "path:/h/Desktop/recipes"},
                          projected_goal=goal)
    assert files.calls == ["make_directory"]
    assert "/h/Desktop/recipes" in files.fs
    assert turn.outcomes[0].status == "done"



def test_binding_folder_identity_does_not_project_its_qualifications(setup):
    files, agent = setup
    turn = _grounded_turn(agent, "make a folder called recipes on my desktop",
                          {"object": "path:/h/Desktop/recipes"})
    assert not files.calls
    assert "/h/Desktop/recipes" not in files.fs
    assert turn.outcomes[0].status in ("declined", "unknown")


def test_moving_somewhere_is_never_done_by_deleting(setup):
    """Regression: 'move X to documents' once ran delete, which achieves half the goal."""
    files, agent = setup
    _grounded_turn(agent, "move notes.txt to documents", {"object": "path:/h/Desktop/notes.txt", "destination": "path:/h/Documents"},
                   projected_goal=_supplied_move("path:/h/Desktop/notes.txt", "path:/h/Documents"))
    assert files.calls == ["move"]
    assert "/h/Documents/notes.txt" in files.fs


def test_a_supplied_canonical_question_is_answered_by_looking(setup):
    from tensorcode.agent.understand import Act, Sentence
    from tensorcode.language import Frame, Question

    files, agent = setup
    frame = Frame("has_location", {"location": Entity("description", "desktop", {"noun": "desktop"}, ref=ref("/h/Desktop"))})
    question = Question(frame, "subject")
    act = Act("question", question, frame)
    sentence = Sentence("supplied location question", (), None, (act,))
    outcome = agent.handle(sentence, act, [], requests_in_message=0)
    assert outcome.status == "answered"
    assert ref("/h/Desktop/notes.txt") in outcome.answer
    _grounded_turn(agent, "move notes.txt to documents", {"object": "path:/h/Desktop/notes.txt", "destination": "path:/h/Documents"},
                   projected_goal=_supplied_move("path:/h/Desktop/notes.txt", "path:/h/Documents"))
    outcome = agent.handle(sentence, act, [], requests_in_message=0)
    assert outcome.status == "answered" and outcome.answer == []


def test_a_request_no_capability_can_achieve_is_declined_without_acting(setup):
    files, agent = setup
    turn = agent.turn("design a device under 250 g.")
    assert files.calls == []
    assert turn.outcomes[0].status in ("declined", "unknown")


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
    """Semantic matching has no regex rules; HTTP/storage syntax is a separate concern."""
    repo = Path(__file__).parents[1]
    offenders = []
    for f in [*(repo / "src" / "tensorcode" / "agent").glob("*.py"),
              *(repo / "examples" / "general_agent" / name for name in ("desktop.py", "discover.py", "plugins.py"))]:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                if "re" in names or "regex" in names:
                    offenders.append(f.name)
    assert offenders == []


class Eyes(Plugin):
    """Supplied scene hypotheses; fixture strings are not inferred pixel content."""

    def __init__(self) -> None:
        super().__init__(name="eyes")

    def interpret_image(self, image, ref):
        if image is None:
            return
        thing = Ref(f"{ref.id}/{image}")
        yield SceneProposal(SceneGraph(
            ref, (thing,),
            (Proposition("is_a", {"entity": thing, "kind": image}),),
        ), provenance=("supplied test scene; no pixel inference",))


def test_picture_proposals_are_retained_and_later_picture_updates_focus():
    agent = Agent([Eyes()])
    first = agent.turn("what is in this picture?", images=["cat"])
    original_focus = agent.last_image
    group = agent.interpretations.get(first.visual_interpretation_ids[0])
    assert group.selected_id is None
    assert group.candidates[0].payload.graph.propositions[0].roles["kind"] == "cat"
    assert "cat" not in first.reply
    assert agent.store.propositions() == []

    second = agent.turn("what is in this photo?", images=["dog"])
    group = agent.interpretations.get(second.visual_interpretation_ids[0])
    assert agent.last_image != original_focus
    assert group.candidates[0].payload.graph.image == agent.last_image
    assert group.candidates[0].payload.graph.propositions[0].roles["kind"] == "dog"
    assert "dog" not in second.reply
    assert agent.store.propositions() == []


def test_empty_visual_proposal_group_is_retained_without_a_guess():
    agent = Agent([Eyes()])
    turn = agent.turn("what is in this picture?", images=[None])
    group = agent.interpretations.get(turn.visual_interpretation_ids[0])
    assert group.candidates == ()
    assert "cat" not in turn.reply and "dog" not in turn.reply
    assert agent.store.propositions() == []


def test_every_event_is_plain_json(setup):
    """The viewer streams events as JSON; a declined plan once held an Unknown object."""
    import json

    _, agent = setup
    for text in ("design a device.", "make a folder called x on my desktop", "what is on my desktop?", "hello there"):
        json.dumps(agent.turn(text).events)


def test_facts_you_tell_it_are_answered_from_the_right_side_of_the_claim():
    agent = Agent([])
    _grounded_turn(agent, "my name is Jacob.", {"subject": "fixture:name", "object": "fixture:Jacob"})
    assert "Jacob" in _grounded_turn(agent, "what is my name?", {"object": "fixture:name"}).reply
    _grounded_turn(agent, "I live in Austin.", {"subject": "fixture:speaker", "location": "fixture:Austin"})
    assert "Austin" in _grounded_turn(agent, "where do I live?", {"subject": "fixture:speaker"}).reply


def test_an_unrelated_question_is_not_answered_from_a_stored_fact():
    """Regression: 'when is the meeting?' answered 'name' from an unrelated be-claim."""
    agent = Agent([])
    agent.turn("my name is Jacob.")
    for question in ("when is the meeting?", "where do I live?", "what is on my desktop?"):
        reply = agent.turn(question).reply
        assert "Jacob" not in reply and "name" not in reply.lower()


def test_selected_location_reading_is_not_silently_rewritten_as_time():
    agent = Agent([])
    _grounded_turn(agent, "the meeting is on Tuesday.", {"subject": "fixture:meeting", "location": "fixture:Tuesday"})
    stored = [record.proposition for record in agent.store.propositions()]
    assert any(p.roles.get("location") == Ref("fixture:Tuesday") for p in stored)
    assert all("time" not in p.roles for p in stored)
    assert "Tuesday" not in _grounded_turn(agent, "when is the meeting?", {"object": "fixture:meeting"}).reply
