"""The agent's turn, as operations the library binds.

Before this, the agent was a program that happened to live next to tensorcode: it picked a
reader in its constructor, scored capabilities in a loop, and took the plugin's word for
whether an action had worked. None of it went through :mod:`tensorcode.ops`, so none of it
could be traced, substituted, or constrained by policy — the library's own machinery was
unreachable from the only program using it.

These tests pin the four places that changed. They are about *where the decision is made*,
not about what the agent happens to decide, so each asserts on the trace as much as on the
reply.
"""

from __future__ import annotations

import pytest

from agent_test_support import selected_agent as Agent
from tensorcode.agent.operations import Transcript, agent_runtime
from tensorcode.agent.plugin import Capability, Effect, Param, Plugin
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Ref
from tensorcode.runtime import Policy

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


class Papers(Plugin):
    """A few files, one capability to delete one, and a switch for whether deleting works.

    ``sticks=False`` is a plugin that reports success and changes nothing — the case that
    tells verification apart from trusting the receipt.
    """

    def __init__(self, entries=("/h/report.txt", "/h/notes.txt"), *, sticks: bool = True) -> None:
        super().__init__(name="papers", kinds={"file": ("path",), "report": ("file",)})
        self.fs = set(entries)
        self.sticks = sticks
        self.calls: list[str] = []

    def capabilities(self):
        return (Capability("delete", (Param("path", "path"),),
                           effects=(Effect("has_location", {"undergoer": "path"}, negated=True),)),
                Capability("move", (Param("path", "path"), Param("destination", "directory")),
                           effects=(Effect("has_location", {"undergoer": "path", "goal": "destination"}),
                                    Effect("has_location", {"undergoer": "path"}, negated=True))))

    def display(self, r):
        return r.id.rsplit("/", 1)[-1] if str(r.id).startswith("path:") else None

    def execute(self, act, *, key):
        self.calls.append(act.capability)
        if self.sticks:
            self.fs -= {v.id[5:] for _, v in act.args}
        return Receipt(act, "applied", idempotency_key=key)

    def holds(self, cap, args):
        return args["path"].id[5:] not in self.fs


def spans_of(agent, op):
    return [s for s in agent.runtime.trace.spans if s.op == op]


# ------------------------------------------------------------------ reading


def test_reading_is_a_parse_operation_with_the_reader_recorded():
    agent = Agent([])
    agent.turn("my name is Jacob.")
    [span] = spans_of(agent, "parse")
    assert span.target == "Transcript"
    assert [(a.implementation, a.outcome) for a in span.attempts] == [("reader:grammar", "answer")]


def test_which_reader_reads_is_the_runtime_policy_not_a_branch_in_the_agent():
    """The same agent code, two runtimes, two readers.

    This is the point of routing the read through an operation: swapping the treebank
    reader for the hand grammar is a change of policy, not a change of the agent.
    """
    pytest.importorskip("numpy")
    learned = Agent([], runtime=agent_runtime(prefer_reader="learned"))
    if not any(a.outcome == "answer" for a in _read(learned)):
        pytest.skip("no trained parser on this host")
    assert _by(learned) == "reader:learned"

    grammar = Agent([], runtime=agent_runtime())
    _read(grammar)
    assert _by(grammar) == "reader:grammar"


def _read(agent):
    agent.turn("my name is Jacob.")
    return spans_of(agent, "parse")[-1].attempts


def _by(agent):
    return next(a.implementation for a in spans_of(agent, "parse")[-1].attempts if a.outcome == "answer")


def test_a_reader_whose_requirement_is_missing_is_excluded_and_the_other_reads():
    """A host without the trained model is a policy fact, not a crash.

    ``Traits.requires`` says the treebank reader needs ``ud-parser``; a policy that does not
    list it excludes that implementation before it runs, and the cascade falls to the
    grammar. The turn still happens.
    """
    agent = Agent([], runtime=agent_runtime(prefer_reader="learned", policy=Policy(available=frozenset())))
    _grounded_turn(agent, "my name is Jacob.", {"subject": "fixture:name", "object": "fixture:Jacob"})
    assert "Jacob" in _grounded_turn(agent, "what is my name?", {"object": "fixture:name"}).reply
    [span] = [s for s in spans_of(agent, "parse")][:1]
    skipped = [a for a in span.attempts if a.outcome == "skipped"]
    assert any("ud-parser" in a.reason for a in skipped)
    assert _by(agent) == "reader:grammar"


def test_a_parse_returns_a_transcript_that_says_which_reader_made_it():
    from tensorcode import ops
    from tensorcode.runtime import use

    with use(agent_runtime()):
        transcript = ops.parse("the lamp is on.", Transcript)
    assert isinstance(transcript, Transcript) and transcript.by == "reader:grammar"
    assert len(transcript) == 1


# ------------------------------------------------------------------ choosing


def test_choosing_a_capability_is_a_choose_operation():
    agent = Agent([Papers()])
    reply = _grounded_turn(agent, "delete report.txt.", {"object": "path:/h/report.txt"}).reply
    assert "report.txt" in reply
    assert spans_of(agent, "choose"), "capability selection did not go through ops.choose"


def test_a_capability_that_would_do_only_part_of_it_is_excluded_by_a_constraint():
    """The constraint that stopped "move" from running "delete", now enforced by the library.

    ``ops.choose`` evaluates hard constraints itself, before any implementation sees the
    options, and the trace names the option it excluded and why. That is the difference
    between a rule and a comment: the reason is in the record.
    """
    plugin = Papers()
    agent = Agent([plugin])
    _grounded_turn(agent, "move report.txt to notes.", {"object": "path:/h/report.txt", "destination": "path:/h/notes"},
                   projected_goal=_supplied_move("path:/h/report.txt", "path:/h/notes"))
    notes = [n for s in spans_of(agent, "choose") for n in s.notes]
    assert any("does all of what was asked" in n for n in notes), notes
    assert "delete" not in plugin.calls


def test_nothing_is_chosen_when_no_capability_serves():
    agent = Agent([Papers()])
    outcome = agent.turn("delete the spreadsheet.").outcomes[0]
    assert outcome.status in ("declined", "unknown")


# ------------------------------------------------------------------ verifying


def test_verification_is_a_fresh_observation_not_the_receipt():
    """A receipt saying "applied" is the executor's report about itself.

    The plugin here reports success and leaves the world unchanged. The turn must come back
    as *failed*, because what is checked afterwards is what can still be seen.
    """
    agent = Agent([Papers(sticks=False)])
    outcome = _grounded_turn(agent, "delete report.txt.", {"object": "path:/h/report.txt"}).outcomes[0]
    assert outcome.receipt.status == "applied"
    assert outcome.status == "failed"
    assert spans_of(agent, "verify"), "verification did not go through ops.verify"


def test_a_verified_action_is_reported_as_done():
    plugin = Papers()
    agent = Agent([plugin])
    outcome = _grounded_turn(agent, "delete report.txt.", {"object": "path:/h/report.txt"}).outcomes[0]
    assert plugin.calls == ["delete"]
    assert outcome.status == "done" and outcome.verified is True


# ------------------------------------------------------------------ ranking


def test_two_remembered_answers_are_ranked_rather_than_returned_in_storage_order():
    agent = Agent([])
    _grounded_turn(agent, "my name is Jacob.", {"subject": "fixture:name", "object": "fixture:Jacob"})
    _grounded_turn(agent, "my name is Jane.", {"subject": "fixture:name", "object": "fixture:Jane"})
    reply = _grounded_turn(agent, "what is my name?", {"object": "fixture:name"}).reply
    assert "Jane" in reply and "Jacob" in reply  # both kept: nothing was overwritten
    assert reply.index("Jane") < reply.index("Jacob"), "the later observation should lead"
    assert spans_of(agent, "rank"), "retrieval order did not go through ops.rank"
