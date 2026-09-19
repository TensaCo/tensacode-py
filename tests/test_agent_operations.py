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

from tensorcode.agent import Agent
from tensorcode.agent.operations import Transcript, agent_runtime
from tensorcode.agent.plugin import Capability, Effect, Param, Plugin
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Ref
from tensorcode.runtime import Policy


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

    def _path(self, description):
        text = getattr(description, "text", "")
        words = [w for w in text.split() if w.lower() not in ("the", "file")]
        hits = [p for p in self.fs if words and p.rsplit("/", 1)[-1] == words[-1]]
        return hits[0] if len(hits) == 1 else Unknown("not_found", text)

    def refer(self, description, param, *, context):
        got = self._path(description)
        return Ref(f"path:{got}") if isinstance(got, str) else got

    def denote(self, description):
        got = self._path(description)
        return Ref(f"path:{got}") if isinstance(got, str) else got

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
    agent.turn("my name is Jacob.")
    assert "Jacob" in agent.turn("what is my name?").reply
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
    reply = agent.turn("delete report.txt.").reply
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
    agent.turn("move report.txt to notes.")
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
    outcome = agent.turn("delete report.txt.").outcomes[0]
    assert outcome.receipt.status == "applied"
    assert outcome.status == "failed"
    assert spans_of(agent, "verify"), "verification did not go through ops.verify"


def test_a_verified_action_is_reported_as_done():
    plugin = Papers()
    agent = Agent([plugin])
    outcome = agent.turn("delete report.txt.").outcomes[0]
    assert plugin.calls == ["delete"]
    assert outcome.status == "done" and outcome.verified is True


# ------------------------------------------------------------------ ranking


def test_two_remembered_answers_are_ranked_rather_than_returned_in_storage_order():
    agent = Agent([])
    agent.turn("the meeting is on Tuesday.")
    agent.turn("the meeting is on Wednesday.")
    reply = agent.turn("when is the meeting?").reply
    assert "Wednesday" in reply and "Tuesday" in reply  # both kept: nothing was overwritten
    assert reply.index("Wednesday") < reply.index("Tuesday"), "the later observation should lead"
    assert spans_of(agent, "rank"), "retrieval order did not go through ops.rank"
