"""Requests whose product is language: the agent explaining itself.

Three things the agent could not do, measured before any of this existed: "explain your
reasoning" parsed perfectly and then found no capability to run, "what are your capabilities?"
found nothing that informs on ``be``, and neither failure was in the language layer — VerbNet
gives ``explain`` the goal ``has_information(Recipient, Topic)`` and always did. What was
missing was a capability whose effect is that someone knows something.

So these tests are about whether the *content* is real. Each one asserts against the record it
should have come from — the spans the runtime actually opened, the plugin registry the agent
actually mounted, the propositions the store actually holds — rather than against wording,
because a reply that matched a phrase list would prove only that something was said.
"""

from __future__ import annotations

import pytest

from agent_test_support import selected_agent as Agent
from tensorcode.agent.discourse import REPORTS, DiscoursePlugin
from tensorcode.agent.plugin import Capability, Effect, Param, Plugin, describe_capabilities
from tensorcode.language import verbnet, wordnet
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Ref

pytestmark = pytest.mark.skipif(wordnet.find_wordnet() is None or verbnet.find_verbnet() is None,
                               reason="needs WordNet and VerbNet on disk")


class Papers(Plugin):
    """Something for the agent to have done, so there is a real turn to explain."""

    def __init__(self) -> None:
        super().__init__(name="papers", kinds={"file": ("path",), "report": ("file",)})
        self.fs = {"/h/report.txt", "/h/notes.txt"}

    def capabilities(self):
        return (Capability("delete", (Param("path", "path"),),
                           effects=(Effect("has_location", {"undergoer": "path"}, negated=True),)),)

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

    def display(self, ref):
        return ref.id.rsplit("/", 1)[-1] if str(getattr(ref, "id", "")).startswith("path:") else None

    def execute(self, act, *, key):
        self.fs -= {v.id[5:] for _, v in act.args}
        return Receipt(act, "applied", idempotency_key=key)

    def holds(self, cap, args):
        return args["path"].id[5:] not in self.fs


def talking_agent(*plugins):
    """An agent that can talk about itself. The plugin reaches the agent through a callable,
    because the agent is built from its plugins and the plugin needs the agent."""
    box: list[Agent] = []
    plugin = DiscoursePlugin(lambda: box[0] if box else None)
    box.append(Agent([*plugins, plugin]))
    return box[0], plugin


def events(turn, kind):
    return [e for e in turn.events if e.get("type") == kind]


# ------------------------------------------------------- the gap this module closes


def test_without_an_informing_capability_the_goal_is_right_and_nothing_can_serve_it():
    """The diagnosis, pinned: the language layer was never the problem.

    An agent with no discourse plugin still reads "explain your reasoning" completely and
    still derives the correct end state from VerbNet. It declines for the one honest reason —
    nothing it can do brings that state about.
    """
    agent = Agent([Papers()])
    turn = agent.turn("explain your reasoning")
    [goal] = events(turn, "goal")
    assert "has_information" in goal["goal"]
    [outcome] = turn.outcomes
    assert outcome.status == "declined"
    assert "has_information" in outcome.reason


def test_the_capability_is_what_makes_the_request_reachable():
    agent, _ = talking_agent(Papers())
    agent.turn("delete report.txt")
    [outcome] = agent.turn("explain your reasoning").outcomes
    assert outcome.status == "done"
    assert outcome.plan[:2] == ("discourse", "explain_what_i_did")


# ------------------------------------------------------- explaining a real turn


def test_explaining_a_turn_names_the_capability_that_was_actually_invoked():
    """Act, then ask. The answer has to contain what was run, not that something was run."""
    agent, _ = talking_agent(Papers())
    acted = agent.turn("delete report.txt")
    [act] = events(acted, "act")
    assert (act["plugin"], act["capability"]) == ("papers", "delete")

    asked = agent.turn("what is your reasoning?")
    [outcome] = asked.outcomes
    assert outcome.status == "answered"
    assert f"invoked={act['plugin']}.{act['capability']}" in asked.reply
    assert f"{act['capability']}/verified=True" in asked.reply


def test_the_explanation_is_read_off_the_spans_the_runtime_opened():
    """Every line of the report is a field of something recorded, and the span it came from
    is still there to check it against. This is the difference between introspection and a
    story about introspection."""
    agent, plugin = talking_agent(Papers())
    agent.turn("delete report.txt")
    agent.turn("what is your reasoning?")

    report = plugin._said[("explain_what_i_did", Ref("entity:reasoning"))]
    start, end = plugin._last_range          # the acting turn, which is the one explained
    spans = agent.runtime.trace.spans[start:end]
    answered = {f"{s.op}={s.answered_by}" for s in spans if s.answered_by}
    assert answered, "the turn opened no spans, so there would be nothing to introspect"
    assert answered <= set(report)
    assert any(line.startswith("parse=reader:") for line in report)
    assert "verify" in {s.op for s in spans}


def test_it_explains_the_finished_turn_and_not_the_one_doing_the_explaining():
    """``Trace.spans`` has no turn boundaries in it, so reporting the whole list would mix in
    every earlier message and reporting the current one would explain the explaining."""
    agent, plugin = talking_agent(Papers())
    agent.turn("delete report.txt")
    agent.turn("delete notes.txt")
    report = plugin.report("explain_what_i_did", Ref("entity:reasoning"))
    assert any("notes.txt" in line for line in report)
    assert not any("report.txt" in line for line in report)


# ------------------------------------------------------- what it can do


def test_what_it_can_do_is_the_registry_and_nothing_else():
    agent, _ = talking_agent(Papers())
    turn = agent.turn("what are your capabilities?")
    [outcome] = turn.outcomes
    assert outcome.status == "answered"
    registered = describe_capabilities(agent.plugins)
    assert registered
    for entry in registered:
        assert f"{entry['plugin']}.{entry['name']}" in turn.reply
    assert "papers.delete" in turn.reply


def test_a_capability_added_later_shows_up_in_the_answer():
    """The answer is the plugin list read now, so it changes when the list does."""
    agent, _ = talking_agent()
    before = agent.turn("what are your capabilities?").reply
    assert "papers.delete" not in before
    agent.plugins.insert(0, Papers())
    assert "papers.delete" in agent.turn("what are your capabilities?").reply


def test_telling_and_asking_reach_the_same_capability_by_different_routes():
    """"tell me your capabilities" fills VerbNet's Recipient as well as its Topic, and
    ``core._achieves`` rejects a plan whose effect does not mention every filled role."""
    agent, _ = talking_agent(Papers())
    [outcome] = agent.turn("tell me your capabilities").outcomes
    assert outcome.status == "done"
    assert outcome.plan[:2] == ("discourse", "say_what_i_can_do")


# ------------------------------------------------------- what it knows


def test_it_says_what_it_knows_about_something_from_the_store():
    agent, _ = talking_agent(Papers())
    agent.turn("Austin is in Texas.")
    turn = agent.turn("what is Austin?")
    [outcome] = turn.outcomes
    assert outcome.status == "answered"
    assert outcome.plan[:2] == ("discourse", "say_what_i_know_about")
    held = [r.proposition for r in agent.store.propositions() if "entity:Austin" in r.proposition.describe()]
    assert held and any(p.describe() in turn.reply for p in held)


def test_what_it_said_back_is_not_something_it_knows():
    """Answering reveals the report as claims into the same store; quoting those back as
    knowledge would let the agent's own answers accumulate into evidence."""
    agent, plugin = talking_agent(Papers())
    agent.turn("Austin is in Texas.")
    first = agent.turn("what is Austin?").reply
    second = plugin.report("say_what_i_know_about", Ref("entity:Austin"))
    assert len(second) == 1
    assert second[0] in first


def test_a_subject_the_store_is_silent_about_gets_no_answer():
    agent, _ = talking_agent(Papers())
    [outcome] = agent.turn("what is Chicago?").outcomes
    assert outcome.status == "unknown"


# ------------------------------------------------------- abstaining


def test_with_no_finished_turn_it_declines_instead_of_explaining_the_request():
    """The first message cannot be explained: there is nothing before it. What the agent must
    not do is report the parse of the sentence that asked."""
    agent, plugin = talking_agent(Papers())
    turn = agent.turn("explain your reasoning")
    [outcome] = turn.outcomes
    assert outcome.status == "declined"
    assert "nothing I have on record" in outcome.reason
    assert plugin.report("explain_what_i_did", Ref("entity:reasoning")) == ()
    assert plugin._said == {}


def test_asked_with_no_finished_turn_it_says_it_does_not_know():
    agent, _ = talking_agent(Papers())
    [outcome] = agent.turn("what is your reasoning?").outcomes
    assert outcome.status == "unknown"


def test_a_self_report_is_only_about_the_addressees_own():
    """WordNet files a meeting as an event, which is what the trace report is about; only the
    grammar's possessor says whose event it is."""
    agent, plugin = talking_agent(Papers())
    agent.turn("delete report.txt")
    turn = agent.turn("explain the meeting")
    [outcome] = turn.outcomes
    assert outcome.status == "declined"
    assert "reports on what is mine" in outcome.reason
    assert events(turn, "act") == []
    assert plugin._said == {}


def test_a_refusal_names_which_report_refused_and_why():
    agent, plugin = talking_agent(Papers())
    agent.turn("delete report.txt")
    got = plugin.refer(_description("reasoning"), Param("ability", "entity"), context={})
    assert isinstance(got, Unknown)
    assert "ability" in got.detail


def test_with_no_agent_to_reach_it_abstains_rather_than_raising():
    plugin = DiscoursePlugin()
    assert list(plugin.perceive()) == []
    for report in REPORTS:
        got = plugin.refer(_description("reasoning"), Param(report.param, "entity"), context={})
        assert isinstance(got, Unknown) and got.reason == "no_agent"


def _description(noun: str):
    from tensorcode.language import Entity

    return Entity("description", noun, {"noun": noun, "possessive": True, "possessor": 2})
