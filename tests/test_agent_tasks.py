"""Task identity changes execution behavior, not just the shape of a log."""

import json

import pytest

from tensorcode.agent import Agent, Capability, Condition, Effect, GoalSpec, Informs, Param, Plugin, Precondition
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import Act, Sentence
from tensorcode.language import Frame, Question, Request, verbnet
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Ref


class Devices(Plugin):
    def __init__(self, *, allowed=True, sticks=True, observable=True):
        super().__init__("devices")
        self.allowed = allowed
        self.sticks = sticks
        self.observable = observable
        self.enabled = set()
        self.calls = []
        self.checks = []

    def capabilities(self):
        return (Capability("enable", (Param("device", "device"),),
                           effects=(Effect("enabled", {"undergoer": "device"}),),
                           preconditions=(Precondition("available", {"undergoer": "device"}),)),)

    def refer(self, description, param, *, context):
        return description if isinstance(description, Ref) else Unknown("not_a_device")

    def precondition_holds(self, condition, args):
        self.checks.append(args["device"])
        return self.allowed

    def execute(self, act, *, key):
        self.calls.append(act.arg("device"))
        if self.sticks:
            self.enabled.add(act.arg("device"))
        return Receipt(act, "applied", idempotency_key=key)

    def holds(self, cap, args):
        return args["device"] in self.enabled if self.observable else Unknown("not_visible")


def desired(name="a"):
    return GoalSpec((Condition("enabled", {"undergoer": Ref(f"device:{name}")}),), label="prepare device")


def test_structured_goal_executes_without_lexical_interpretation(monkeypatch):
    plugin = Devices()
    agent = Agent([plugin])
    monkeypatch.setattr(verbnet, "goal_of", lambda *args: pytest.fail("structured task consulted VerbNet"))
    events = []
    result = agent.pursue(desired(), events=events)
    task = agent.tasks.get(result.task_id)
    assert result.status == task.status == "done"
    assert task.goal == desired()
    assert task.attempts[0].plan[1] == "enable"
    assert task.attempts[0].verified is True
    assert plugin.calls == [Ref("device:a")]
    json.dumps(events)


def test_completion_does_not_license_replaying_a_task():
    plugin = Devices()
    agent = Agent([plugin])
    result = agent.pursue(desired())
    with pytest.raises(ValueError, match="completed task"):
        agent.pursue(task_id=result.task_id)
    assert len(plugin.calls) == 1


def test_explicit_revision_keeps_identity_and_old_plan_but_uses_new_target():
    plugin = Devices()
    agent = Agent([plugin])
    first = agent.pursue(desired("a"))
    agent.tasks.revise(first.task_id, desired("b"), reason="use the second device instead")
    second = agent.pursue(task_id=first.task_id)
    task = agent.tasks.get(first.task_id)
    assert first.task_id == second.task_id
    assert task.revision == 2
    assert [a.revision for a in task.attempts] == [1, 2]
    assert task.revisions[0].goal == desired("a")
    assert task.revisions[1].goal == desired("b")
    assert plugin.calls == [Ref("device:a"), Ref("device:b")]


@pytest.mark.parametrize("allowed,status", [(False, "fails"), (Unknown("offline"), "unknown")])
def test_unestablished_precondition_blocks_dispatch_and_can_be_rechecked(allowed, status):
    plugin = Devices(allowed=allowed)
    agent = Agent([plugin])
    events = []
    first = agent.pursue(desired(), events=events)
    assert first.receipt.status == "rejected"
    assert plugin.calls == []
    assert next(e for e in events if e["type"] == "precondition")["status"] == status
    plugin.allowed = True
    second = agent.pursue(task_id=first.task_id)
    assert second.status == "done"
    assert len(plugin.checks) == 2
    assert len(agent.tasks.get(first.task_id).attempts) == 2


@pytest.mark.parametrize("sticks,observable,status", [(False, True, "failed"), (True, False, "unverified")])
def test_receipt_does_not_prove_completion_or_license_blind_retry(sticks, observable, status):
    plugin = Devices(sticks=sticks, observable=observable)
    agent = Agent([plugin])
    result = agent.pursue(desired())
    assert result.status == status
    assert agent.tasks.get(result.task_id).status == status
    assert result.receipt.status == "applied"
    with pytest.raises(ValueError, match="may have changed"):
        agent.pursue(task_id=result.task_id)
    assert len(plugin.calls) == 1


def test_one_action_cannot_claim_two_incompatible_target_bindings():
    plugin = Devices()
    agent = Agent([plugin])
    result = agent.pursue(GoalSpec(desired("a").conditions + desired("b").conditions))
    assert result.status == "declined"
    assert plugin.calls == []


def test_nullary_condition_in_explicit_conjunction_cannot_be_ignored():
    plugin = Devices()
    agent = Agent([plugin])
    result = agent.pursue(GoalSpec(desired().conditions + (Condition("safe", {}),)))
    assert result.status == "declined"
    assert plugin.calls == []


def test_unbound_explicit_condition_is_not_a_wildcard_for_success():
    with pytest.raises(ValueError, match="must be bound"):
        GoalSpec((Condition("enabled", {"undergoer": None}),))


def test_turn_request_links_to_ledger_and_survives_a_later_turn(monkeypatch):
    # Control interpretation to exercise the real turn path without lexical data.
    from tensorcode.agent import core

    frame = Frame("enable", {"object": Ref("device:a")})
    act = Act("request", Request(frame), frame)
    sentence = Sentence("enable a", ("enable", "a"), None, (act,))
    monkeypatch.setattr(core.ops, "parse", lambda *args, **kwargs: Transcript((sentence,), "fixture"))
    monkeypatch.setattr(verbnet, "goal_of", lambda *args: desired())
    agent = Agent([Devices()])
    first = agent.turn("enable a")
    task_id = first.outcomes[0].task_id
    monkeypatch.setattr(core.ops, "parse", lambda *args, **kwargs: Transcript())
    agent.turn("unrelated")
    assert len(agent.tasks) == 1
    assert agent.tasks.get(task_id).status == "done"
    assert next(e for e in first.events if e["type"] == "task")["task_id"] == task_id


def test_empty_goal_is_not_vacuous_success():
    with pytest.raises(ValueError, match="at least one"):
        GoalSpec(())


def test_rejected_information_action_neither_runs_nor_reveals_claims():
    class Inspection(Devices):
        def capabilities(self):
            return (Capability("inspect", (Param("device", "device"),),
                               informs=(Informs("has_location", "goal", "device"),),
                               effect_kind="read",
                               preconditions=(Precondition("available", {"undergoer": "device"}),)),)

        def reveal(self, *args):
            pytest.fail("rejected capability was asked to reveal claims")

    plugin = Inspection(allowed=Unknown("offline"))
    agent = Agent([plugin])
    question = Question(Frame("be", {"location": Ref("device:a")}), "object")
    outcome = agent._look(question, "has_location", Act("question", question, question.frame), [])
    assert outcome.status == "unknown"
    assert outcome.receipt.status == "rejected"
    assert plugin.calls == []


def test_unbound_precondition_never_reaches_plugin_checker_or_executor():
    plugin = Devices()
    agent = Agent([plugin])
    receipt = agent._invoke(plugin, plugin.capabilities()[0], {}, [])
    assert receipt.status == "rejected"
    assert "device" in receipt.error
    assert plugin.calls == plugin.checks == []
