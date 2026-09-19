"""Execution must retain the model and mutation history that justified a plan."""
from dataclasses import replace

import pytest

from tensorcode.agent import Agent, Capability, Condition, Effect, GoalSpec, Param, Plugin, Precondition
from tensorcode.agent.plugin import Call
from tensorcode.outcomes import Receipt


class PartialExecution(Plugin):
    def __init__(self):
        super().__init__("partial", planning_enabled=True)
        self.prepared = False
        self.finished = False
        self.reject_finish = True
        self.calls = []

    def capabilities(self):
        return (
            Capability("prepare", (), effects=(Effect("prepared", {}),)),
            Capability("finish", (), effects=(Effect("finished", {}),),
                       preconditions=(Precondition("prepared", {}),)),
        )

    def enumerate_actions(self, goal):
        return [Call(self.name, name, ()) for name in ("prepare", "finish")]

    def observe_condition(self, condition):
        observed = self.prepared if condition.pred == "prepared" else self.finished
        return not observed if condition.negated else observed

    def execute(self, act, *, key):
        self.calls.append(act.capability)
        if act.capability == "prepare":
            self.prepared = True
        elif self.reject_finish:
            return Receipt(act, "rejected", idempotency_key=key)
        else:
            self.finished = True
        return Receipt(act, "applied", idempotency_key=key)


def test_rejected_final_step_does_not_hide_an_earlier_mutation():
    plugin = PartialExecution()
    agent = Agent([plugin])
    goal = GoalSpec((Condition("finished", {}),))
    first = agent.pursue(goal)
    assert first.status == "failed"
    assert [step.receipt.status for step in first.steps] == ["applied", "rejected"]
    plugin.reject_finish = False
    with pytest.raises(ValueError, match="revise"):
        agent.pursue(task_id=first.task_id)
    assert plugin.calls == ["prepare", "finish"]

    agent.tasks.revise(first.task_id, goal, reason="authorize continuation after partial execution")
    resumed = agent.pursue(task_id=first.task_id)
    assert resumed.status == "done"
    assert plugin.calls == ["prepare", "finish", "finish"]
    assert len(agent.tasks.get(first.task_id).attempts) == 2


class DriftingModel(Plugin):
    def __init__(self, drift):
        super().__init__("drifting", planning_enabled=True)
        self.drift = drift
        self.enumerated = False
        self.safe = True
        self.done = False
        self.calls = []
        self.model = Capability("set", (Param("x", "object"),),
                                effects=(Effect("done", {"target": "x"}),))

    def capabilities(self):
        if not self.enumerated or self.drift == "nested":
            return (self.model,)
        if self.drift == "effects":
            return (replace(self.model, effects=(*self.model.effects, Effect("safe", {}, True))),)
        return (replace(self.model, params=(*self.model.params, Param("new", "object"))),)

    def enumerate_actions(self, goal):
        self.enumerated = True
        if self.drift == "nested":
            self.model.effects[0].roles["unexpected"] = "x"
        return [Call(self.name, "set", (("x", "item"),))]

    def observe_condition(self, condition):
        observed = self.safe if condition.pred == "safe" else self.done
        return not observed if condition.negated else observed

    def execute(self, act, *, key):
        self.calls.append(act)
        self.safe = False
        self.done = True
        return Receipt(act, "applied", idempotency_key=key)


@pytest.mark.parametrize("drift", ["effects", "params", "nested"])
def test_changed_model_is_rejected_before_dispatch(drift):
    plugin = DriftingModel(drift)
    result = Agent([plugin]).pursue(GoalSpec(
        (Condition("done", {"target": "item"}),), invariants=(Condition("safe", {}),)))
    assert result.status == "failed"
    assert "model changed" in result.reason
    assert plugin.calls == []
    assert plugin.safe is True
    assert result.receipt is None


class LegacyModel(Plugin):
    def __init__(self, effects):
        super().__init__("legacy")
        self.effects = effects
        self.calls = []

    def capabilities(self):
        return (Capability("set", (), effects=self.effects),)

    def execute(self, act, *, key):
        self.calls.append(act)
        return Receipt(act, "applied", idempotency_key=key)

    def holds(self, cap, args):
        return True


def test_legacy_path_rejects_contradictory_explicit_goal():
    plugin = LegacyModel((Effect("p", {}),))
    result = Agent([plugin]).pursue(GoalSpec((Condition("p", {}), Condition("p", {}, True))))
    assert result.status == "declined"
    assert "contradictory" in result.reason
    assert plugin.calls == []


def test_legacy_path_rejects_contradictory_effect_model():
    plugin = LegacyModel((Effect("p", {}), Effect("p", {}, True)))
    result = Agent([plugin]).pursue(GoalSpec((Condition("p", {}),)))
    assert result.status == "declined"
    assert result.receipt is None
    assert plugin.calls == []
