"""Supplied goals carry exact roles and grounded values through all executors."""
from dataclasses import dataclass

import pytest

from tensorcode.agent.core import Agent
from tensorcode.agent.plugin import Call, Capability, Effect, Param, Plugin
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Entity, Frame
from tensorcode.language import verbnet
from tensorcode.outcomes import Receipt, Unknown
from tensorcode.records import Ref


class ExactPlugin(Plugin):
    def __init__(self, effects, *, modeled=False, params=None):
        super().__init__("exact", planning_enabled=modeled)
        self.effects = effects
        self.params = tuple(params or dict.fromkeys(parameter for effect in effects for parameter in effect.roles.values()))
        self.calls = []
        self.seen_goal = None

    def capabilities(self):
        return (Capability("set", tuple(Param(name, "arbitrary-type") for name in self.params), effects=self.effects),)

    def refer(self, *args, **kwargs):
        raise AssertionError("a grounded goal cannot ask a plugin to guess its identity")

    def execute(self, call, *, key):
        self.calls.append(call)
        return Receipt(call, "applied", idempotency_key=key)

    def holds(self, capability, args):
        return True

    def enumerate_actions(self, goal):
        self.seen_goal = goal
        effect = self.effects[0]
        condition = goal.conditions[0]
        yield Call(self.name, "set", tuple((parameter, condition.args[role]) for role, parameter in effect.roles.items()))

    def observe_condition(self, condition):
        return bool(self.calls)


def forbid_lexical_interpretation(agent, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("supplied goal names are not lexical roles or noun kinds")
    monkeypatch.setattr(agent, "kinds", fail)
    monkeypatch.setattr(verbnet, "role_class", fail)


@pytest.mark.parametrize("modeled", [False, True])
def test_bound_entities_normalize_identically_before_either_execution_path(modeled, monkeypatch):
    identity = Ref("opaque:target")
    plugin = ExactPlugin((Effect("custom:ready", {"OwnerSlot": "target"}),), modeled=modeled)
    agent = Agent([plugin])
    forbid_lexical_interpretation(agent, monkeypatch)
    goal = GoalSpec((Condition("custom:ready", {"OwnerSlot": Entity("description", "misleading other target", ref=identity)}),))
    assert goal.conditions[0].args["OwnerSlot"] is identity
    outcome = agent.pursue(goal)
    assert outcome.status == "done"
    assert plugin.calls[0].arg("target") == identity
    if modeled:
        assert plugin.seen_goal.conditions[0].args["OwnerSlot"] == identity


def test_supplied_role_names_that_lexical_aliases_collapse_remain_distinct(monkeypatch):
    roles = {"Theme": "first", "Patient": "second", "theme": "third"}
    plugin = ExactPlugin((Effect("relation", roles),))
    agent = Agent([plugin])
    forbid_lexical_interpretation(agent, monkeypatch)
    goal = GoalSpec((Condition("relation", {"Theme": Ref("world:a"), "Patient": Ref("world:b"), "theme": Ref("world:c")}),))
    outcome = agent.pursue(goal)
    assert outcome.status == "done"
    assert dict(plugin.calls[0].args) == {"first": Ref("world:a"), "second": Ref("world:b"), "third": Ref("world:c")}
    assert agent._achieves(plugin.capabilities()[0], goal.conditions[0])


def test_supplied_role_and_predicate_spelling_never_matches_a_lexical_alias(monkeypatch):
    plugin = ExactPlugin((Effect("Ready", {"undergoer": "target"}),))
    agent = Agent([plugin])
    forbid_lexical_interpretation(agent, monkeypatch)
    for predicate, role in [("ready", "undergoer"), ("Ready", "Theme")]:
        outcome = agent.pursue(GoalSpec((Condition(predicate, {role: Ref("world:a")}),)))
        assert outcome.status == "declined"
    assert not plugin.calls


@pytest.mark.parametrize("value", [False, 0, "", "addressee", Entity("literal", ""), Entity("number", "zero", {"value": 0})])
def test_explicit_false_zero_and_empty_literals_are_bound_values(value):
    plugin = ExactPlugin((Effect("value", {"slot": "input"}),))
    agent = Agent([plugin])
    goal = GoalSpec((Condition("value", {"slot": value}),))
    outcome = agent.pursue(goal)
    assert outcome.status == "done"
    expected = 0 if isinstance(value, Entity) and value.kind == "number" else "" if isinstance(value, Entity) else value
    assert plugin.calls[0].arg("input") == expected
    assert type(plugin.calls[0].arg("input")) is type(expected)


@dataclass(frozen=True)
class TypedPayload:
    content: object


@pytest.mark.parametrize("value", [Entity("description", "folder"), {Entity("name", "key"): "value"}, [Entity("name", "folder")], {"nested": (Entity("pronoun", "it"),)}, TypedPayload(Entity("path", "/tmp/a")), Frame("nested", {"object": Entity("description", "folder")})])
def test_ungrounded_entities_are_rejected_even_inside_typed_nested_values(value):
    with pytest.raises(ValueError, match="explicit grounding required"):
        GoalSpec((Condition("ready", {"target": value}),))


def test_nested_explicit_bindings_normalize_without_changing_domain_types():
    value = TypedPayload({"collection": [Entity("description", "folder", ref=Ref("world:a")), Entity("literal", "text")]})
    goal = GoalSpec((Condition("ready", {"target": value}),))
    normalized = goal.conditions[0].args["target"]
    assert isinstance(normalized, TypedPayload)
    assert normalized.content == {"collection": [Ref("world:a"), "text"]}
    unchanged = TypedPayload((Ref("world:a"), False))
    goal = GoalSpec((Condition("ready", {"target": unchanged}),))
    assert goal.conditions[0].args["target"] is unchanged


def test_unbound_invariants_and_none_roles_are_rejected_at_same_boundary():
    with pytest.raises(ValueError, match="invariants"):
        GoalSpec((Condition("ready", {}),), invariants=(Condition("held", {"target": Entity("name", "thing")}),))
    with pytest.raises(ValueError, match="must be bound"):
        GoalSpec((Condition("ready", {"target": None}),))
    with pytest.raises(ValueError, match="unknown goal value"):
        GoalSpec((Condition("ready", {"target": Unknown("unsupplied")}),))


def test_lexical_request_description_does_not_reach_plugin_reference_guessing():
    plugin = ExactPlugin((Effect("ready", {"undergoer": "target"}),))
    agent = Agent([plugin])
    # Lexical adapter knows Theme->undergoer but does not know what a folder denotes.
    lexical = verbnet.Goal("prepare", "authored-test", (Condition("ready", {"Theme": Entity("description", "folder")}),), Frame("prepare"))
    options, reasons = agent.plans(lexical)
    assert not options and "explicit grounding required" in reasons[0]
    assert not plugin.calls


@pytest.mark.parametrize("reverse", [False, True])
def test_competing_complete_grounded_actions_require_choice_before_invocation(reverse):
    class Alternatives(ExactPlugin):
        def capabilities(self):
            names = ("first", "second") if not reverse else ("second", "first")
            return tuple(Capability(name, (Param("target", "entity"),), effects=self.effects) for name in names)
    plugin = Alternatives((Effect("ready", {"slot": "target"}),))
    agent = Agent([plugin])
    outcome = agent.pursue(GoalSpec((Condition("ready", {"slot": Ref("world:a")}),)))
    assert outcome.status == "declined"
    assert outcome.plan.reason == "ambiguous_capability"
    assert "exact.first" in outcome.reason and "exact.second" in outcome.reason
    assert not plugin.calls


def test_lexical_adapter_rejects_colliding_roles_instead_of_overwriting_values():
    plugin = ExactPlugin((Effect("ready", {"undergoer": "target"}),))
    agent = Agent([plugin])
    lexical = verbnet.Goal("prepare", "authored-test", (
        Condition("ready", {"Theme": Ref("world:a"), "Patient": Ref("world:b")}),), Frame("prepare"))
    options, reasons = agent.plans(lexical)
    assert not options
    assert "incompatible values for undergoer" in reasons[0]
    assert not plugin.calls
