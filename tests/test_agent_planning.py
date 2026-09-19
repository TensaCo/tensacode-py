"""Search tests use arbitrary predicates; no lexical or domain rules participate."""

import pytest

from tensorcode.actions import Plan, plan_order, RunnablePlan
from tensorcode.agent.planning import plan_goal
from tensorcode.agent.plugin import Call, Capability, Effect, Param, Plugin, Precondition
from tensorcode.goals import Condition, GoalSpec
from tensorcode.outcomes import Unknown


class World(Plugin):
    def __init__(self, caps=(), facts=(), calls=None):
        super().__init__("world")
        self.caps = tuple(caps)
        self.facts = tuple(facts)
        self.calls = calls
        self.executions = 0

    def capabilities(self):
        return self.caps

    def enumerate_actions(self, goal):
        if self.calls is not None:
            return self.calls
        return (Call(self.name, cap.name, (("x", "object"),)) for cap in self.caps)

    def observe_condition(self, condition):
        for fact in self.facts:
            if fact.pred == condition.pred and fact.args == condition.args:
                return fact.negated == condition.negated
        return Unknown("unobserved")


def cond(pred, negated=False, **args):
    return Condition(pred, args or {"Thing": "object"}, negated)


def cap(name, effects, pre=()):
    return Capability(name, (Param("x", "anything"),),
                      tuple(Effect(pred, {"Thing": "x"}, neg) for pred, neg in effects),
                      preconditions=tuple(Precondition(pred, {"Thing": "x"}, neg) for pred, neg in pre))


def names(plan):
    assert isinstance(plan, Plan), plan
    assert isinstance(plan_order(plan), RunnablePlan)
    return [step.action.capability for step in plan.steps]


@pytest.mark.parametrize("a,b,c", [("ready", "prepared", "finished"), ("nu7", "eta3", "xi9")])
def test_prerequisite_chain_is_predicate_independent(a, b, c):
    world = World([cap("second", [(c, False)], [(b, False)]),
                   cap("first", [(b, False)], [(a, False)])], [cond(a)])
    plan = plan_goal(GoalSpec((cond(c),)), [world])
    assert names(plan) == ["first", "second"]
    assert "predicts" in plan.rationale
    assert world.executions == 0


def test_delete_interference_requires_repair_and_conjunction():
    world = World([cap("finish", [("b", False), ("a", True)], [("a", False)]),
                   cap("prepare", [("a", False)])])
    assert names(plan_goal(GoalSpec((cond("a"), cond("b"))), [world])) == ["prepare", "finish", "prepare"]


def test_alternative_action_avoids_dead_end():
    world = World([cap("blocked", [("goal", False)], [("unknown", False)]),
                   cap("available", [("goal", False)])])
    assert names(plan_goal(GoalSpec((cond("goal"),)), [world])) == ["available"]


@pytest.mark.parametrize("negated", [False, True])
def test_unknown_never_satisfies_a_precondition(negated):
    world = World([cap("blocked", [("goal", False)], [("missing", negated)])])
    outcome = plan_goal(GoalSpec((cond("goal"),)), [world])
    assert isinstance(outcome, Unknown)
    assert outcome.reason == "no_plan"


def test_observed_false_enables_negative_precondition():
    world = World([cap("run", [("goal", False)], [("absent", True)])], [cond("absent", True)])
    assert names(plan_goal(GoalSpec((cond("goal"),)), [world])) == ["run"]


def test_contradictory_goal_and_effect_model_fail_explicitly():
    assert plan_goal(GoalSpec((cond("a"), cond("a", True))), []).reason == "contradictory_goal"
    world = World([cap("bad", [("a", False), ("a", True)])])
    assert plan_goal(GoalSpec((cond("a"),)), [world]).reason == "invalid_action_model"


def test_cycles_terminate_without_claiming_real_world_impossibility():
    world = World([cap("on", [("a", False)], [("a", True)]),
                   cap("off", [("a", True)], [("a", False)])], [cond("a")])
    outcome = plan_goal(GoalSpec((cond("unreachable"),)), [world])
    assert outcome.reason == "no_plan"
    assert "supplied" in outcome.detail


@pytest.mark.parametrize("budget", [{"max_depth": 0}, {"max_states": 1}, {"max_actions": 0}])
def test_budgets_return_unknown(budget):
    world = World([cap("run", [("goal", False)])])
    assert plan_goal(GoalSpec((cond("goal"),)), [world], **budget).reason == "planning_budget"


def test_invariants_block_temporary_violation_even_if_repaired():
    world = World([cap("finish", [("goal", False), ("safe", True)]),
                   cap("repair", [("safe", False)])], [cond("safe")])
    goal = GoalSpec((cond("goal"),), invariants=(cond("safe"),))
    assert plan_goal(goal, [world]).reason == "no_plan"
    assert plan_goal(goal, [World(world.caps)]).reason == "invariant_unestablished"


def test_exact_role_sets_and_structured_values():
    world = World([cap("partial", [("relation", True)])])
    goal = GoalSpec((cond("relation", True, Thing="object", Other=[1, 2]),))
    assert plan_goal(goal, [world]).reason == "no_plan"
    world = World(facts=[goal.conditions[0]])
    assert names(plan_goal(goal, [world])) == []


def test_conflicting_observers_do_not_supply_evidence():
    world = World(facts=[cond("goal")])
    other = World(facts=[cond("goal", True)])
    other.name = "other"
    assert plan_goal(GoalSpec((cond("goal"),)), [world, other]).reason == "no_plan"


def test_bad_argument_bindings_fail_before_search():
    world = World([cap("run", [("goal", False)])], calls=[Call("world", "run", ())])
    assert plan_goal(GoalSpec((cond("goal"),)), [world]).reason == "invalid_action_model"


def test_satisfied_goal_needs_no_candidate_generation():
    world = World(facts=[cond("goal")], calls=iter([None] * 10))
    assert names(plan_goal(GoalSpec((cond("goal"),)), [world], max_actions=0)) == []


def test_search_selects_alternative_preserving_invariant():
    world = World([cap("damage", [("goal", False), ("safe", True)]),
                   cap("preserve", [("goal", False)])], [cond("safe")])
    goal = GoalSpec((cond("goal"),), invariants=(cond("safe"),))
    assert names(plan_goal(goal, [world])) == ["preserve"]


def test_infinite_groundings_are_bounded():
    def calls():
        while True:
            yield Call("world", "run", (("x", "object"),))
    world = World([cap("run", [("goal", False)])], calls=calls())
    assert plan_goal(GoalSpec((cond("goal"),)), [world], max_actions=3).reason == "planning_budget"


def test_predicted_negative_effect_enables_following_action():
    world = World([cap("remove", [("a", True)]),
                   cap("finish", [("goal", False)], [("a", True)])])
    assert names(plan_goal(GoalSpec((cond("goal"),)), [world])) == ["remove", "finish"]
