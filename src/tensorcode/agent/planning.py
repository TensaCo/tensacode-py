"""Bounded planning over plugin-provided, grounded action models.

This is a finite classical search with open-world initial evidence. It does not
invent task refinements or groundings, call executors, or treat model predictions
as observations. A returned plan still needs checks during and after execution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..actions import Plan, Step
from ..goals import Condition, GoalSpec
from ..outcomes import Unknown
from .plugin import Call, Capability, Plugin


@dataclass(frozen=True)
class _Action:
    call: Call
    preconditions: tuple[tuple[int, bool], ...]
    effects: tuple[tuple[int, bool], ...]


def plan_goal(
    goal: GoalSpec,
    plugins: Iterable[Plugin],
    *,
    max_depth: int = 12,
    max_states: int = 10_000,
    max_actions: int = 1_000,
    capability_models: Mapping[str, Sequence[Capability]] | None = None,
) -> Plan | Unknown:
    """Find a shortest plan within a finite grounding and explicit search budget.

    Plugins must enumerate all candidate calls they want considered, including
    prerequisite actions. Their effects use exact predicate/role/value equality;
    there is no implicit exclusivity, role subsumption, or closed-world default.
    Unmentioned atoms retain their value in the declared action model. Unknown
    preconditions cannot enable an action. Multiple observers that disagree leave
    an atom unknown. Bounds count action candidates and unique explored states.

    Groundings and observations are snapshots, not execution authorization.
    Failure reports are relative to these models and bounds, never a proof that
    the real-world task is impossible. An empty plan means observations already
    support every desired condition. An executor can supply ``capability_models``
    to search a detached snapshot and reject model changes before dispatch.
    """
    if max_depth < 0 or max_states < 1 or max_actions < 0:
        raise ValueError("planning requires nonnegative depth/actions and positive states")
    providers = tuple(plugins)
    by_name = {plugin.name: plugin for plugin in providers}
    if len(by_name) != len(providers):
        return Unknown("invalid_action_model", "plugin names must be unique")
    # An equality-based registry supports structured/unhashable role values and
    # never uses repr as identity (different references may print identically).
    atoms: list[Condition] = []

    def literal(condition: Condition) -> tuple[int, bool]:
        positive = Condition(condition.pred, dict(condition.args))
        try:
            index = atoms.index(positive)
        except ValueError:
            index = len(atoms)
            atoms.append(positive)
        return index, not condition.negated

    targets = tuple(literal(condition) for condition in goal.conditions)
    invariants = tuple(literal(condition) for condition in goal.invariants)
    if len(dict(targets)) != len(set(targets)):
        return Unknown("contradictory_goal", "the same grounded condition is required true and false")

    def observe(atom: Condition) -> bool | None | Unknown:
        evidence = set()
        for plugin in providers:
            observed = plugin.observe_condition(atom)
            if observed is True or observed is False:
                evidence.add(observed)
            elif not isinstance(observed, Unknown):
                return Unknown("invalid_observation", f"{plugin.name} returned neither bool nor Unknown")
        return next(iter(evidence)) if len(evidence) == 1 else None

    initial: list[bool | None] = []
    for atom in atoms:
        observed = observe(atom)
        if isinstance(observed, Unknown):
            return observed
        initial.append(observed)
    if not all(initial[index] is value for index, value in invariants):
        return Unknown("invariant_unestablished", "observations do not establish every held condition initially")
    if all(initial[index] is value for index, value in targets):
        return Plan((), "Current observations support every goal condition.")

    actions: list[_Action] = []
    count = 0
    for plugin in providers:
        declared = tuple(plugin.capabilities()) if capability_models is None else tuple(capability_models[plugin.name])
        capabilities = {cap.name: cap for cap in declared}
        if len(capabilities) != len(declared):
            return Unknown("invalid_action_model", f"{plugin.name} has duplicate capability names")
        for call in plugin.enumerate_actions(goal):
            count += 1
            if count > max_actions:
                return Unknown("planning_budget", f"candidate actions exceed max_actions={max_actions}")
            if not isinstance(call, Call) or call.plugin != plugin.name or call.capability not in capabilities:
                return Unknown("invalid_action_model", f"{plugin.name} supplied an unknown capability call")
            cap = capabilities[call.capability]
            args = dict(call.args)
            names = [param.name for param in cap.params]
            if len(args) != len(call.args) or len(set(names)) != len(names) or set(args) != set(names):
                return Unknown("invalid_action_model", f"{plugin.name}.{cap.name} has missing, duplicate, or extra arguments")
            try:
                preconditions = tuple(literal(Condition(c.pred, {role: args[param] for role, param in c.roles.items()}, c.negated)) for c in cap.preconditions)
                effects = tuple(literal(Condition(e.pred, {role: args[param] for role, param in e.roles.items()}, e.negated)) for e in cap.effects)
            except KeyError as exc:
                return Unknown("invalid_action_model", f"{plugin.name}.{cap.name} references unbound parameter {exc}")
            if len(dict(effects)) != len(set(effects)):
                return Unknown("invalid_action_model", f"{plugin.name}.{cap.name} has contradictory effects")
            actions.append(_Action(call, preconditions, effects))

    for atom in atoms[len(initial):]:
        observed = observe(atom)
        if isinstance(observed, Unknown):
            return observed
        initial.append(observed)
    start = tuple(initial)
    frontier = deque([(start, ())])
    visited = {start}
    depth_limited = False
    while frontier:
        state, path = frontier.popleft()
        if all(state[index] is value for index, value in targets):
            steps = tuple(Step(f"step-{i + 1}", actions[action].call, (f"step-{i}",) if i else ()) for i, action in enumerate(path))
            rationale = (
                "Current observations support every goal condition."
                if not path else
                "A bounded search predicts these declared action effects satisfy the goal; "
                "preconditions and outcomes require fresh observation during execution."
            )
            return Plan(steps, rationale)
        for action_index, action in enumerate(actions):
            if not all(state[index] is value for index, value in action.preconditions):
                continue
            successor = list(state)
            for index, value in action.effects:
                successor[index] = value
            next_state = tuple(successor)
            if not all(next_state[index] is value for index, value in invariants):
                continue
            if next_state in visited:
                continue
            if len(path) >= max_depth:
                depth_limited = True
                continue
            if len(visited) >= max_states:
                return Unknown("planning_budget", f"search reached max_states={max_states}")
            visited.add(next_state)
            frontier.append((next_state, path + (action_index,)))
    if depth_limited:
        return Unknown("planning_budget", f"search reached max_depth={max_depth}")
    return Unknown("no_plan", "no plan in the supplied grounded action models and observed evidence")
