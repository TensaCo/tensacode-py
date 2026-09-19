"""Finite contingent planning over empirically supported transition outcomes.

The caller supplies state semantics, goal, actions, and resource bounds. A policy
covers every retained empirical successor, not every possible future outcome.
Planning neither observes the world nor executes actions or installs effects.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from ..learning.experience import _same
from ..learning.empirical_dynamics import DynamicsPrediction, _state
from ..outcomes import Unknown
from .plugin import Call


@dataclass(frozen=True)
class ExaminedEdge:
    state: Any
    call: Call
    prediction: DynamicsPrediction | Unknown


@dataclass(frozen=True)
class StatePlan:
    state: Any
    rank: int | None
    calls: tuple[Call, ...]


@dataclass(frozen=True)
class EmpiricalPlan:
    model_id: str
    provider: str
    current_state: Any
    goal_state: Any
    reason: str
    depth: int | None
    first_calls: tuple[Call, ...]
    selected_call: Call | None
    nodes: tuple[StatePlan, ...]
    edges: tuple[ExaminedEdge, ...]
    max_depth: int
    max_states: int
    max_edges: int
    state_count: int
    prediction_count: int
    depth_limited: bool
    budget_exhausted: bool
    policy: str = "authored: shortest worst-case empirical outcome depth; retain all equal first choices"

    @property
    def supported(self) -> bool:
        return self.depth is not None and not self.budget_exhausted


def plan(model, current_state: Any, goal_state: Any, calls: Iterable[Call], *,
         max_depth: int = 12, max_states: int = 10000, max_edges: int = 100000) -> EmpiricalPlan:
    """Return a bounded AND-OR policy without choosing among equally short calls.

    Forward expansion considers only reachable projected states, stopping at the
    supplied goal. Backward layers admit a call only when every empirical outcome
    reaches the goal in fewer steps. A cycle cannot justify itself; unobserved or
    unvalidated edges remain Unknown. Hard graph budgets suppress action choices
    rather than silently omit an unexamined competitor. Depth failure is relative
    to these observations and bounds, not proof of real-world impossibility.
    """
    for name, value, minimum in (("max_depth", max_depth, 0), ("max_states", max_states, 1),
                                  ("max_edges", max_edges, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    current_state, goal_state = (_state(value) for value in deepcopy((current_state, goal_state)))
    supplied_calls = tuple(deepcopy(tuple(calls)))
    for index, call in enumerate(supplied_calls):
        if not isinstance(call, Call):
            raise TypeError("planning requires explicitly supplied Call values")
        if any(_same(call, other) for other in supplied_calls[:index]):
            raise ValueError("duplicate planning call")
    model_id, provider = model.id, model.provider
    states, distances = [current_state], [0]
    edges: list[ExaminedEdge] = []
    successors: dict[int, tuple[int, ...]] = {}
    edge_origins: list[int] = []
    exhausted = False
    depth_limited = False
    cursor = 0
    while cursor < len(states) and not exhausted:
        state, distance = states[cursor], distances[cursor]
        if _same(state, goal_state):
            cursor += 1
            continue
        if distance >= max_depth:
            depth_limited = True
            cursor += 1
            continue
        for call in supplied_calls:
            if len(edges) >= max_edges:
                exhausted = True
                break
            try:
                prediction = model.predict(deepcopy(state), deepcopy(call))
            except Exception as error:
                prediction = Unknown("dynamics_prediction_error", f"{type(error).__name__}: {error}")
            if not isinstance(prediction, (DynamicsPrediction, Unknown)):
                prediction = Unknown("invalid_dynamics_prediction", "expected DynamicsPrediction or Unknown")
            if isinstance(prediction, DynamicsPrediction) and (
                    prediction.model_id != model_id or not _same(prediction.state, state)
                    or not _same(prediction.call, call) or not prediction.outcomes):
                prediction = Unknown("invalid_dynamics_prediction", "prediction linkage or empirical outcomes invalid")
            edge_index = len(edges)
            edges.append(ExaminedEdge(deepcopy(state), deepcopy(call), deepcopy(prediction)))
            edge_origins.append(cursor)
            if isinstance(prediction, DynamicsPrediction):
                targets = []
                for outcome in prediction.outcomes:
                    index = next((i for i, seen in enumerate(states) if _same(seen, outcome.state)), None)
                    if index is None:
                        if len(states) >= max_states:
                            exhausted = True
                            break
                        index = len(states)
                        states.append(deepcopy(outcome.state))
                        distances.append(distance + 1)
                    if index not in targets:
                        targets.append(index)
                if not exhausted:
                    successors[edge_index] = tuple(targets)
            if exhausted:
                break
        cursor += 1
    ranks = [0 if _same(state, goal_state) else None for state in states]
    choices: list[tuple[Call, ...]] = [() for _ in states]
    by_origin = [[] for _ in states]
    for index, origin in enumerate(edge_origins):
        by_origin[origin].append(index)
    # Synchronous layers ensure a self-loop or mutually dependent cycle cannot
    # manufacture a finite plan. All tied calls are evaluated in the same layer.
    for depth in range(1, max_depth + 1):
        additions = []
        for index in range(len(states)):
            if ranks[index] is not None:
                continue
            supported_calls = []
            for edge_index in by_origin[index]:
                targets = successors.get(edge_index)
                if targets and all(ranks[target] is not None and ranks[target] < depth for target in targets):
                    supported_calls.append(deepcopy(edges[edge_index].call))
            if supported_calls:
                additions.append((index, tuple(supported_calls)))
        if not additions:
            break
        for index, supported_calls in additions:
            ranks[index] = depth
            choices[index] = supported_calls
    first_calls = choices[0] if ranks[0] is not None and not exhausted else ()
    depth = ranks[0] if not exhausted else None
    reason = ("graph_budget_exhausted" if exhausted else
              "supplied_state_matches_goal" if depth == 0 else
              "supported_contingent_policy" if depth is not None and len(first_calls) == 1 else
              "first_action_choice_required" if depth is not None else "no_supported_policy_within_bounds")
    result = EmpiricalPlan(model_id, provider, current_state, goal_state, reason, depth, first_calls,
        first_calls[0] if len(first_calls) == 1 else None,
        tuple(StatePlan(deepcopy(state), rank, tuple(deepcopy(calls)))
              for state, rank, calls in zip(states, ranks, choices)), tuple(edges),
        max_depth, max_states, max_edges, len(states), len(edges), depth_limited, exhausted)
    return deepcopy(result)
