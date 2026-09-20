"""Resumable task attempts that replan from actual empirical observations.

Goals, action candidates, and optional tie choices are supplied. Task identity and
receipts persist in the in-memory ledger; neither interpretation nor execution is
replayed when a bounded attempt resumes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4
from typing import Any

from ..learning.empirical_dynamics import EmpiricalDynamics, _state
from ..learning.experience import _same
from ..outcomes import Receipt, Unknown
from .empirical_execution import propose, execute, _validate
from .experience_planning import _provider
from .plugin import Call
from .tasks import StepAttempt
from .task_dependencies import validate_dependencies
from .understand import Act


@dataclass(frozen=True)
class EmpiricalGoal:
    state: Any
    calls: tuple[Call, ...]
    model_id: str
    provider: str

    def __post_init__(self):
        _state(self.state)
        if not isinstance(self.calls, tuple) or any(not isinstance(call, Call) for call in self.calls):
            raise TypeError("empirical goals require an explicit tuple of Calls")
        if any(_same(call, other) for i, call in enumerate(self.calls) for other in self.calls[:i]):
            raise ValueError("empirical goal calls must be unique")
        if any(not isinstance(value, str) or not value.strip() for value in (self.model_id, self.provider)):
            raise ValueError("empirical goals require model and provider identities")


@dataclass(frozen=True)
class EmpiricalTaskTrace:
    model_id: str
    provider: str
    proposal_ids: tuple[str, ...]
    execution_ids: tuple[str, ...]
    observation_source_ids: tuple[str, ...]
    choice_policy: str
    resume_safe: bool


def _validated_goal(task, model):
    if task.status == "done":
        raise ValueError("completed task requires an explicit revision before another attempt")
    if not isinstance(task.goal, EmpiricalGoal):
        raise TypeError("task does not contain an EmpiricalGoal")
    goal = deepcopy(task.goal)
    goal.__post_init__()
    if goal.model_id != model.id or goal.provider != model.provider:
        raise ValueError("task goal identifies a different empirical model or provider; revise explicitly")
    for previous in task.attempts:
        if previous.revision != task.revision:
            continue
        receipts = [r for r in (previous.receipt, *(step.receipt for step in previous.steps)) if r is not None]
        may_have_acted = any(r.status != "rejected" for r in receipts)
        resumable = (previous.status == "suspended" and isinstance(previous.plan, EmpiricalTaskTrace)
                     and previous.plan.resume_safe)
        if may_have_acted and not resumable:
            raise ValueError("previous attempt may have changed the world; revise explicitly before retrying")
    return goal


def pursue(agent, model, goal: EmpiricalGoal | None = None, *, task_id: str | None = None,
           source: str = "empirical", max_steps: int = 1, choose=None, dependencies=(), **bounds):
    """Attempt or resume one task revision, with at most max_steps dispatches.

    ``choose`` is an authored callback receiving a detached EmpiricalPlan; it may
    select only one retained best first call. Without it an equal-cost tie defers.
    Explicit interpretation dependencies are captured for a task revision, never
    inferred from its goal. Their withdrawal blocks dispatch and completion.
    A fresh observation and new plan precede every step. Only controlled budget
    suspension after supported observations licenses resuming an applied attempt.
    """
    from .core import Outcome

    if (goal is None) == (task_id is None):
        raise ValueError("supply either an EmpiricalGoal or a task_id")
    if not isinstance(model, EmpiricalDynamics):
        raise TypeError("empirical tasks require a fitted EmpiricalDynamics model")
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if choose is not None and not callable(choose):
        raise TypeError("choose must be an explicit callable")
    if goal is not None and not isinstance(goal, EmpiricalGoal):
        raise TypeError("empirical tasks require an EmpiricalGoal")
    dependencies = tuple(dependencies)
    if task_id is not None and dependencies:
        raise ValueError("task dependencies can change only through an explicit revision")
    task = (agent.tasks.create(source, goal, dependencies=dependencies) if goal is not None
            else agent.tasks.get(task_id))
    goal = _validated_goal(task, model)
    if not hasattr(agent, "_empirical_task_locks"):
        agent._empirical_task_locks = {}
    lock = agent._empirical_task_locks.setdefault(task.id, Lock())
    if not lock.acquire(blocking=False):
        raise ValueError("empirical task already has an active attempt")
    try:
        task = agent.tasks.get(task.id)
        goal = _validated_goal(task, model)
        revision = task.revision
        act = Act("request", deepcopy(goal), None)
        proposals, executions, source_ids, steps = [], [], [], []
        last_receipt = None

        def unchanged():
            current = agent.tasks.get(task.id)
            if current.revision != revision or not _same(current.goal, goal):
                return Unknown("task_revision_changed")
            validity = validate_dependencies(agent.interpretations, task.dependencies)
            if validity is not True:
                return validity
            # Finish with a callback-free scalar read: copying goal payloads here
            # could itself withdraw an interpretation after dependency validation.
            if agent.tasks.current_revision(task.id) != revision:
                return Unknown("task_revision_changed")
            return True

        def finish(status, verified, reason, *, resume_safe=False):
            # Even an observation/choice callback can revise the task. The ledger
            # records this attempt under its captured revision without changing the new one.
            validity = unchanged()
            if validity is not True:
                status, verified, reason, resume_safe = "unknown", validity, validity.reason, False
            trace = EmpiricalTaskTrace(model.id, model.provider, tuple(proposals), tuple(executions),
                tuple(source_ids), "authored callback among retained best calls" if choose else
                "unique retained first call; equal alternatives defer", resume_safe)
            outcome = Outcome(act, status, goal=deepcopy(goal), plan=trace, receipt=deepcopy(last_receipt),
                verified=deepcopy(verified), reason=reason, task_id=task.id, steps=tuple(deepcopy(steps)))
            agent.tasks.record(task.id, outcome, revision=revision)
            return outcome

        dispatched = 0
        while True:
            validity = unchanged()
            if validity is not True:
                return finish("unknown", validity, validity.reason)
            events = []
            try:
                provider = _provider(agent, model.provider)
                ids = agent._capture_observations(events, stage="empirical_task", providers=(provider,),
                                                  retain_unavailable=True)
                source_ids.extend(ids)
                observed = [agent.interpretations.get_source(sid) for sid in ids]
                if len(observed) != 1 or observed[0].metadata.get("status") != "observed":
                    return finish("unknown", Unknown("fresh_observation_unavailable"), "fresh_observation_unavailable")
                state = _state(model.projection.state(deepcopy(observed[0].payload)))
                _validate(agent, model)
                if _provider(agent, model.provider) is not provider:
                    return finish("unknown", Unknown("provider_changed"), "provider_changed")
            except Exception as error:
                return finish("unknown", Unknown("empirical_observation_error", f"{type(error).__name__}: {error}"),
                              "empirical_observation_error")
            validity = unchanged()
            if validity is not True:
                return finish("unknown", validity, validity.reason)
            if _same(state, goal.state):
                return finish("done", True, "goal_observed")
            if dispatched >= max_steps:
                return finish("suspended", False, "step_budget_exhausted", resume_safe=True)
            try:
                proposal = propose(agent, model, observed[0].id, goal.calls, goal.state, **bounds)
                proposals.append(proposal.id)
            except Exception as error:
                return finish("unknown", Unknown("empirical_planning_error", f"{type(error).__name__}: {error}"),
                              "empirical_planning_error")
            validity = unchanged()
            if validity is not True:
                return finish("unknown", validity, validity.reason)
            selected = proposal.plan.selected_call
            if proposal.plan.first_calls and choose is not None:
                try:
                    selected = choose(deepcopy(proposal.plan))
                    valid_choice = selected is None or any(_same(selected, call) for call in proposal.plan.first_calls)
                except Exception as error:
                    return finish("unknown", Unknown("empirical_choice_error", f"{type(error).__name__}: {error}"),
                                  "empirical_choice_error")
                if not valid_choice:
                    return finish("unknown", Unknown("invalid_empirical_choice"), "invalid_empirical_choice")
            validity = unchanged()
            if validity is not True:
                return finish("unknown", validity, validity.reason)
            if selected is None:
                return finish("unknown", Unknown(proposal.plan.reason), proposal.plan.reason)
            try:
                result = execute(agent, proposal.id, call=deepcopy(selected), execution_guard=unchanged)
            except Exception as error:
                # Crossing the execution boundary can change the world before an
                # exception hides its result. This records uncertainty, not a
                # fabricated applied receipt, and prevents silent task replay.
                uncertainty = Unknown("empirical_execution_error", f"{type(error).__name__}: {error}")
                last_receipt = Receipt(deepcopy(selected), "indeterminate", error=uncertainty.detail)
                steps.append(StepAttempt("empirical-exception:" + uuid4().hex,
                                         deepcopy(selected), last_receipt, uncertainty))
                return finish("unverified", uncertainty, "empirical_execution_error")
            executions.append(result.id)
            source_ids.extend(result.source_ids)
            last_receipt = result.receipt
            if result.receipt is not None:
                step_verified = (True if result.receipt.status == "applied" and
                    result.reason in ("goal_observed", "step_observed_replan_required") and
                    type(result.verification) is bool else deepcopy(result.verification))
                steps.append(StepAttempt(result.id, deepcopy(selected), deepcopy(result.receipt), step_verified))
            dispatched += 1
            validity = unchanged()
            if validity is not True:
                return finish("unknown", validity, validity.reason)
            if result.receipt is None or result.receipt.status != "applied":
                return finish("unknown", result.verification, result.reason)
            if result.verification is True and result.reason == "goal_observed":
                return finish("done", True, "goal_observed")
            if result.verification is not False or result.reason != "step_observed_replan_required":
                return finish("unverified", result.verification, result.reason)
            # An observed supported intermediate state is not failure. The next
            # loop captures fresh evidence before replanning or bounded suspension.
    finally:
        lock.release()
