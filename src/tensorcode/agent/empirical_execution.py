"""Fresh-observation execution of one step from a contingent empirical plan.

The state projection, target, exploration, and allowed actions are supplied.
Recorded transitions determine graph edges. Hypothetical reachability is never
an observed goal, and subsequent steps require a new plan from fresh evidence.
"""
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any
from uuid import uuid4

from ..learning.empirical_dynamics import EmpiricalDynamics, DynamicsPrediction
from ..learning.experience import extract_transitions, _same
from ..outcomes import Receipt, Unknown
from .empirical_planning import plan
from .experience_planning import _provider, _capability
from .plugin import Call


@dataclass(frozen=True)
class EmpiricalProposal:
    id: str
    source_id: str
    plan: Any
    record_source_id: str = ""


@dataclass(frozen=True)
class EmpiricalExecution:
    id: str
    proposal_id: str
    receipt: Receipt | None
    observed_state: Any
    verification: bool | Unknown
    source_ids: tuple[str, ...]
    reason: str
    record_source_id: str = ""


@dataclass
class _Retained:
    proposal: EmpiricalProposal
    model: EmpiricalDynamics
    projection: Any
    provider: Any
    original: Any
    calls: tuple
    capabilities: tuple
    consumed: bool = False
    lock: Any = field(default_factory=Lock)


def _registry(agent):
    if not hasattr(agent, "_empirical_plans"):
        agent._empirical_plans = {}
    return agent._empirical_plans


def _validate(agent, model):
    rows = extract_transitions(agent.interpretations.sources(), provider=model.provider).transitions
    valid = model.validate_transitions(rows)
    if valid is not True:
        raise ValueError(f"empirical model disagrees with retained execution evidence: {valid}")


def propose(agent, model, observation_source_id, calls, goal_state, **bounds):
    if not isinstance(model, EmpiricalDynamics):
        raise TypeError("require an EmpiricalDynamics model fitted from transitions")
    provider = _provider(agent, model.provider)
    source = agent.interpretations.get_source(observation_source_id)
    observations = [s for s in agent.interpretations.sources()
                    if s.provider == model.provider and s.modality == "observation"]
    if (not observations or observations[-1].id != source.id
            or source.metadata.get("status") != "observed"):
        raise ValueError("planning requires the latest successful retained provider observation")
    calls = deepcopy(tuple(calls))
    caps = tuple(deepcopy(_capability(provider, c)) for c in calls)
    _validate(agent, model)
    state = model.projection.state(deepcopy(source.payload))
    planned = plan(model, state, goal_state, calls, **bounds)
    _validate(agent, model)
    proposal = EmpiricalProposal("empirical-plan:" + uuid4().hex, source.id, planned)
    record = agent.interpretations.add_source("Contingent reachability through empirical transitions",
        modality="hypothesis", provider="empirical-planning", payload=deepcopy(proposal),
        metadata={"observation_source_id": source.id})
    proposal = replace(proposal, record_source_id=record.id)
    _registry(agent)[proposal.id] = _Retained(deepcopy(proposal), model, model.projection,
        provider, deepcopy(source), calls, caps)
    return deepcopy(proposal)


def execute(agent, proposal_id, *, call: Call | None = None):
    retained = _registry(agent)[proposal_id]
    proposal, model = retained.proposal, retained.model
    chosen = deepcopy(call if call is not None else proposal.plan.selected_call)
    if chosen is not None and not any(_same(chosen, c) for c in proposal.plan.first_calls):
        raise ValueError("explicit action must be a retained best first step")
    events = []

    def finish(receipt, state, verification, reason):
        ids = tuple(e['source_id'] for e in events if e.get('type') == 'observation')
        result = EmpiricalExecution("empirical-execution:" + uuid4().hex, proposal_id,
            receipt, deepcopy(state), verification, ids, reason)
        source = agent.interpretations.add_source("Observed contingent-plan step",
            modality="assessment", provider="empirical-planning", payload=deepcopy(result),
            metadata={"proposal_id": proposal_id, "status": reason})
        return deepcopy(replace(result, record_source_id=source.id))

    with retained.lock:
        consumed = retained.consumed
        retained.consumed = True
    if consumed:
        return finish(None, Unknown("not_observed"), Unknown("proposal_already_consumed"), "proposal_already_consumed")
    if chosen is None:
        reason = proposal.plan.reason
        return finish(None, Unknown("not_observed"), Unknown(reason), reason)
    index = next(i for i, c in enumerate(retained.calls) if _same(c, chosen))
    predicted = next(e.prediction for e in proposal.plan.edges
                     if _same(e.state, proposal.plan.current_state) and _same(e.call, chosen))
    if not isinstance(predicted, DynamicsPrediction):
        raise ValueError("chosen action lacks a retained empirical prediction")

    def contract():
        _validate(agent, model)
        if model.projection is not retained.projection or model.id != proposal.plan.model_id:
            return Unknown("stale_empirical_model")
        current = _provider(agent, model.provider)
        if (current is not retained.provider or
                not _same(_capability(current, chosen), retained.capabilities[index])):
            return Unknown("capability_model_changed")
        if not _same(model.predict(proposal.plan.current_state, chosen), predicted):
            return Unknown("stale_empirical_prediction")
        # Capability callbacks can unmount or replace their own provider. Check
        # after callback-bearing validation, before authorizing the real action.
        if _provider(agent, model.provider) is not retained.provider:
            return Unknown("capability_model_changed")
        return True

    def guard(before_ids):
        try:
            valid = contract()
            if valid is not True:
                return valid
            extra = agent._capture_observations(events, stage="prediction_guard",
                providers=(retained.provider,), action=chosen, retain_unavailable=True)
            sources = [agent.interpretations.get_source(i) for i in (*before_ids, *extra)]
            sources = [s for s in sources if s.provider == model.provider]
            if not sources or any(s.metadata.get("status") != "observed" for s in sources):
                return Unknown("fresh_observation_unavailable")
            if any(not _same(s.payload, retained.original.payload) for s in sources):
                return Unknown("stale_empirical_observation")
            return contract()
        except Exception as exc:
            return Unknown("invalid_empirical_binding", f"{type(exc).__name__}: {exc}")

    try:
        current = _provider(agent, model.provider)
        capability = _capability(current, chosen)
    except ValueError:
        return finish(None, Unknown("not_observed"), Unknown("invalid_empirical_binding"), "invalid_empirical_binding")
    receipt = agent._invoke(current, capability, dict(chosen.args), events, before_dispatch=guard)
    if receipt.status != "applied":
        return finish(receipt, Unknown("not_observed"), Unknown("action_not_applied"), "action_not_applied")
    after = [agent.interpretations.get_source(e['source_id']) for e in events
             if e.get('type') == 'observation' and e.get('stage') == 'after_action'
             and e.get('provider') == model.provider]
    actual = Unknown("after_observation_unavailable")
    if len(after) == 1 and after[0].metadata.get('status') == 'observed':
        try:
            actual = retained.projection.state(deepcopy(after[0].payload))
        except Exception as exc:
            actual = Unknown("state_unprojectable", f"{type(exc).__name__}: {exc}")
    if isinstance(actual, Unknown):
        return finish(receipt, actual, actual, "unresolved_observation")
    supported = any(_same(actual, outcome.state) for outcome in predicted.outcomes)
    if not supported:
        return finish(receipt, actual, Unknown("unmodeled_outcome"), "unmodeled_outcome")
    try:
        valid = contract()
    except Exception as exc:
        valid = Unknown("invalid_empirical_binding", f"{type(exc).__name__}: {exc}")
    if valid is not True:
        return finish(receipt, actual, valid, "stale_model_after_execution")
    reached = _same(actual, proposal.plan.goal_state)
    return finish(receipt, actual, reached, "goal_observed" if reached else "step_observed_replan_required")
