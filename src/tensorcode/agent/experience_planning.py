"""One-step hypothetical choices from sample-validated transition models.

Actions, goal labels, observation projections, and validation policy are supplied.
Only the transition predictions are induced. No prediction enters the world store
or a capability's effects, and observation failure never certifies a goal.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import uuid4
from threading import Lock

from ..learning.experience import LearnedTransitionModel, TransitionPrediction, extract_transitions, _same
from ..outcomes import Receipt, Unknown
from .plugin import Call


@dataclass(frozen=True)
class ActionPrediction:
    call: Call
    prediction: TransitionPrediction | Unknown


@dataclass(frozen=True)
class ExperienceProposal:
    id: str
    source_id: str
    model_id: str
    model_revision: int
    desired_outcome: Any
    candidates: tuple[ActionPrediction, ...]
    selected_call: Call | None
    reason: str
    record_source_id: str = ""


@dataclass(frozen=True)
class ExperienceExecution:
    id: str
    proposal_id: str
    receipt: Receipt | None
    verification: bool | Unknown
    source_ids: tuple[str, ...]
    reason: str
    record_source_id: str = ""


@dataclass
class _Retained:
    proposal: ExperienceProposal
    model: LearnedTransitionModel
    projection: Any
    capabilities: tuple[Any, ...]
    executed: bool = False
    lock: Any = field(default_factory=Lock)


def _registry(agent):
    if not hasattr(agent, "_experience_plans"):
        agent._experience_plans = {}
    return agent._experience_plans


def _provider(agent, name):
    matches = [plugin for plugin in agent.plugins if f"plugin:{plugin.name}" == name]
    if len(matches) != 1:
        raise ValueError("model provider must identify exactly one mounted plugin")
    return matches[0]


def _capability(plugin, call):
    if not isinstance(call, Call) or call.plugin != plugin.name:
        raise ValueError("candidate call must belong to the model's mounted provider")
    caps = [cap for cap in plugin.capabilities() if cap.name == call.capability]
    if len(caps) != 1:
        raise ValueError("candidate call must identify one declared capability")
    names = [param.name for param in caps[0].params]
    args = dict(call.args)
    if len(args) != len(call.args) or len(set(names)) != len(names) or set(args) != set(names):
        raise ValueError("candidate arguments must exactly match declared parameters")
    return caps[0]


def _validate_evidence(agent, model, prediction):
    batch = extract_transitions(agent.interpretations.sources(), provider=model.provider)
    by_attempt = {row.attempt_id: row for row in batch.transitions}
    evidence = prediction.evidence
    training = tuple(evidence.training_attempt_ids)
    evaluation = tuple(evidence.evaluation_attempt_ids)
    ids = training + evaluation
    if (not training or not evaluation or len(set(ids)) != len(ids)
            or any(attempt not in by_attempt for attempt in ids)):
        raise ValueError("model sample evidence requires nonempty disjoint unique retained transition splits")
    if (evidence.rule_index != prediction.rule_index or evidence.action_family != prediction.action_family
            or not evidence.verified or not model.is_current(prediction)):
        raise ValueError("model rule evidence disagrees with current prediction")
    examples = {example.attempt_id: example for example in model.examples}
    membership = {"training": set(), "evaluation": set()}
    replayed = {}
    for attempt, example in examples.items():
        if attempt not in by_attempt or example.split not in membership:
            raise ValueError("model fit examples require retained transitions and explicit splits")
        row = by_attempt[attempt]
        if model.validate_transition(row) is not True:
            raise ValueError("model sample content disagrees with retained transition evidence")
        current = model.predict(deepcopy(row.before), deepcopy(row.action))
        replayed[attempt] = current
        if (isinstance(current, TransitionPrediction) and current.rule_id == prediction.rule_id
                and current.action_family == prediction.action_family):
            if not model.is_current(current):
                raise ValueError("model changed during support membership validation")
            membership[example.split].add(attempt)
    if membership["training"] != set(training) or membership["evaluation"] != set(evaluation):
        raise ValueError("model rule support membership disagrees with retained fit examples")
    correct = {"training": 0, "evaluation": 0}
    for split, attempts in (("training", training), ("evaluation", evaluation)):
        for attempt in attempts:
            row = by_attempt[attempt]
            validation = model.validate_transition(row)
            if validation is not True:
                raise ValueError("model sample content disagrees with retained transition evidence")
            if attempt not in examples or examples[attempt].split != split:
                raise ValueError("model support disagrees with retained fit split")
            # Validate the rule currently being used, not merely the historical
            # projected sample. A changed label can leave fit samples untouched.
            current = replayed[attempt]
            if (not isinstance(current, TransitionPrediction) or not model.is_current(current)
                    or current.rule_id != prediction.rule_id or current.rule_index != prediction.rule_index
                    or current.model_id != prediction.model_id or current.model_revision != prediction.model_revision
                    or current.action_family != prediction.action_family
                    or not _same(current.outcome, prediction.outcome)
                    or current.evidence != evidence):
                raise ValueError("model rule no longer covers its cited support")
            observed = model.projection.outcome(deepcopy(row.after))
            if isinstance(observed, Unknown):
                raise ValueError("model support outcome is unresolved")
            correct[split] += int(_same(observed, prediction.outcome))
    if (type(evidence.training_correct) is not int or type(evidence.evaluation_correct) is not int
            or correct["training"] != evidence.training_correct
            or correct["evaluation"] != evidence.evaluation_correct):
        raise ValueError("model rule accuracy disagrees with retained observed outcomes")
    policy = model.policy
    if (len(training) < policy.min_training_support or len(evaluation) < policy.min_evaluation_support
            or correct["training"] / len(training) < policy.min_accuracy
            or correct["evaluation"] / len(evaluation) < policy.min_accuracy):
        raise ValueError("model rule observed support fails validation policy")
    expected = {source for attempt in ids for source in by_attempt[attempt].source_ids}
    if len(evidence.source_ids) != len(set(evidence.source_ids)) or set(evidence.source_ids) != expected:
        raise ValueError("model source evidence disagrees with retained transition identities")
    if not model.is_current(prediction):
        raise ValueError("model changed during evidence validation")



def propose(agent, model: LearnedTransitionModel, observation_source_id: str,
            calls: Iterable[Call], desired_outcome: Any) -> ExperienceProposal:
    """Compare supplied actions without executing or installing their predictions."""
    if not isinstance(model, LearnedTransitionModel):
        raise TypeError("require a sample-validated LearnedTransitionModel")
    plugin = _provider(agent, model.provider)
    source = agent.interpretations.get_source(observation_source_id)
    observations = [s for s in agent.interpretations.sources()
                    if s.provider == model.provider and s.modality == "observation"]
    if (source.provider != model.provider or source.modality != "observation"
            or source.metadata.get("status") != "observed" or observations[-1].id != source.id):
        raise ValueError("prediction requires the latest successful retained provider observation")
    if isinstance(desired_outcome, Unknown):
        raise ValueError("an unknown desired outcome cannot define a goal")
    candidates, caps, selected = [], [], []
    calls = tuple(deepcopy(tuple(calls)))
    for i, call in enumerate(calls):
        if any(_same(call, previous) for previous in calls[:i]):
            raise ValueError("duplicate candidate call")
        caps.append(deepcopy(_capability(plugin, call)))
        try:
            prediction = model.predict(deepcopy(source.payload), deepcopy(call))
        except Exception as exc:
            prediction = Unknown("transition_prediction_error", f"{type(exc).__name__}: {exc}")
        if not isinstance(prediction, (TransitionPrediction, Unknown)):
            raise TypeError("transition model returned neither a prediction nor Unknown")
        if isinstance(prediction, TransitionPrediction):
            if not model.is_current(prediction):
                raise ValueError("model returned a stale prediction")
            _validate_evidence(agent, model, prediction)
            if _same(prediction.outcome, desired_outcome):
                selected.append(call)
        candidates.append(ActionPrediction(call, prediction))
    unknown = any(isinstance(candidate.prediction, Unknown) for candidate in candidates)
    choice = selected[0] if len(selected) == 1 else None
    reason = ("sole_supported_target_prediction" if choice is not None else "unresolved_predictions" if unknown
              else "multiple_supported_targets" if len(selected) > 1 else "no_supported_target")
    record = ExperienceProposal("experience-plan:" + uuid4().hex, source.id, model.model_id,
                                model.revision, deepcopy(desired_outcome), tuple(candidates), choice, reason)
    audit = agent.interpretations.add_source(
        "Hypothetical one-step action comparison", modality="hypothesis", provider="experience-planning",
        payload=record, metadata={"status": "proposed", "observation_source_id": source.id})
    from dataclasses import replace
    record = replace(record, record_source_id=audit.id)
    _registry(agent)[record.id] = _Retained(deepcopy(record), model, model.projection, tuple(caps))
    return deepcopy(record)


def execute(agent, proposal_id: str) -> ExperienceExecution:
    """Execute a retained unique choice once and check independent after evidence."""
    retained = _registry(agent)[proposal_id]
    proposal, model = retained.proposal, retained.model
    events: list[dict] = []

    def finish(receipt, verification, reason):
        ids = tuple(event["source_id"] for event in events if event.get("type") == "observation")
        result = ExperienceExecution("experience-execution:" + uuid4().hex, proposal_id,
                                     receipt, verification, ids, reason)
        source = agent.interpretations.add_source(
            "Observed result of an experience-based action proposal", modality="assessment",
            provider="experience-planning", payload=result,
            metadata={"proposal_id": proposal_id, "status": reason})
        from dataclasses import replace
        return deepcopy(replace(result, record_source_id=source.id))

    with retained.lock:
        already_consumed = retained.executed
        retained.executed = True
    if already_consumed:
        return finish(None, Unknown("proposal_already_consumed"), "proposal_already_consumed")
    if proposal.selected_call is None:
        return finish(None, Unknown(proposal.reason), proposal.reason)
    call = proposal.selected_call
    index = next(i for i, candidate in enumerate(proposal.candidates) if _same(candidate.call, call))
    prediction = proposal.candidates[index].prediction
    original = agent.interpretations.get_source(proposal.source_id)

    def guard(before_ids):
        if (model.model_id != proposal.model_id or model.revision != proposal.model_revision
                or model.projection is not retained.projection or not model.is_current(prediction)):
            return Unknown("stale_transition_model")
        try:
            current = _provider(agent, model.provider)
            if not _same(_capability(current, call), retained.capabilities[index]):
                return Unknown("capability_model_changed")
            _validate_evidence(agent, model, prediction)
        except (ValueError, KeyError) as exc:
            return Unknown("invalid_model_binding", str(exc))
        ids = agent._capture_observations(events, stage="prediction_guard", providers=(current,),
                                          action=call, retain_unavailable=True)
        for source_id in (*before_ids, *ids):
            fresh = agent.interpretations.get_source(source_id)
            if fresh.provider != model.provider:
                continue
            if fresh.metadata.get("status") != "observed":
                return Unknown("fresh_observation_unavailable")
            if not _same(fresh.payload, original.payload):
                return Unknown("stale_transition_observation")
        # The final sensor callback can change a model or capability too. Check
        # the dispatch contract after all observation/projection callbacks.
        try:
            _validate_evidence(agent, model, prediction)
            if (model.model_id != proposal.model_id or model.revision != proposal.model_revision
                    or model.projection is not retained.projection or not model.is_current(prediction)):
                return Unknown("stale_transition_model")
            # Capability enumeration is a callback: check mounting after it too.
            if (_provider(agent, model.provider) is not current
                    or not _same(_capability(current, call), retained.capabilities[index])
                    or _provider(agent, model.provider) is not current):
                return Unknown("capability_model_changed")
        except (ValueError, KeyError) as exc:
            return Unknown("invalid_model_binding", str(exc))
        return True

    try:
        plugin = _provider(agent, model.provider)
        cap = _capability(plugin, call)
    except ValueError as exc:
        return finish(None, Unknown("invalid_model_binding", str(exc)), "invalid_model_binding")
    receipt = agent._invoke(plugin, cap, dict(deepcopy(call.args)), events, before_dispatch=guard)
    if receipt.status != "applied":
        return finish(receipt, Unknown("action_not_applied", receipt.error or ""), "action_not_applied")
    after = [agent.interpretations.get_source(event["source_id"]) for event in events
             if event.get("type") == "observation" and event.get("stage") == "after_action"
             and event.get("provider") == model.provider]
    if len(after) != 1 or after[0].metadata.get("status") != "observed":
        return finish(receipt, Unknown("after_observation_unavailable"), "after_observation_unavailable")
    try:
        actual = retained.projection.outcome(deepcopy(after[0].payload))
        if isinstance(actual, Unknown):
            return finish(receipt, actual, "outcome_unresolved")
        verified = _same(actual, prediction.outcome)
        model.observe_outcome(prediction, actual, source_ids=(after[0].id,),
                              reason="independent observation after executed proposed action")
    except Exception as exc:
        return finish(receipt, Unknown("outcome_unprojectable", f"{type(exc).__name__}: {exc}"),
                      "outcome_unprojectable")
    return finish(receipt, verified, "prediction_confirmed" if verified else "prediction_contradicted")
