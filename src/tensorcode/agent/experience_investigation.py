"""Compare supplied model-applicability hypotheses through empirical predictions.

Models learn transitions; callers supply their correspondence to interpretations,
measurement semantics, and available probes. Assessments never select a workspace
interpretation, install a belief, or suspend a rule in a different model context.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from numbers import Number
from threading import Lock
from typing import Any, Iterable
from uuid import uuid4

from ..learning.experience import LearnedTransitionModel, TransitionPrediction, extract_transitions, _same
from ..outcomes import Receipt, Unknown
from .experience_planning import _provider, _capability, _validate_evidence
from .plugin import Call


@dataclass(frozen=True)
class ModelApplicability:
    candidate_id: str
    model: LearnedTransitionModel
    basis: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmpiricalOutcome:
    """Observed alternatives and their sources, not calibrated future support."""
    outcome: Any
    training_attempt_ids: tuple[str, ...]
    evaluation_attempt_ids: tuple[str, ...]
    source_ids: tuple[str, ...]

    @property
    def training_count(self):
        return len(self.training_attempt_ids)

    @property
    def evaluation_count(self):
        return len(self.evaluation_attempt_ids)


@dataclass(frozen=True)
class CandidatePrediction:
    candidate_id: str
    prediction: TransitionPrediction | Unknown
    supported_outcomes: tuple[EmpiricalOutcome, ...] = ()


@dataclass(frozen=True)
class ProbePrediction:
    call: Call
    predictions: tuple[CandidatePrediction, ...]
    outcome_survivor_counts: tuple[int, ...]
    unknown_count: int
    score: tuple[int, int] | None


@dataclass(frozen=True)
class ApplicabilitySnapshot:
    candidate_id: str
    model_id: str
    model_revision: int
    basis: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExperienceInvestigationProposal:
    id: str
    group_id: str
    source_id: str
    group_revision: int
    continuation_generation: int
    applicability: tuple[ApplicabilitySnapshot, ...]
    probes: tuple[ProbePrediction, ...]
    eligible_calls: tuple[Call, ...]
    best_calls: tuple[Call, ...]
    selected_call: Call | None
    reason: str
    policy: str
    record_source_id: str = ""


@dataclass(frozen=True)
class CandidateAssessment:
    candidate_id: str
    prediction: TransitionPrediction | Unknown
    actual_outcome: Any
    status: str
    supported_outcomes: tuple[EmpiricalOutcome, ...] = ()

    @property
    def viable(self):
        return self.status != "contradicted"


@dataclass(frozen=True)
class ExperienceInvestigationResult:
    id: str
    proposal_id: str
    receipt: Receipt | None
    assessments: tuple[CandidateAssessment, ...]
    supported_candidate_id: str | None
    source_ids: tuple[str, ...]
    reason: str
    record_source_id: str = ""


@dataclass
class _Retained:
    proposal: ExperienceInvestigationProposal
    bindings: tuple[ModelApplicability, ...]
    projection: Any
    provider: Any
    capabilities: tuple[Any, ...]
    original: Any
    candidate_ids: tuple[str, ...]
    continuation: Any
    consumed: bool = False
    lock: Any = field(default_factory=Lock)


def _registry(agent):
    if not hasattr(agent, "_experience_investigations"):
        agent._experience_investigations = {}
    return agent._experience_investigations


def _finite(value):
    """Reject nonfinite numeric labels, including nested scalar-label tuples."""
    if isinstance(value, Number):
        try:
            return isfinite(value.real) and isfinite(value.imag) if isinstance(value, complex) else isfinite(value)
        except (TypeError, ValueError, OverflowError):
            return False
    if isinstance(value, (tuple, frozenset)):
        return all(_finite(item) for item in value)
    # Outcome fitting already requires hashability. Equality must also be reflexive.
    try:
        return _same(value, value)
    except Exception:
        return False


def _empirical_outcomes(model, prediction):
    # The caller first validates every current fit example against actual retained
    # transitions. These snapshots therefore reproduce those projected outcomes;
    # both fit and validation observations contribute explicitly marked support.
    examples = {example.attempt_id: example for example in model.examples}
    buckets = []
    for split, attempts in (("training", prediction.evidence.training_attempt_ids),
                            ("evaluation", prediction.evidence.evaluation_attempt_ids)):
        for attempt in attempts:
            example = examples[attempt]
            if not _finite(example.outcome):
                return None
            bucket = next((b for b in buckets if _same(b["outcome"], example.outcome)), None)
            if bucket is None:
                bucket = {"outcome": deepcopy(example.outcome), "training": [], "evaluation": [], "sources": []}
                buckets.append(bucket)
            bucket[split].append(attempt)
            bucket["sources"].extend(example.source_ids)
    return tuple(EmpiricalOutcome(b["outcome"], tuple(b["training"]), tuple(b["evaluation"]),
                                  tuple(b["sources"])) for b in buckets)


def _contains(outcomes, value):
    return any(_same(outcome.outcome, value) for outcome in outcomes)


def _unchanged(agent, retained):
    proposal = retained.proposal
    group = agent.interpretations.get(proposal.group_id)
    if (group.revision != proposal.group_revision
            or tuple(c.id for c in group.candidates if not c.rejected) != retained.candidate_ids
            or agent.interpretations.continuation_status(group.id) != retained.continuation):
        return Unknown("stale_interpretation_group")
    for binding, snapshot in zip(retained.bindings, proposal.applicability):
        model = binding.model
        if (model.model_id != snapshot.model_id or model.revision != snapshot.model_revision
                or model.projection is not retained.projection):
            return Unknown("stale_transition_model")
    return True


def propose(agent, group_id: str, bindings: Iterable[ModelApplicability],
            observation_source_id: str, calls: Iterable[Call]) -> ExperienceInvestigationProposal:
    """Predict the same supplied probes under every supplied candidate model.

    The authored policy minimizes the worst empirically supported survivor set
    (unknown models survive every result), then unknown predictions. Ties require an
    explicit call at execution. Labels are compared structurally, never converted
    into authored Condition lists or treated as calibrated probabilities.
    """
    group = agent.interpretations.get(group_id)
    candidate_ids = tuple(c.id for c in group.candidates if not c.rejected)
    continuation = agent.interpretations.continuation_status(group_id)
    bindings = tuple(bindings)
    if (not bindings or any(not isinstance(b, ModelApplicability) for b in bindings)
            or len({b.candidate_id for b in bindings}) != len(bindings)
            or {b.candidate_id for b in bindings} != set(candidate_ids)):
        raise ValueError("model applicability must cover every non-rejected candidate exactly once")
    by_id = {b.candidate_id: b for b in bindings}
    bindings = tuple(by_id[candidate_id] for candidate_id in candidate_ids)
    for binding in bindings:
        if not isinstance(binding.model, LearnedTransitionModel):
            raise TypeError("applicability requires a fitted LearnedTransitionModel")
        if (not isinstance(binding.basis, tuple) or not binding.basis
                or any(not isinstance(v, str) or not v.strip() for v in binding.basis)):
            raise ValueError("applicability requires explicit nonempty basis")
        if not isinstance(binding.evidence_ids, tuple):
            raise TypeError("applicability evidence IDs must be a tuple")
        for source_id in binding.evidence_ids:
            agent.interpretations.get_source(source_id)
    model = bindings[0].model
    projection = model.projection
    if any(b.model.provider != model.provider or b.model.projection is not projection for b in bindings):
        raise ValueError("candidate models require one provider and the same outcome Projection object")
    provider = _provider(agent, model.provider)
    source = agent.interpretations.get_source(observation_source_id)
    observations = [s for s in agent.interpretations.sources()
                    if s.provider == model.provider and s.modality == "observation"]
    if (source.provider != model.provider or source.modality != "observation"
            or source.metadata.get("status") != "observed" or not observations
            or observations[-1].id != source.id):
        raise ValueError("investigation requires latest successful retained provider observation")
    calls = tuple(deepcopy(tuple(calls)))
    probes, capabilities = [], []
    snapshots = tuple(ApplicabilitySnapshot(b.candidate_id, b.model.model_id, b.model.revision,
                                           b.basis, b.evidence_ids) for b in bindings)
    for index, call in enumerate(calls):
        if any(_same(call, previous) for previous in calls[:index]):
            raise ValueError("duplicate probe call")
        capabilities.append(deepcopy(_capability(provider, call)))
        predictions, observed_outcomes = [], []
        unknown = 0
        for binding in bindings:
            supported = ()
            try:
                prediction = binding.model.predict(deepcopy(source.payload), deepcopy(call))
            except Exception as error:
                prediction = Unknown("transition_prediction_error", f"{type(error).__name__}: {error}")
            if not isinstance(prediction, (TransitionPrediction, Unknown)):
                raise TypeError("model must return TransitionPrediction or Unknown")
            if isinstance(prediction, TransitionPrediction):
                if not binding.model.is_current(prediction):
                    raise ValueError("model returned stale prediction")
                _validate_evidence(agent, binding.model, prediction)
                supported = _empirical_outcomes(binding.model, prediction)
                if not _finite(prediction.outcome) or supported is None:
                    prediction = Unknown("nonfinite_predicted_outcome")
                    supported = ()
            if isinstance(prediction, Unknown):
                unknown += 1
            else:
                for support in supported:
                    if not any(_same(value, support.outcome) for value in observed_outcomes):
                        observed_outcomes.append(deepcopy(support.outcome))
            predictions.append(CandidatePrediction(binding.candidate_id, deepcopy(prediction), supported))
        known = [p for p in predictions if isinstance(p.prediction, TransitionPrediction)]
        different_sets = (len(known) >= 2 and any(
            _contains(candidate.supported_outcomes, value) != _contains(known[0].supported_outcomes, value)
            for candidate in known[1:] for value in observed_outcomes))
        survivors = tuple(sorted((unknown + sum(_contains(p.supported_outcomes, value) for p in known)
                                  for value in observed_outcomes), reverse=True))
        score = (max(survivors), unknown) if different_sets else None
        probes.append(ProbePrediction(call, tuple(predictions), survivors, unknown, score))
    eligible = tuple(probe.call for probe in probes if probe.score is not None)
    best_score = min((probe.score for probe in probes if probe.score is not None), default=None)
    best = tuple(probe.call for probe in probes if probe.score is not None and probe.score == best_score)
    selected = best[0] if len(best) == 1 else None
    reason = "unique_discriminating_probe" if selected else "probe_choice_required" if best else "no_supported_discriminating_probe"
    proposal = ExperienceInvestigationProposal("experience-investigation:" + uuid4().hex, group_id,
        source.id, group.revision, continuation.generation, snapshots, tuple(probes), eligible, best,
        selected, reason, "authored:minimize worst empirical outcome survivor set, then unknown count; no tie fallback")
    retained = _Retained(deepcopy(proposal), bindings, projection, provider, tuple(capabilities),
                         deepcopy(source), candidate_ids, continuation)
    if _unchanged(agent, retained) is not True:
        raise RuntimeError("investigation inputs changed during prediction")
    record = agent.interpretations.add_source("Empirical model-applicability probe predictions",
        modality="hypothesis", provider="experience-investigation", payload=deepcopy(proposal),
        metadata={"status": "proposed", "group_id": group_id, "observation_source_id": source.id})
    proposal = replace(proposal, record_source_id=record.id)
    retained.proposal = deepcopy(proposal)
    _registry(agent)[proposal.id] = retained
    return deepcopy(proposal)


def execute(agent, proposal_id: str, *, call: Call | None = None) -> ExperienceInvestigationResult:
    """Execute one retained discriminating probe and assess applicability only."""
    retained = _registry(agent)[proposal_id]
    proposal = retained.proposal
    chosen = deepcopy(call if call is not None else proposal.selected_call)
    if chosen is not None and not any(_same(chosen, candidate) for candidate in proposal.best_calls):
        raise ValueError("explicit probe must be one of the retained best calls")
    events = []
    probe = next((p for p in proposal.probes if _same(p.call, chosen)), None) if chosen is not None else None

    def finish(receipt, assessments, selected_id, reason):
        source_ids = tuple(event["source_id"] for event in events if event.get("type") == "observation")
        result = ExperienceInvestigationResult("experience-investigation-result:" + uuid4().hex,
            proposal_id, receipt, tuple(assessments), selected_id, source_ids, reason)
        record = agent.interpretations.add_source("Observed empirical model-applicability assessment",
            modality="assessment", provider="experience-investigation", payload=deepcopy(result),
            metadata={"proposal_id": proposal_id, "group_id": proposal.group_id, "status": reason})
        return deepcopy(replace(result, record_source_id=record.id))

    with retained.lock:
        consumed = retained.consumed
        retained.consumed = True
    if consumed:
        return finish(None, (), None, "proposal_already_consumed")
    if probe is None:
        return finish(None, (), None, proposal.reason)
    index = next(i for i, p in enumerate(proposal.probes) if p is probe)
    provider_name = retained.bindings[0].model.provider

    def guard(before_ids):
        unchanged = _unchanged(agent, retained)
        if unchanged is not True:
            return unchanged
        try:
            current = _provider(agent, provider_name)
            if current is not retained.provider or not _same(_capability(current, chosen), retained.capabilities[index]):
                return Unknown("capability_model_changed")
            for binding, predicted in zip(retained.bindings, probe.predictions):
                if isinstance(predicted.prediction, TransitionPrediction):
                    if not binding.model.is_current(predicted.prediction):
                        return Unknown("stale_transition_model")
                    _validate_evidence(agent, binding.model, predicted.prediction)
            ids = agent._capture_observations(events, stage="prediction_guard", providers=(current,),
                                              action=chosen, retain_unavailable=True)
            fresh_sources = [agent.interpretations.get_source(sid) for sid in (*before_ids, *ids)]
            fresh_sources = [s for s in fresh_sources if s.provider == provider_name]
            if not fresh_sources:
                return Unknown("fresh_observation_unavailable")
            for fresh in fresh_sources:
                if fresh.metadata.get("status") != "observed":
                    return Unknown("fresh_observation_unavailable")
                if not _same(fresh.payload, retained.original.payload):
                    return Unknown("stale_transition_observation")
            # Observation callbacks can change mounted executors or model inputs.
            # Revalidate after the final observation, then check structural state
            # once more after projection callbacks used by evidence validation.
            for binding, predicted in zip(retained.bindings, probe.predictions):
                if isinstance(predicted.prediction, TransitionPrediction):
                    if not binding.model.is_current(predicted.prediction):
                        return Unknown("stale_transition_model")
                    _validate_evidence(agent, binding.model, predicted.prediction)
            # A capability callback may unmount the provider while returning
            # unchanged declarations; the trailing lookup must follow it.
            final_provider = _provider(agent, provider_name)
            if (final_provider is not retained.provider or
                    not _same(_capability(final_provider, chosen), retained.capabilities[index]) or
                    _provider(agent, provider_name) is not retained.provider):
                return Unknown("capability_model_changed")
            return _unchanged(agent, retained)
        except Exception as error:
            return Unknown("invalid_model_binding", f"{type(error).__name__}: {error}")

    try:
        provider = _provider(agent, provider_name)
        capability = _capability(provider, chosen)
    except ValueError:
        return finish(None, (), None, "invalid_model_binding")
    receipt = agent._invoke(provider, capability, dict(deepcopy(chosen.args)), events, before_dispatch=guard)
    if receipt.status != "applied":
        return finish(receipt, (), None, "action_not_applied")
    after = [agent.interpretations.get_source(event["source_id"]) for event in events
             if event.get("type") == "observation" and event.get("stage") == "after_action"
             and event.get("provider") == provider_name]
    actual = Unknown("after_observation_unavailable")
    if len(after) == 1 and after[0].metadata.get("status") == "observed":
        try:
            paired = extract_transitions((agent.interpretations.get_source(event['source_id']) for event in events
                if event.get('type') == 'observation'), provider=provider_name)
            rows = [row for row in paired.transitions if row.source_ids[1] == after[0].id]
            if len(rows) != 1:
                raise ValueError('transition context unavailable')
            row = rows[0]
            actual = retained.projection.outcome(deepcopy(row.before), deepcopy(row.action), deepcopy(row.after))
            if not isinstance(actual, Unknown) and not _finite(actual):
                actual = Unknown("nonfinite_observed_outcome")
        except Exception as error:
            actual = Unknown("outcome_unprojectable", f"{type(error).__name__}: {error}")
    assessments = tuple(CandidateAssessment(p.candidate_id, deepcopy(p.prediction), deepcopy(actual),
        "unresolved" if isinstance(actual, Unknown) or isinstance(p.prediction, Unknown) else
        "confirmed" if _contains(p.supported_outcomes, actual) else "contradicted",
        deepcopy(p.supported_outcomes)) for p in probe.predictions)
    viable = [a for a in assessments if a.viable]
    selected = viable[0].candidate_id if len(viable) == 1 and viable[0].status == "confirmed" else None
    state = _unchanged(agent, retained)
    try:
        final_provider = _provider(agent, provider_name)
        if (final_provider is not retained.provider or
                not _same(_capability(final_provider, chosen), retained.capabilities[index]) or
                _provider(agent, provider_name) is not retained.provider):
            state = Unknown("capability_model_changed")
        # Capability callbacks also precede the final group/model snapshot check.
        if state is True:
            state = _unchanged(agent, retained)
    except Exception as error:
        state = Unknown("invalid_model_binding", f"{type(error).__name__}: {error}")
    pending = agent.interpretations.continuation_status(proposal.group_id).pending
    reason = ("stale_investigation_after_execution" if state is not True else
              "pending_interpretations" if pending else
              "supported_unique_applicability" if selected else
              "all_applicability_hypotheses_contradicted" if not viable else "unresolved_applicability")
    if state is not True or pending:
        selected = None
    return finish(receipt, assessments, selected, reason)
