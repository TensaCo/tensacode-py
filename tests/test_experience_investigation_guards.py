"""Learned probe forecasts remain bound to their evidence and execution context."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tensorcode.agent.experience_investigation import ModelApplicability
from tensorcode.outcomes import Unknown
from experience_investigation_fixtures import make_setup


def propose(setup):
    bindings = tuple(ModelApplicability(candidate, model, ('authored context-to-model association',))
                     for candidate, model in zip(setup.candidate_ids, setup.models))
    return setup.agent.propose_experience_investigation(
        setup.group_id, bindings, setup.before_source_id, setup.calls)


@pytest.mark.parametrize('drift', ['world', 'model', 'capability', 'group'])
def test_changed_execution_context_cannot_dispatch(drift, monkeypatch):
    s = make_setup()
    proposal = propose(s)
    count = len(s.plugin.executions)
    if drift == 'world':
        s.plugin.lamp = 9
    elif drift == 'model':
        prediction = proposal.probes[0].predictions[0].prediction
        s.models[0].observe_outcome(prediction, 99, source_ids=(s.before_source_id,),
            reason='authored revision guard fixture, not a valid learned counterexample')
    elif drift == 'capability':
        caps = s.plugin.capabilities()
        monkeypatch.setattr(s.plugin, 'capabilities', lambda: tuple(replace(c, description='changed') for c in caps))
    else:
        s.agent.interpretations.reject(s.group_id, s.candidate_ids[0], reason='new supplied evidence')
    result = s.agent.execute_experience_investigation(proposal.id)
    assert result.receipt is None or result.receipt.status == 'rejected'
    assert result.supported_candidate_id is None
    assert len(s.plugin.executions) == count


def test_last_observation_callback_cannot_change_capability_before_dispatch(monkeypatch):
    s = make_setup()
    proposal = propose(s)
    count = len(s.plugin.executions)
    caps = s.plugin.capabilities()
    calls = []
    original = s.plugin.observe_evidence
    def observe():
        calls.append(None)
        if len(calls) == 2:  # Fresh prediction_guard observation, after capability check.
            monkeypatch.setattr(s.plugin, 'capabilities', lambda: tuple(replace(c, description='changed in sensor') for c in caps))
        return original()
    monkeypatch.setattr(s.plugin, 'observe_evidence', observe)
    result = s.agent.execute_experience_investigation(proposal.id)
    assert result.receipt.status == 'rejected'
    assert len(s.plugin.executions) == count
    assert result.supported_candidate_id is None


def test_group_change_during_action_preserves_receipt_but_withholds_applicability(monkeypatch):
    s = make_setup()
    proposal = propose(s)
    original = s.plugin.execute
    def execute(call, *, key=None):
        receipt = original(call, key=key)
        s.agent.interpretations.reject(s.group_id, s.candidate_ids[0], reason='callback changes question')
        return receipt
    monkeypatch.setattr(s.plugin, 'execute', execute)
    result = s.agent.execute_experience_investigation(proposal.id)
    assert result.receipt.status == 'applied'
    assert result.supported_candidate_id is None
    assert result.reason == 'stale_investigation_after_execution'
    assert result.source_ids and result.assessments
    assert all(model.revision == 0 for model in s.models)


def test_missing_after_observation_keeps_every_hypothesis_unresolved(monkeypatch):
    s = make_setup()
    proposal = propose(s)
    count = len(s.plugin.executions)
    original = s.plugin.observe_evidence
    monkeypatch.setattr(s.plugin, 'observe_evidence', lambda: original() if len(s.plugin.executions) == count
                        else Unknown('sensor_unavailable'))
    result = s.agent.execute_experience_investigation(proposal.id)
    assert result.receipt.status == 'applied'
    assert result.supported_candidate_id is None
    assert all(a.status == 'unresolved' for a in result.assessments)
    assert all(model.revision == 0 for model in s.models)


def test_returned_proposal_mutation_cannot_replace_retained_probe_or_predictions():
    s = make_setup()
    proposal = propose(s)
    object.__setattr__(proposal, 'selected_call', None)
    object.__setattr__(proposal.probes[0].predictions[0].prediction, 'outcome', 2)
    result = s.agent.execute_experience_investigation(proposal.id)
    assert result.receipt.status == 'applied'
    assert result.supported_candidate_id == s.candidate_ids[1]
    assert [a.status for a in result.assessments] == ['contradicted', 'confirmed']
    count = len(s.plugin.executions)
    again = s.agent.execute_experience_investigation(proposal.id)
    assert again.reason == 'proposal_already_consumed'
    assert len(s.plugin.executions) == count


class Pending:
    pending = 1
    def advance(self, *, max_expansions, max_candidates):
        return SimpleNamespace(alternatives=(), explored=0, pending=1)


def test_known_pending_interpretations_prevent_unique_applicability_claim():
    s = make_setup()
    s.agent.interpretations.attach_continuation(s.group_id, Pending())
    proposal = propose(s)
    result = s.agent.execute_experience_investigation(proposal.id)
    assert result.receipt.status == 'applied'
    assert result.supported_candidate_id is None
    assert result.reason == 'pending_interpretations'
    assert [a.status for a in result.assessments] == ['contradicted', 'confirmed']
    assert s.agent.interpretations.get(s.group_id).selected_id is None


def test_foreign_sample_ids_cannot_authorize_a_probe_in_another_workspace():
    from tensorcode.agent import Agent
    s = make_setup()
    foreign = Agent([s.plugin])
    observation = foreign.interpretations.add_source('copied observation', modality='observation',
        provider='plugin:switch', payload={'lamp': 0}, metadata={'status': 'observed'})
    group = foreign.interpretations.create_group(observation.id)
    candidates = tuple(foreign.interpretations.propose(group.id, {'context': i}) for i in (1, 2))
    bindings = tuple(ModelApplicability(c.id, m, ('authored association',)) for c, m in zip(candidates, s.models))
    count = len(s.plugin.executions)
    with pytest.raises(ValueError, match='sample evidence'):
        foreign.propose_experience_investigation(group.id, bindings, observation.id, s.calls)
    assert len(s.plugin.executions) == count


def test_forged_fitted_labels_cannot_borrow_authentic_transition_ids():
    from tensorcode.learning.experience import extract_transitions, fit_transitions
    from experience_investigation_fixtures import PROJECTION
    s = make_setup()
    examples = s.models[0].examples
    ids = {e.attempt_id for e in examples}
    actual = tuple(row for row in extract_transitions(s.agent.interpretations.sources(),
                   provider='plugin:switch').transitions if row.attempt_id in ids)
    forged = fit_transitions(tuple(replace(row, after={'lamp': row.after['lamp'] + 10}) for row in actual),
        projection=PROJECTION,
        train_attempt_ids=[e.attempt_id for e in examples if e.split == 'training'],
        evaluation_attempt_ids=[e.attempt_id for e in examples if e.split == 'evaluation'])
    bindings = tuple(ModelApplicability(candidate, model, ('authored association',))
        for candidate, model in zip(s.candidate_ids, (forged, s.models[1])))
    count = len(s.plugin.executions)
    with pytest.raises(ValueError, match='sample content disagrees'):
        s.agent.propose_experience_investigation(s.group_id, bindings, s.before_source_id, s.calls)
    assert len(s.plugin.executions) == count


def test_contract_drift_during_probe_withholds_applicability(monkeypatch):
    s = make_setup()
    proposal = propose(s)
    original = s.plugin.execute
    caps = s.plugin.capabilities()
    def execute(call, *, key=None):
        receipt = original(call, key=key)
        monkeypatch.setattr(s.plugin, 'capabilities', lambda: tuple(replace(c, description='changed during probe') for c in caps))
        return receipt
    monkeypatch.setattr(s.plugin, 'execute', execute)
    result = s.agent.execute_experience_investigation(proposal.id)
    assert result.receipt.status == 'applied'
    assert result.supported_candidate_id is None
    assert result.reason == 'stale_investigation_after_execution'
    assert result.assessments and result.source_ids


@pytest.mark.parametrize('alteration', ['label', 'coverage'])
def test_changed_rule_cannot_claim_old_samples_as_evidence_of_new_outcome(alteration):
    from tensorcode.learning.experience import LearnedTransitionModel
    s = make_setup()
    original = s.models[0]
    artifact = original.artifact
    if alteration == 'label':
        artifact.rules[0].label = 99
    else:
        artifact.rules[0].conditions = ()  # Now covers baseline counterexamples too.
    # Public construction supplies authentic projected examples/support records,
    # but changes the rule's claimed outcome. Source linkage alone is insufficient.
    domains = {}
    for example in original.examples:
        for name, value in example.facts:
            domains.setdefault(name, set()).add(value)
    family_evidence = {(e.rule_index, family): replace(e, action_family=family)
                       for e in original.evidence for family in original.action_families}
    changed = LearnedTransitionModel(artifact, original.projection, original.evidence,
        original.evaluation, original.policy, original.provider,
        {k: tuple(v) for k, v in domains.items()}, family_evidence,
        original.action_families, original.snapshot().source_ids, original.examples)
    bindings = tuple(ModelApplicability(candidate, model, ('authored association',))
                     for candidate, model in zip(s.candidate_ids, (changed, s.models[1])))
    count = len(s.plugin.executions)
    with pytest.raises(ValueError):
        s.agent.propose_experience_investigation(s.group_id, bindings, s.before_source_id, s.calls)
    assert len(s.plugin.executions) == count
