"""Supplied literal projection learns outcomes only from authenticated paired samples."""
from copy import deepcopy
from dataclasses import dataclass, replace
from uuid import uuid4
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.plugin import Call
from tensorcode.agent.document_transition_evidence import (
    browser_transition_projection, retain_document_transition_batch, fit_document_transitions,
    predict_document_transition, observe_document_transition, validate_document_split,
)
from tensorcode.learning.experience import extract_transitions
from tensorcode.outcomes import Unknown, Receipt


@dataclass(frozen=True)
class Target:
    token: str
    connection_id: str
    session_id: str
    frame_id: str
    frame_loaders: tuple
    backend_node_id: int
    action: Call


class Provider:
    name = 'authored-browser'
    def __init__(self):
        self.issued = {}
        self.targets = {}
        self.current = None
    def target(self, document):
        token = uuid4().hex
        result = Target(token, 'connection', 'session', 'frame', (('frame', document),),
                        99, Call(self.name, 'activate_node', (('target', token),)))
        self.targets[token] = result
        return result
    def observation(self, target, checked):
        result = {'document_observation_id': uuid4().hex,
            'document_identity': {'connection_id': target.connection_id, 'session_id': target.session_id,
                                  'frame_loaders': target.frame_loaders},
            'document_targets': (target,),
            'document_snapshot': {'strings': ['frame', 'INPUT', 'type', 'checkbox'],
                'documents': [{'frameId': 0, 'nodes': {'backendNodeId': [99], 'nodeName': [1],
                  'nodeType': [1], 'attributes': [[2, 3]], 'inputChecked': {'index': [0] if checked else []}}}]}}
        self.issued[result['document_observation_id']] = deepcopy(result)
        return result
    def authenticate_document_observation(self, observed):
        return True if self.issued.get(observed.get('document_observation_id')) == observed else Unknown('forged')
    def validate_document_target_observation(self, target, observed):
        return True if (self.authenticate_document_observation(observed) is True
            and self.targets.get(target.token) == target and target in observed['document_targets']
            and observed['document_identity']['frame_loaders'] == target.frame_loaders) else Unknown('identity')
    def validate_document_target(self, token):
        return True if token in self.targets else Unknown('target')
    def observe_evidence(self):
        return deepcopy(self.current)


def retain_pair(agent, provider, target, before, after):
    attempt = uuid4().hex
    for stage, payload, receipt in [('before_action', before, None),
                                    ('after_action', after, Receipt(target.action, 'applied'))]:
        agent.interpretations.add_source('', modality='observation', provider='plugin:' + provider.name,
            metadata={'stage': stage, 'attempt_id': attempt, 'action': target.action,
                      'receipt': receipt, 'status': 'observed'}, payload=payload)
    return attempt


def trained():
    agent, provider = Agent([]), Provider()
    attempts = []
    for index in range(3):
        target = provider.target('document-' + str(index))
        attempts.append(retain_pair(agent, provider, target, provider.observation(target, False), provider.observation(target, True)))
    batch = retain_document_transition_batch(agent, provider)
    assert not isinstance(batch, Unknown), batch
    model = fit_document_transitions(agent, provider, batch, train_attempt_ids=attempts[:2], evaluation_attempt_ids=attempts[2:])
    assert not isinstance(model, Unknown), model
    return agent, provider, batch, model


def test_novel_target_prediction_and_counterexample_without_toggle_prior():
    agent, provider, _, model = trained()
    target = provider.target('fresh-document')
    provider.current = provider.observation(target, False)
    prediction = predict_document_transition(agent, provider, model, target.token)
    assert not isinstance(prediction, Unknown), prediction
    assert prediction.prediction.outcome is True
    before = provider.observation(target, False)
    after = provider.observation(target, False)  # observed counterexample, no forced toggle
    attempt = retain_pair(agent, provider, target, before, after)
    event = observe_document_transition(agent, provider, model, prediction, attempt)
    assert not isinstance(event, Unknown), event
    assert event.observed is False and event.predicted is True
    assert isinstance(model.predict(provider.current, target.action), Unknown)
    assert isinstance(observe_document_transition(agent, provider, model, prediction, attempt), Unknown)


def test_projection_contains_literal_target_features_and_uses_contextual_after():
    provider = Provider()
    target = provider.target('document')
    before, after = provider.observation(target, False), provider.observation(target, True)
    projection = browser_transition_projection()
    assert projection.features(before, target.action) == {'nodeName': 'INPUT', 'nodeType': 1,
        'type_attribute': 'checkbox', 'inputChecked': False}
    assert projection.outcome(before, target.action, after) is True
    altered = replace(target, token='different-token')
    after['document_targets'] = (altered,)
    with pytest.raises(ValueError):
        projection.outcome(before, target.action, after)


@pytest.mark.parametrize('indices', [None, [True], [0, 0], [-1], [1], '0'])
def test_malformed_sparse_boolean_never_becomes_negative_evidence(indices):
    provider = Provider()
    target = provider.target('doc')
    before = provider.observation(target, False)
    nodes = before['document_snapshot']['documents'][0]['nodes']
    if indices is None:
        del nodes['inputChecked']
    else:
        nodes['inputChecked'] = {'index': indices}
    with pytest.raises((ValueError, KeyError)):
        browser_transition_projection().features(before, target.action)


def test_repeated_document_targets_cannot_cross_training_validation_split():
    agent, provider = Agent([]), Provider()
    attempts = []
    for _ in range(2):
        target = provider.target('same-document')
        attempts.append(retain_pair(agent, provider, target, provider.observation(target, False), provider.observation(target, True)))
    batch = extract_transitions(agent.interpretations.sources(), provider='plugin:' + provider.name)
    with pytest.raises(ValueError, match='reuse a document'):
        validate_document_split(batch.transitions, attempts[:1], attempts[1:])


def test_forged_raw_observations_are_excluded_and_cached_batch_changes_rejected():
    agent, provider, batch, _ = trained()
    target = provider.target('forged-document')
    before, after = provider.observation(target, False), provider.observation(target, True)
    before['document_snapshot']['strings'][1] = 'DIV'
    attempt = retain_pair(agent, provider, target, before, after)
    fresh = retain_document_transition_batch(agent, provider)
    assert attempt not in {row.attempt_id for row in fresh.batch.transitions}
    assert attempt in {row.attempt_id for row in fresh.batch.exclusions}
    batch.batch.transitions[0].before['document_snapshot']['strings'][1] = 'CHANGED'
    result = fit_document_transitions(agent, provider, batch,
        train_attempt_ids=[r.attempt_id for r in batch.batch.transitions[:2]],
        evaluation_attempt_ids=[batch.batch.transitions[2].attempt_id])
    assert isinstance(result, Unknown)


def test_missing_replaced_or_navigated_target_is_not_a_false_outcome():
    provider = Provider()
    target = provider.target('doc')
    before, after = provider.observation(target, False), provider.observation(target, False)
    after['document_snapshot']['documents'][0]['nodes']['backendNodeId'] = [100]
    with pytest.raises(ValueError, match='missing'):
        browser_transition_projection().outcome(before, target.action, after)
    after = provider.observation(target, False)
    after['document_identity']['frame_loaders'] = (('frame', 'navigated'),)
    with pytest.raises(ValueError, match='identity'):
        browser_transition_projection().outcome(before, target.action, after)


def test_feedback_must_match_retained_prestate_and_original_prediction():
    agent, provider, _, model = trained()
    target = provider.target('fresh')
    provider.current = provider.observation(target, False)
    prediction = predict_document_transition(agent, provider, model, target.token)
    attempt = retain_pair(agent, provider, target, provider.observation(target, True), provider.observation(target, False))
    result = observe_document_transition(agent, provider, model, prediction, attempt)
    assert isinstance(result, Unknown) and not model.history
    forged = replace(prediction, evidence_source_id='forged')
    assert isinstance(observe_document_transition(agent, provider, model, forged, attempt), Unknown)
    assert not model.history


def test_forged_future_observation_is_not_a_prediction_context():
    agent, provider, _, model = trained()
    target = provider.target('fresh')
    provider.current = provider.observation(target, False)
    provider.current['document_snapshot']['strings'][1] = 'DIV'
    prediction = predict_document_transition(agent, provider, model, target.token)
    assert isinstance(prediction, Unknown)


@pytest.mark.parametrize('change', ['label', 'conditions', 'projection'])
def test_changed_model_cannot_borrow_authentic_transition_evidence(change):
    agent, provider, _, model = trained()
    target = provider.target('fresh')
    provider.current = provider.observation(target, False)
    if change == 'label': model._artifact.rules[0].label = False
    elif change == 'conditions': model._artifact.rules[0].conditions = ()
    else: model._projection = replace(model.projection, outcome=lambda before, action, after: False)
    result = predict_document_transition(agent, provider, model, target.token)
    assert isinstance(result, Unknown) and 'changed' in result.detail


def test_provider_callback_rule_change_cannot_publish_prediction():
    agent, provider, _, model = trained()
    target = provider.target('fresh')
    provider.current = provider.observation(target, False)
    original = provider.authenticate_document_observation
    def changed(observation):
        model._artifact.rules[0].label = False
        return original(observation)
    provider.authenticate_document_observation = changed
    assert isinstance(predict_document_transition(agent, provider, model, target.token), Unknown)


def test_cloned_workspace_attempts_cannot_inflate_provider_observation_support():
    agent, provider = Agent([]), Provider()
    target = provider.target('shared')
    before, after = provider.observation(target, False), provider.observation(target, True)
    attempts = {retain_pair(agent, provider, target, before, after) for _ in range(2)}
    retained = retain_document_transition_batch(agent, provider)
    assert not isinstance(retained, Unknown)
    assert not retained.batch.transitions
    assert {row.attempt_id for row in retained.batch.exclusions} == attempts
    assert all('reused' in row.reason for row in retained.batch.exclusions)


def test_literal_cdp_empty_boolean_attribute_value_is_decoded():
    provider = Provider()
    target = provider.target('document')
    before = provider.observation(target, True)
    snapshot = before['document_snapshot']
    snapshot['strings'].append('checked')
    snapshot['documents'][0]['nodes']['attributes'][0].extend([4, -1])
    assert browser_transition_projection().features(before, target.action)['inputChecked'] is True


def fresh_prediction():
    agent, provider, _, model = trained()
    target = provider.target('fresh')
    provider.current = provider.observation(target, False)
    prediction = predict_document_transition(agent, provider, model, target.token)
    assert not isinstance(prediction, Unknown)
    return agent, provider, model, target, prediction


def before_source(agent, provider, target, *, checked=False, action=None):
    return agent.interpretations.add_source('', modality='observation', provider='plugin:' + provider.name,
        metadata={'stage': 'before_action', 'attempt_id': uuid4().hex, 'action': action or target.action,
                  'receipt': None, 'status': 'observed'}, payload=provider.observation(target, checked))


def test_prospective_validation_accepts_new_observation_identity_with_same_target_state():
    from tensorcode.agent.document_transition_evidence import validate_document_prediction
    agent, provider, model, target, prediction = fresh_prediction()
    source = before_source(agent, provider, target)
    original = agent.interpretations.get_source(prediction.evidence_source_id)
    assert source.payload['document_observation_id'] != original.payload['document_observation_id']
    assert validate_document_prediction(agent, provider, model, prediction, (source.id,)) is True
    assert validate_document_prediction(agent, provider, model, prediction, (source.id,)) is True


@pytest.mark.parametrize('change', ['state', 'action', 'model', 'duplicate'])
def test_prospective_validation_refuses_changed_context(change):
    from tensorcode.agent.document_transition_evidence import validate_document_prediction
    agent, provider, model, target, prediction = fresh_prediction()
    other = provider.target('other')
    source = before_source(agent, provider, target, checked=change == 'state',
                           action=other.action if change == 'action' else None)
    if change == 'model': model._artifact.rules[0].label = False
    ids = (source.id, source.id) if change == 'duplicate' else (source.id,)
    assert isinstance(validate_document_prediction(agent, provider, model, prediction, ids), Unknown)


def test_final_transport_callback_cannot_change_model_before_dispatch_validation_returns():
    from tensorcode.agent.document_transition_evidence import validate_document_prediction
    agent, provider, model, target, prediction = fresh_prediction()
    source = before_source(agent, provider, target)
    def mutate(token):
        model._artifact.rules[0].conditions = ()
        return True
    provider.validate_document_target = mutate
    assert isinstance(validate_document_prediction(agent, provider, model, prediction, (source.id,)), Unknown)


def test_assessment_returns_actual_observation_and_receipt_and_consumes_once():
    from tensorcode.agent.document_transition_evidence import assess_document_transition, validate_document_prediction
    agent, provider, model, target, prediction = fresh_prediction()
    attempt = retain_pair(agent, provider, target, provider.observation(target, False), provider.observation(target, True))
    result = assess_document_transition(agent, provider, model, prediction, attempt)
    assert not isinstance(result, Unknown), result
    assert result.outcome is True and result.receipt.status == 'applied'
    assert result.attempt_id == attempt and len(result.source_ids) == 2
    assert result.suspension is None
    assert agent.interpretations.get_source(result.evidence_source_id).payload['outcome'] is True
    assert isinstance(observe_document_transition(agent, provider, model, prediction, attempt), Unknown)
    source = before_source(agent, provider, target)
    assert isinstance(validate_document_prediction(agent, provider, model, prediction, (source.id,)), Unknown)


def test_assessment_counterexample_preserves_receipt_and_suspension():
    from tensorcode.agent.document_transition_evidence import assess_document_transition
    agent, provider, model, target, prediction = fresh_prediction()
    attempt = retain_pair(agent, provider, target, provider.observation(target, False), provider.observation(target, False))
    result = assess_document_transition(agent, provider, model, prediction, attempt)
    assert result.outcome is False and result.receipt.status == 'applied'
    assert result.suspension is not None and result.suspension.observed is False


def test_terminal_authority_validation_has_no_provider_callbacks():
    from tensorcode.agent.document_transition_evidence import validate_document_prediction_authority
    agent, provider, model, _, prediction = fresh_prediction()
    def forbidden(*args):
        pytest.fail('terminal authority check invoked provider')
    provider.observe_evidence = forbidden
    provider.authenticate_document_observation = forbidden
    provider.validate_document_target_observation = forbidden
    provider.validate_document_target = forbidden
    assert validate_document_prediction_authority(agent, provider, model, prediction) is True


@pytest.mark.parametrize('mutation', ['rule', 'suspension'])
def test_terminal_authority_rejects_model_changed_by_preceding_provider_callback(mutation):
    from tensorcode.agent.document_transition_evidence import validate_document_prediction_authority
    agent, provider, model, target, prediction = fresh_prediction()
    def final_provider_callback(token):
        if mutation == 'rule':
            model._artifact.rules[0].label = False
        else:
            model.observe_outcome(prediction.prediction, False, source_ids=('authored-counterexample',),
                                  reason='supplied callback counterexample fixture')
        return True
    provider.validate_document_target = final_provider_callback
    assert provider.validate_document_target(target.token) is True
    assert isinstance(validate_document_prediction_authority(agent, provider, model, prediction), Unknown)
