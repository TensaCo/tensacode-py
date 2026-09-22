"""CPU mechanics checks; small supplied policies do not establish general planning."""
import json

import pytest
import torch

from tensorcode.runtime.action_loop import ActionOutcome
from tensorcode.runtime.planning import (ExecutablePlan, PlanExecutionResult,
    PlanExecutor, PlanStep)
from tensorcode.tools.planner import Planner
from test_chatbot_model import tiny_config


def plan(candidate_id, *actions):
    return ExecutablePlan(candidate_id, tuple(PlanStep(action) for action in actions))


def test_real_observation_replans_with_injected_learned_policy(tmp_path):
    # Fit a tiny policy to explicit fixture labels; runtime does not supply policy.
    torch.manual_seed(8)
    classifier = torch.nn.Linear(1, 2)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.4)
    for _ in range(35):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(classifier(torch.tensor([[-1.], [1.]])), torch.tensor([0, 1]))
        loss.backward()
        optimizer.step()
    calls, policy_inputs = [], []
    def inspect(state):
        state['visited'].append('inspect')
        calls.append('inspect')
        return ActionOutcome(state, {'measurement': 1})
    def finish(state):
        calls.append('positive')
        return ActionOutcome(state, {'result': 'finished'}, done=True)
    def policy(request):
        policy_inputs.append(request.evidence)
        measurement = request.experiences[-1].observation['measurement']
        selected = classifier(torch.tensor([[float(measurement)]])).argmax().item()
        return plan('revised', ['negative', 'positive'][selected])
    source = {'visited': []}
    result = PlanExecutor(actions={'inspect': inspect, 'negative': lambda state: ActionOutcome(state, 'wrong'),
        'positive': finish}, replan=policy, max_steps=2)(source, plan('initial', 'inspect', 'negative'))
    assert result.stop_reason == 'completed'
    assert calls == ['inspect', 'positive'] and source == {'visited': []}
    assert [x.candidate_id for x in result.experiences] == ['initial', 'revised']
    assert policy_inputs[0][0]['source_id'] == result.experiences[0].source_id
    assert result.experiences[1].to_target(1.) == {'candidate_id': 'revised', 'outcome': 1.}
    result.save(tmp_path / 'trajectory.json')
    assert PlanExecutionResult.load(tmp_path / 'trajectory.json') == result
    assert 'classifier' not in (tmp_path / 'trajectory.json').read_text()


@pytest.mark.parametrize('bad', [plan('bad', 'allowed', 'unknown'),
    ExecutablePlan('bad', (PlanStep('allowed'), PlanStep('allowed', {'unexpected': 1})))])
def test_whole_plan_validation_prevents_any_effect(bad):
    calls = []
    executor = PlanExecutor(actions={'allowed': lambda state: calls.append(1)}, replan=lambda _: None, max_steps=2)
    with pytest.raises((ValueError, TypeError)):
        executor({}, bad)
    assert calls == []


@pytest.mark.parametrize('output', ['do allowed next', {'selected_id': 'allowed'}, plan('bad', 'unknown')])
def test_invalid_policy_never_defaults_to_first_action(output):
    calls = []
    def allowed(state):
        calls.append(1)
        return ActionOutcome(state, {'actual': True})
    result = PlanExecutor(actions={'allowed': allowed}, replan=lambda _: output, max_steps=3)({}, plan('initial', 'allowed'))
    assert calls == [1] and result.stop_reason == 'policy_error'
    assert len(result.experiences) == 1 and result.policy_errors


def test_failure_is_observation_and_policy_errors_are_visible():
    external = []
    def effect(state):
        external.append('effect')
        state['local'].append('mutated')
        raise RuntimeError('after external effect')
    def policy(request):
        assert request.experiences[0].status == 'error'
        assert request.state == {'local': []}
        raise ValueError('model callback failed')
    result = PlanExecutor(actions={'effect': effect}, replan=policy, max_steps=2)({'local': []}, plan('p', 'effect'))
    assert external == ['effect']
    assert result.experiences[0].observation['error_type'] == 'RuntimeError'
    assert result.policy_errors == ('ValueError: model callback failed',)


def test_owned_plan_generation_checkpoint_and_observed_only_loss(tmp_path, monkeypatch):
    model = Planner({'vocabulary': ['goal', 'step', 'hello'], 'dimensions': 8,
                     'generator': tiny_config()}).eval()
    inputs = {'goal': 'hello', 'evidence': [{'source_id': 's1', 'text': 'hello'}]}
    monkeypatch.setattr(model.generator.tokenizer, 'batch_decode',
                        lambda *a, **k: ['1. Read observation\n2. Revise action', ''])
    generated = model.propose(inputs, count=2)
    assert len(generated) == 1 and generated[0]['text'] == '1. Read observation\n2. Revise action'
    assert generated[0]['origin'] == 'generated' and generated[0]['source_ids'] == ['s1']
    receipt = model(inputs)
    assert receipt['selected_id'] == generated[0]['id']
    loss = model.loss(dict(inputs, plans=generated), {'candidate_id': generated[0]['id'], 'outcome': 0.5})
    loss.backward()
    assert any(p.grad is not None for p in model.rank.parameters())
    model.save_pretrained(tmp_path / 'model')
    loaded = Planner.from_pretrained(tmp_path / 'model', local_files_only=True)
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items())
    assert loaded.generator.configuration() == model.generator.configuration()
    monkeypatch.setattr(model.generator.tokenizer, 'batch_decode', lambda *a, **k: ['', '', ''])
    assert model(inputs)['selected_id'] is None
    assert inputs == {'goal': 'hello', 'evidence': [{'source_id': 's1', 'text': 'hello'}]}


def test_generation_objective_reaches_owned_generator_and_invalid_scores_fail():
    model = Planner({'vocabulary': ['hello'], 'dimensions': 8, 'generator': tiny_config()})
    loss = model.generation_loss({'goal': 'hello'}, 'answer')
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.generator.parameters())
    with torch.no_grad():
        model.rank.score.module[-1].bias.fill_(float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        model({'goal': 'hello', 'plans': [{'id': 'a', 'text': 'hello'}]})


@pytest.mark.parametrize('field,value', [
    ('version', True), ('version', 1.), ('stop_reason', 'fabricated'),
    ('stop_reason', []), ('policy_errors', 'oops'), ('policy_errors', [1]),
    ('policy_errors', ['unexpected']), ('experiences', {}),
    ('state', float('nan')), ('extra', 'unknown')])
def test_malformed_trajectory_topology_rejected(tmp_path, field, value):
    result = PlanExecutor(actions={'a': lambda state: ActionOutcome(state, 'ok', True)},
                          replan=lambda _: None, max_steps=1)({}, plan('p', 'a'))
    path = tmp_path / 'trajectory.json'
    result.save(path)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PlanExecutionResult.load(path)


@pytest.mark.parametrize('field,value', [
    ('candidate_id', 123), ('candidate_id', ' '), ('action', {'bad': 1}),
    ('source_id', []), ('arguments', []), ('status', 'invented'),
    ('observation', float('inf')), ('expected_observation', float('nan')),
    ('extra', 'unknown')])
def test_malformed_experience_rejected(tmp_path, field, value):
    result = PlanExecutor(actions={'a': lambda state: ActionOutcome(state, 'ok', True)},
                          replan=lambda _: None, max_steps=1)({}, plan('p', 'a'))
    path = tmp_path / 'trajectory.json'
    result.save(path)
    data = json.loads(path.read_text())
    data['experiences'][0][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        PlanExecutionResult.load(path)


@pytest.mark.parametrize('arguments', [{'value': (1, 2)}, {'value': {1: 'coerced'}}])
def test_non_json_arguments_fail_before_effect(arguments):
    calls = []
    executor = PlanExecutor(actions={'a': lambda state, value: calls.append(value)},
                            replan=lambda _: None, max_steps=1)
    with pytest.raises(ValueError, match='JSON'):
        executor({}, ExecutablePlan('p', (PlanStep('a', arguments),)))
    assert calls == []


def test_trajectory_save_is_atomic_and_rejects_duplicate_fields(tmp_path, monkeypatch):
    import tensorcode.runtime.planning as module
    path = tmp_path / 'trajectory.json'
    result = PlanExecutionResult({}, (), 'budget_exhausted')
    result.save(path)
    previous = path.read_bytes()
    def failed_replace(*args):
        raise OSError('injected replace failure')
    monkeypatch.setattr(module.os, 'replace', failed_replace)
    with pytest.raises(OSError):
        result.save(path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError):
        PlanExecutionResult(float('nan'), (), 'budget_exhausted').save(path)
    assert path.read_bytes() == previous
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError, match='duplicate'):
        PlanExecutionResult.load(path)


@pytest.mark.parametrize('candidate_id', ['', ' ', '\t\n'])
def test_invalid_candidate_ids_rejected_before_effect(candidate_id):
    calls = []
    executor = PlanExecutor(actions={'valid': lambda state: calls.append('effect')},
                            replan=lambda _: None, max_steps=1)
    with pytest.raises(ValueError, match='candidate_id'):
        executor({}, plan(candidate_id, 'valid'))
    assert calls == []


@pytest.mark.parametrize('action_id', ['', ' ', '\t\n'])
def test_invalid_registry_ids_rejected_before_effect(action_id):
    calls = []
    with pytest.raises(ValueError, match='names'):
        executor = PlanExecutor(actions={action_id: lambda state: calls.append('effect')},
                                replan=lambda _: None, max_steps=1)
        executor({}, plan('candidate', action_id))
    assert calls == []


def test_valid_nonempty_ids_preserve_exact_identity_through_persistence(tmp_path):
    result = PlanExecutor(actions={' action ': lambda state: ActionOutcome(state, {'value': 1}, True)},
                          replan=lambda _: None, max_steps=1)({}, plan(' candidate ', ' action '))
    path = tmp_path / 'trajectory.json'
    result.save(path)
    restored = PlanExecutionResult.load(path)
    assert restored == result
    assert restored.experiences[0].candidate_id == ' candidate '
    assert restored.experiences[0].action == ' action '
