import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('learn_action_outcomes', Path(__file__).parents[2] / 'examples/learn_action_outcomes.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_measured_feedback_improves_simulated_policy_and_roundtrips(tmp_path):
    report = example.run(tmp_path)
    assert set(report['train_ids']).isdisjoint(report['test_ids'])
    assert report['after']['success_rate'] == 1.
    assert report['after']['mean_reward'] > report['before']['mean_reward']
    assert report['after']['success_rate'] > report['fixed_cool_baseline']['success_rate']
    assert report['frozen_feedback_success_rate'] == 0.
    assert all(report[key] for key in ['model_parity', 'session_parity', 'trajectory_parity',
        'optimizer_continuation_parity', 'unseen_label_rejected'])
    records = json.loads((tmp_path / 'observations.json').read_text())
    assert len(records) == 18
    for record in records:
        assert set(record['target']) == {'candidate_id', 'outcome'}
        assert record['target']['candidate_id'] == record['candidate_id']
        trajectory = example.PlanExecutionResult.load(tmp_path / record['file'].replace('experience', 'trajectory'))
        experience = trajectory.experiences[0]
        assert experience.source_id == record['source_id']
        assert experience.observation['reward'] == record['target']['outcome']


def test_unregistered_action_is_never_dispatched():
    with pytest.raises(ValueError, match='no fallback'):
        example.structured('unseen-action')
    source = {'scenario': 'isolated', 'status': 'hot'}
    result = example.registry()['cool'](source)
    assert result.receipt['reward'] == .5 and result.state['status'] == 'ready'
    assert source['status'] == 'hot'
