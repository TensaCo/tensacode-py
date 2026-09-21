"""Authored outcomes isolate local regression/replay, not planning intelligence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / 'examples' / 'plan_learning.py'
spec = importlib.util.spec_from_file_location('plan_learning', SCRIPT)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def row(identifier='incident-1', task='restore checkout service'):
    return {'id': identifier, 'task': task, 'evidence': [{'id': 'log-1', 'text': 'new deployment caused failures'}],
            'plans': [{'id': 'rollback', 'text': 'rollback deployment', 'outcome': 1., 'source': 'incident:resolved'},
                      {'id': 'wait', 'text': 'wait for recovery', 'outcome': 0., 'source': 'incident:no-recovery'}]}


def put(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def test_training_checkpoint_and_heldout_restart(tmp_path):
    torch.set_num_threads(2)
    source = put(tmp_path / 'train.jsonl', [row()])
    heldout = put(tmp_path / 'heldout.jsonl', [row('incident-2', 'restore checkout service urgently')])
    artifacts = tmp_path / 'model'
    example.collect(source, artifacts, dimensions=12)
    before = example.evaluate(heldout, artifacts, checkpoint='initial.json')
    report = example.train(artifacts, epochs=100, lr=.03)
    after = example.evaluate(heldout, artifacts)
    assert report['last_loss'] < report['first_loss'] * .02
    assert after['heldout_mse'] < before['heldout_mse']
    assert after['tasks'][0]['selected_plan'] == 'rollback'
    assert len(after['tasks'][0]['candidates']) == 2
    assert after['tasks'][0]['executed'] is False
    manifest = json.loads((artifacts / 'manifest.json').read_text())
    assert 'task_urgently' not in manifest['vocabulary']
    assert manifest['observations'][0]['plans'][0]['source'] == 'incident:resolved'
    env = dict(os.environ)
    env['PYTHONPATH'] = str(SCRIPT.parents[1] / 'src')
    completed = subprocess.run([sys.executable, str(SCRIPT), 'predict', '--input', str(heldout),
                                '--artifacts', str(artifacts)], check=True, capture_output=True, text=True, env=env)
    assert json.loads(completed.stdout) == after
    with pytest.raises(ValueError, match='held out'):
        example.evaluate(source, artifacts)


@pytest.mark.parametrize('mutation', [
    lambda r: r['plans'][0].update(outcome=float('nan')),
    lambda r: r['plans'][0].update(outcome=True),
    lambda r: r['plans'][0].pop('source'),
    lambda r: r['plans'][1].update(id='rollback'),
    lambda r: r.update(evidence='logs'),
])
def test_bad_observations_rejected(tmp_path, mutation):
    record = row()
    mutation(record)
    source = put(tmp_path / 'bad.jsonl', [record])
    with pytest.raises(ValueError):
        example.collect(source, tmp_path / 'model')
    assert not (tmp_path / 'model').exists()


def test_revisit_task_without_feedback_and_preserve_observations(tmp_path):
    observed = row()
    # Only the executed plan has an observed result. Do not invent alternatives.
    observed['plans'] = observed['plans'][:1]
    source = put(tmp_path / 'observed.jsonl', [observed])
    artifacts = tmp_path / 'model'
    example.collect(source, artifacts, dimensions=8)
    assert json.loads((artifacts / 'observations.json').read_text()) == [observed]
    inference = row()
    for plan in inference['plans']:
        del plan['outcome'], plan['source']
    result = example.evaluate(put(tmp_path / 'revisit.jsonl', [inference]), artifacts, checkpoint='initial.json')
    assert result['heldout_mse'] is None
    assert result['tasks'][0]['task'] == inference['task']
    assert result['tasks'][0]['evidence'] == inference['evidence']
    assert len(result['tasks'][0]['candidates']) == 2


def test_outcome_and_source_never_enter_prediction(tmp_path):
    artifacts = tmp_path / 'model'
    example.collect(put(tmp_path / 'train.jsonl', [row()]), artifacts, dimensions=8)
    heldout = row('heldout', 'restore checkout urgently')
    original = example.evaluate(put(tmp_path / 'before.jsonl', [heldout]), artifacts, checkpoint='initial.json')
    original_texts = example.texts(heldout)
    for plan in heldout['plans']:
        plan['outcome'] = 1000.
        plan['source'] = 'changed-feedback-source'
    changed = example.evaluate(put(tmp_path / 'after.jsonl', [heldout]), artifacts, checkpoint='initial.json')
    assert example.texts(heldout) == original_texts
    assert changed['tasks'] == original['tasks']
    assert changed['heldout_mse'] != original['heldout_mse']
