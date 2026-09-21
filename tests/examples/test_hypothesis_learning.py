import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')


def example():
    path = Path(__file__).parents[2] / 'examples/hypothesis_learning.py'
    spec = importlib.util.spec_from_file_location('hypothesis_learning_example', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_cases(path, cases):
    path.write_text(''.join(json.dumps(case) + '\n' for case in cases))


def test_persisted_supervision_learns_revision_and_fresh_checkpoint_parity(tmp_path):
    mod = example()
    torch.set_num_threads(1)
    # Explicit authored fixture tests the learning mechanism, not real cognition.
    cases = [{'case_id': 'reviewed', 'evidence': [
        {'source_id': 'sensor', 'text': 'silent', 'target': 'idle', 'reviewer': 'operator'},
        {'source_id': 'inspection', 'text': 'smoke heat alarm', 'target': 'fire', 'reviewer': 'operator'},
    ]}]
    training = tmp_path / 'train.jsonl'
    write_cases(training, cases)
    artifacts = tmp_path / 'weights'
    assert mod.collect(training, artifacts, ['idle', 'fire'])['supervised_steps'] == 2
    receipt = mod.train(artifacts, epochs=30)
    assert receipt['mean_step_loss_by_epoch'][-1] < receipt['mean_step_loss_by_epoch'][0] * .1
    inference = tmp_path / 'new.jsonl'
    for step in cases[0]['evidence']:
        del step['target'], step['reviewer']
    write_cases(inference, cases)
    predictions = mod.predict(inference, artifacts)
    assert [p['interpretation'] for p in predictions] == ['idle', 'fire']
    assert [p['revised'] for p in predictions] == [False, True]
    assert predictions[1]['evidence'] == cases[0]['evidence']
    assert predictions[1]['realization'] == 'Current interpretation: fire.'
    _, first = mod.restore(artifacts)
    _, second = mod.restore(artifacts)
    assert first['evidence'] is not second['evidence']
    assert torch.equal(mod.infer(first, 'silent smoke heat alarm').logits,
                       mod.infer(second, 'silent smoke heat alarm').logits)
    evidence = json.loads((artifacts / 'evidence.json').read_text())
    assert evidence[0]['evidence'][1]['reviewer'] == 'operator'
    manifest = json.loads((artifacts / 'manifest.json').read_text())
    assert 'fire' not in manifest['vocabulary']  # targets are not model inputs
    assert 'operator' not in manifest['vocabulary']
    before = (artifacts / 'manifest.json').read_bytes()
    cases[0]['evidence'][0]['text'] = 'previously_unseen_token'
    write_cases(inference, cases)
    mod.predict(inference, artifacts)
    assert (artifacts / 'manifest.json').read_bytes() == before
    assert mod.predict(inference, artifacts) == mod.predict(inference, artifacts)


@pytest.mark.parametrize('case', [
    {'case_id': '', 'evidence': [{'source_id': 'a', 'text': 'x'}]},
    {'case_id': 'a', 'evidence': []},
    {'case_id': 'a', 'evidence': [{'source_id': 'a', 'text': 'x'}, {'source_id': 'a', 'text': 'y'}]},
    {'case_id': 'a', 'evidence': [{'source_id': 'a', 'text': 'x', 'target': 'guessed'}]},
])
def test_invalid_prediction_cases_rejected(tmp_path, case):
    path = tmp_path / 'input.jsonl'
    write_cases(path, [case])
    with pytest.raises(ValueError):
        example().read_cases(path)


def test_collection_requires_reviewed_targets_from_explicit_hypotheses(tmp_path):
    mod = example()
    path = tmp_path / 'input.jsonl'
    case = {'case_id': 'a', 'evidence': [{'source_id': 's', 'text': 'x', 'target': 'other', 'reviewer': 'human'}]}
    write_cases(path, [case])
    with pytest.raises(ValueError, match='outside'):
        mod.collect(path, tmp_path / 'artifacts', ['a', 'b'])
    assert not (tmp_path / 'artifacts').exists()
    del case['evidence'][0]['reviewer']
    write_cases(path, [case])
    with pytest.raises(ValueError, match='reviewer'):
        mod.collect(path, tmp_path / 'artifacts', ['a', 'b'])
    with pytest.raises(ValueError, match='unique'):
        mod.collect(path, tmp_path / 'artifacts', ['a', 'a'])


@pytest.mark.parametrize('lr', [0, -1, float('nan'), float('inf')])
def test_invalid_training_rate_rejected_before_loading(tmp_path, lr):
    with pytest.raises(ValueError, match='finite'):
        example().train(tmp_path, lr=lr)
