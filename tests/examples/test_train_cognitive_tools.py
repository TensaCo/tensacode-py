"""Authored miniature records isolate data and checkpoint mechanisms only."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('train_cognitive_tools', Path(__file__).parents[2] / 'examples/train_cognitive_tools.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def row(identity='train', question='trainingword'):
    return {'id': identity, 'question': question,
            'context': {'title': ['A', 'B', 'C'], 'sentences': [['alpha'], ['beta'], ['gamma']]},
            'supporting_facts': {'title': ['B', 'C']}}


def test_annotations_and_inputs_do_not_expose_targets():
    record = example.prepare_record(row())
    assert record['target'] == 1
    assert record['targets'] == [0., 1., 1.]
    assert set(example.inputs(record, 'investigator')) == {'question', 'hypotheses', 'evidence'}
    assert example.inputs(record, 'planner')['evidence'] == []


def test_train_only_vocabulary_and_split_overlap():
    train = [example.prepare_record(row())]
    validation = [example.prepare_record(row('validation', 'heldoutword'))]
    example.check_splits(train, validation)
    assert 'heldoutword' not in example.vocabulary(train)
    with pytest.raises(ValueError, match='disjoint'):
        example.check_splits(train, train)


def test_training_saves_reconstructible_owned_models(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    def fixture(split, count):
        return [example.prepare_record(row(split + str(i))) for i in range(count)], {'fixture': True}
    monkeypatch.setattr(example, 'load_records', fixture)
    result = example.run(tmp_path, train_count=2, validation_count=1, epochs=1, dimensions=4)
    for kind in ('investigator', 'planner'):
        assert result['results'][kind]['after'] == result['results'][kind]['reloaded']
        assert (tmp_path / kind / 'tensorcode_config.json').exists()
        assert (tmp_path / kind / 'training-manifest.json').exists()
