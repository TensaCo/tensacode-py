"""Authored records test isolation and label mechanics, not NLI competence."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('train_verifier', Path(__file__).parents[2] / 'examples/train_verifier.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_target_mapping_and_input_isolation():
    for label, expected in [(0, 'support'), (1, 'unknown'), (2, 'contradiction')]:
        record = example.prepare_record({'premise': 'One person runs.', 'hypothesis': 'Someone moves.', 'label': label}, 'train', 4)
        assert record['target'] == expected
        assert example.model_inputs([record]) == [{'premise': 'One person runs.', 'hypothesis': 'Someone moves.'}]
    assert example.prepare_record({'premise': 'p', 'hypothesis': 'h', 'label': -1}, 'train', 5) is None


def test_partition_rejects_content_leakage_even_with_different_ids():
    a = example.prepare_record({'premise': 'p', 'hypothesis': 'h', 'label': 0}, 'train', 1)
    b = example.prepare_record({'premise': 'p', 'hypothesis': 'h', 'label': 0}, 'test', 9)
    with pytest.raises(ValueError, match='overlap'):
        example.check_splits({'train': [a], 'test': [b]})


def test_selection_is_fixed_deduplicated_and_excludes_unlabeled():
    rows = [{'premise': 'p', 'hypothesis': 'h', 'label': -1},
            {'premise': 'p1', 'hypothesis': 'h1', 'label': 0},
            {'premise': 'p1', 'hypothesis': 'h1', 'label': 0},
            {'premise': 'p2', 'hypothesis': 'h2', 'label': 2}]
    selected = example.select_records(rows, 'train', 2)
    assert [x['id'] for x in selected] == ['train:1', 'train:3']
    with pytest.raises(ValueError, match='available'):
        example.select_records(rows, 'train', 3)


def test_label_mapping_rejects_ambiguous_classifier_labels():
    assert example.label_mapping({'0': 'contradiction', '1': 'entailment', '2': 'neutral'}) == {'contradiction': 0, 'support': 1, 'unknown': 2}
    with pytest.raises(ValueError, match='label'):
        example.label_mapping({'0': 'LABEL_0', '1': 'LABEL_1', '2': 'LABEL_2'})
