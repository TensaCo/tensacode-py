"""Authored fixtures validate the explicitly selected-statement realization task."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('train_realization', Path(__file__).parents[2] / 'examples/train_realization.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def record(identifier='one', document='Article'):
    return {'id': identifier, 'document_id': document, 'question': 'Who ran?',
            'evidence': [{'source_id': 'original-source', 'text': 'Alice ran home.'}],
            'target': 'Alice ran.', 'target_origin': 'QA2D.turker_answer'}


def budget_model(limit=2000):
    return SimpleNamespace(config={'max_input_tokens': limit},
                           tokenizer=lambda text, **kwargs: {'input_ids': list(text)})


def test_realization_intentionally_conditions_on_selected_human_statement():
    row = record()
    prompt, visible, truncation = example.realization_input(budget_model(), row)
    assert 'Selected hypothesis (not an observation): Alice ran.' in prompt
    assert 'Question: Who ran?' in prompt
    assert '[original-source] Alice ran home.' in prompt
    assert visible[0]['text'] == 'Alice ran home.'
    assert truncation == []
    assert example.interpretation(row)['selected_id'] == 'selected-statement'
    assert 'verifications' not in example.interpretation(row)['candidates'][0]


def test_token_validation_rejects_lost_selected_statement_or_target_truncation():
    row = record()
    with pytest.raises(ValueError, match='target'):
        example.validate_tokens(budget_model(), [row], target_limit=2)
    with pytest.raises(ValueError, match='input'):
        example.validate_tokens(budget_model(5), [row], target_limit=100)
    result = example.validate_tokens(budget_model(), [row], target_limit=100)
    assert result['target_truncated_count'] == 0


def test_fixed_selection_spreads_across_documents_and_rejects_overlap():
    rows = [record(str(i), f'Article{i // 2}') for i in range(12)]
    selected = example.select_records(rows, 4, seed=7)
    assert len({row['document_id'] for row in selected}) == 4
    assert selected == example.select_records(list(reversed(rows)), 4, seed=7)
    with pytest.raises(ValueError, match='overlap'):
        example.check_splits([rows[0]], [rows[1]])
    with pytest.raises(ValueError, match='available'):
        example.select_records(rows, 100, seed=7)


def test_preservation_metrics_penalize_answer_fragments_and_changed_numbers():
    assert example.preservation_scores('Alice', 'Alice ran.')['statement_exact'] == 0
    assert example.preservation_scores('The rate is 8%.', 'The rate is 14%.')['statement_exact'] == 0
    assert example.preservation_scores('Alice ran.', 'Alice ran .')['statement_exact'] == 1
    assert example.preservation_scores('Alice ran.', 'Alice ran .')['verbatim_exact'] == 0


def test_card_uses_configured_foundation_and_schedule_without_claiming_qa_accuracy():
    card = example.model_card(foundation='vendor/alternate', revision='revision-7', epochs=2)
    assert 'vendor/alternate' in card and 'revision-7' in card
    assert '2 epochs' in card
    assert 'FLAN-T5-base' not in card and 'three-epoch' not in card
    assert 'not target-blind' in card
    assert 'license: apache-2.0' not in card
