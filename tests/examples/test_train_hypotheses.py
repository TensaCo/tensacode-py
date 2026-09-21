"""Authored fixtures verify source isolation, never claim language competence."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('train_hypotheses', Path(__file__).parents[2] / 'examples/train_hypotheses.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def fixture(identifier='id', title='Article'):
    source = {'data': [{'title': title, 'paragraphs': [{'context': 'The original paragraph.', 'qas': [
        {'id': identifier, 'question': 'What happened?', 'answers': [{'text': 'SECRET ANSWER', 'answer_start': 0}]}]}]}]}
    row = {'dataset': 'SQuAD', 'example_uid': identifier, 'question': 'What happened ?',
           'answer': 'SECRET ANSWER', 'turker_answer': 'Human declaration.', 'rule-based': 'MACHINE LABEL'}
    return source, row


def test_join_uses_only_original_question_context_and_human_target():
    source, row = fixture()
    record = example.join_record(row, example.squad_index([source]), 'train')
    assert record['target'] == 'Human declaration.'
    assert record['question'] == 'What happened?'
    assert record['evidence'][0]['text'] == 'The original paragraph.'
    assert record['document_id'] == 'Article'
    prompt = example.model_input(record)
    assert 'SECRET ANSWER' not in prompt
    assert 'Human declaration' not in prompt
    assert 'MACHINE LABEL' not in prompt
    assert 'The original paragraph.' in prompt


def test_join_rejects_missing_or_mismatched_source_and_nonhuman_target():
    source, row = fixture()
    assert example.join_record(dict(row, dataset='RACE'), {}, 'train') is None
    with pytest.raises(ValueError, match='source'):
        example.join_record(row, {}, 'train')
    with pytest.raises(ValueError, match='question'):
        example.join_record(dict(row, question='Different question?'), example.squad_index([source]), 'train')
    with pytest.raises(ValueError, match='human'):
        example.join_record(dict(row, turker_answer=''), example.squad_index([source]), 'train')


def test_split_is_deterministic_and_groups_whole_original_documents():
    records = []
    for i in range(12):
        source, row = fixture(str(i), f'Article{i // 2}')
        records.append(example.join_record(row, example.squad_index([source]), 'train'))
    # Distinct original contexts avoid deliberate text-overlap rejection.
    for i, record in enumerate(records):
        record['evidence'][0]['text'] += str(i // 2)
    a = example.document_splits(records, train_count=4, dev_count=2, test_count=2, seed=7)
    b = example.document_splits(list(reversed(records)), train_count=4, dev_count=2, test_count=2, seed=7)
    assert a == b
    documents = [{r['document_id'] for r in a[s]} for s in ('train', 'dev', 'test')]
    assert all(not documents[i] & documents[j] for i in range(3) for j in range(i))
    with pytest.raises(ValueError, match='overlap'):
        example.check_splits({'train': [records[0]], 'test': [records[1]]})


def test_declaration_metrics_are_not_short_answer_scores():
    score = example.declaration_scores('Bertie', "Prince Albert's nickname was Bertie.")
    assert score['exact_declaration'] == 0
    assert 0 < score['token_f1'] < 1
    assert example.declaration_scores('A cat sits.', 'A cat sits .')['exact_declaration'] == 1


def test_model_card_identifies_configured_foundation_revision_and_schedule():
    card = example.model_card(foundation='vendor/alternate', revision='revision-7', epochs=2)
    assert 'vendor/alternate' in card and 'revision-7' in card
    assert '2 epochs' in card
    assert 'FLAN-T5-small' not in card and 'license: apache-2.0' not in card


def test_token_audit_reports_supervised_target_truncation():
    from types import SimpleNamespace
    model = SimpleNamespace(tokenizer=lambda text, **kwargs: {'input_ids': list(text)})
    source, row = fixture()
    record = example.join_record(row, example.squad_index([source]), 'train')
    audit = example.token_limits(model, {'train': [record]}, input_limit=10000, target_limit=3)
    assert audit['input_truncated_counts'] == {'train': 0}
    assert audit['target_truncated_counts'] == {'train': 1}


def test_nli_summary_separates_truncated_pairs_and_denominators():
    summary = example.nli_summary([
        {'nli_model_label': 'support', 'nli_input_truncated': True},
        {'nli_model_label': 'unknown', 'nli_input_truncated': False},
        {'nli_model_label': 'support', 'nli_input_truncated': False},
        {'nli_model_label': None, 'nli_input_truncated': None},
    ])
    assert summary['nli_evaluated_count'] == 3
    assert summary['nli_truncated_count'] == 1
    assert summary['nli_untruncated_count'] == 2
    assert summary['nli_model_support_rate'] == pytest.approx(2 / 3)
    assert summary['nli_support_rate_all_samples'] == .5
    assert summary['nli_model_support_rate_untruncated'] == .5
