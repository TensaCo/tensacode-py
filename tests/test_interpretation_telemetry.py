"""Evaluation telemetry counts sentences, not copies attached to alternatives."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from eval.parsing.span_evaluation import Gold, Word, READER_PHASES, RETENTION_COUNTERS, reader_telemetry, select_cohort


def reading(copies=3):
    stats = {key: 2 for key in RETENTION_COUNTERS}
    phases = {key: 1.5 for key in READER_PHASES}
    return SimpleNamespace(alternatives=tuple(SimpleNamespace(metadata={'retention_stats': deepcopy(stats),
                        'reader_phase_ms': deepcopy(phases)}) for _ in range(copies)))


def test_repeated_alternative_metadata_is_counted_once_per_sentence():
    result = reader_telemetry([reading(8), reading(2)])
    assert result['instrumented_sentences'] == 2
    assert result['retention']['syntax_generated'] == 4
    assert result['phase_ms']['semantics'] == 3
    assert not result['errors']


@pytest.mark.parametrize('field,value', [('syntax_generated', True), ('syntax_generated', -1),
                                         ('semantics', float('nan')), ('semantics', True)])
def test_malformed_metadata_on_any_copy_is_reported_without_counting(field, value):
    sentence = reading()
    target = sentence.alternatives[1].metadata['retention_stats' if field.startswith('syntax') else 'reader_phase_ms']
    target[field] = value
    result = reader_telemetry([sentence])
    assert result['instrumented_sentences'] == 0 and result['errors']
    assert not result['retention']['syntax_generated']


def test_disagreeing_metadata_is_not_arbitrarily_resolved_by_first_alternative():
    sentence = reading()
    sentence.alternatives[1].metadata['retention_stats']['syntax_generated'] = 20
    result = reader_telemetry([sentence])
    assert result['errors'] and result['instrumented_sentences'] == 0


def test_uninstrumented_readers_are_explicit_not_zero_work_measurements():
    sentence = SimpleNamespace(alternatives=(SimpleNamespace(metadata={}),))
    result = reader_telemetry([sentence])
    assert result['missing_sentences'] == 1 and result['instrumented_sentences'] == 0
    assert not result['errors']


def records():
    return tuple(Gold(str(i), 'a b', (Word(1, 'a', 'NOUN', 0, 'root'), Word(2, 'b', 'NOUN', 1, 'dep')),
                      {1: (0, 1), 2: (2, 3)}) for i in range(5))


def test_cohort_replay_keeps_order_and_exclusion_does_not_resample_it(tmp_path):
    report = tmp_path / 'cohort.json'
    report.write_text(json.dumps({'sentences': [{'sent_id': '3'}, {'sent_id': '1'}]}))
    chosen, _, selection = select_cohort(records(), sample=1, seed=99, max_tokens=20, cohort_report=report)
    assert [r.sent_id for r in chosen] == ['3', '1']
    assert selection['kind'] == 'exact report cohort replay'
    with pytest.raises(ValueError, match='absent or excluded'):
        select_cohort(records(), sample=1, seed=99, max_tokens=20, cohort_report=report, exclude_results=[report])
    sampled, eligible, metadata = select_cohort(records(), sample=99, seed=99, max_tokens=20, exclude_results=[report])
    assert {r.sent_id for r in sampled} == {'0', '2', '4'} and eligible == 3
    assert metadata['excluded_present_in_dataset'] == 2


def test_replayed_cohort_cannot_silently_drop_absent_or_duplicate_ids(tmp_path):
    report = tmp_path / 'cohort.json'
    report.write_text(json.dumps({'sentences': [{'sent_id': 'missing'}]}))
    with pytest.raises(ValueError, match='absent'):
        select_cohort(records(), sample=2, seed=0, max_tokens=20, cohort_report=report)
    report.write_text(json.dumps({'sentences': [{'sent_id': '1'}, {'sent_id': '1'}]}))
    with pytest.raises(ValueError, match='repeats'):
        select_cohort(records(), sample=2, seed=0, max_tokens=20, cohort_report=report)
