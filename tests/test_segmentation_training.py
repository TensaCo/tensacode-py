"""Evaluation/training data contracts; fixture candidates are authored, not learned."""
from types import SimpleNamespace

import pytest

from eval.parsing.span_evaluation import Gold, Word
from eval.parsing.train_segmentation import evaluate, old_tokenizer_spans, select, validate_spans


@pytest.mark.parametrize('spans', [((0, 1),), ((0, 2), (1, 2)), ((False, 2),), ((0, 3),)])
def test_candidate_spans_cannot_drop_duplicate_or_fabricate_source(spans):
    with pytest.raises(ValueError):
        validate_spans('ab', spans)


def test_source_spans_keep_clitics_and_whitespace_boundaries_without_normalizing():
    assert validate_spans("I'm here", ((0, 1), (1, 3), (4, 8))) == ((0, 1), (1, 3), (4, 8))
    with pytest.raises(ValueError):
        validate_spans("I'm here", ((0, 8),))


def test_reference_tokenizer_preserves_quoted_surface_in_evaluation_only():
    text = 'say "hello there"'
    spans = old_tokenizer_spans(text)
    assert [text[slice(*s)] for s in spans] == ['say', '"hello there"']


def test_oracle_never_unions_boundaries_from_different_candidates():
    gold = Gold('fixture', 'abcd', tuple(Word(i + 1, c, 'NOUN', 0, 'root') for i, c in enumerate('abcd')),
                {i + 1: (i, i + 1) for i in range(4)})
    class Model:
        def segment(self, text, **kwargs):
            return SimpleNamespace(candidates=(SimpleNamespace(spans=((0, 1), (1, 4))),
                                               SimpleNamespace(spans=((0, 3), (3, 4)))),
                                   expansions=4, truncated=True, complete=False, reason='beam_pruned')
    report = evaluate(Model(), (gold,), beam_width=4, max_candidates=3, max_expansions=100, compare_old=False)
    assert report['metrics']['oracle']['matched_spans'] == 1
    assert report['metrics']['oracle']['recall'] == .25
    assert report['exact_candidate_recall'] == 0


def test_dataset_alignment_error_survives_and_cannot_count_as_exact():
    gold = Gold('broken', 'a', (Word(1, 'a', 'NOUN', 0, 'root'),), {}, error='source mismatch')
    report = evaluate(object(), (gold,), beam_width=4, max_candidates=3, max_expansions=100)
    assert report['search']['error_inputs'] == 1
    assert report['metrics']['top']['exact_sentences'] == 0
    assert report['sentences'][0]['error'] == 'source mismatch'


def test_deterministic_selection_does_not_mutate_or_drop_alignment_errors():
    records = tuple(Gold(str(i), '', (), {}, error='fixture') for i in range(6))
    assert select(records, 0, 12) is records
    assert select(records, 3, 12) == select(records, 3, 12)
    assert len(select(records, 99, 12)) == 6
    with pytest.raises(ValueError):
        select(records, -1, 12)
