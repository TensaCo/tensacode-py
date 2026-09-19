"""Authored syntax and role priors isolate continuation, not learned accuracy."""
from copy import deepcopy

import pytest

from tensorcode.agent.understand import SentenceAlternative, SentenceContinuation
from tensorcode.language.deps_semantics import Reader
from test_learned_reader_alternatives import fixture_reader
from test_reader_retention import reader_with_counter
from test_semantic_frontier import source, reader as semantic_reader


def signature(candidate):
    m = candidate.metadata
    return (m['tags'], tuple(m['heads'].items()), m['semantic_candidate_index'])


def test_read_continues_all_families_without_replaying_syntax_or_first_candidates():
    reader = reader_with_counter(3)
    sentence, = reader.read('birds fly.')
    def forbidden(*args, **kwargs):
        raise AssertionError('continuation reran a decoder')
    reader._decode_segment = reader.segmenter.segment = forbidden
    continuation = sentence.continuation
    assert continuation.pending == 7
    initial = tuple(map(signature, sentence.alternatives))
    new = []
    while continuation.pending:
        batch = continuation.advance(max_expansions=1, max_candidates=1)
        assert batch.explored <= 1 and len(batch.alternatives) <= 1
        new.extend(batch.alternatives)
    signatures = initial + tuple(map(signature, new))
    assert len(signatures) == len(set(signatures)) == 12
    assert len({s[:2] for s in signatures}) == 4
    assert sentence.alternatives[0].metadata['semantic_candidate_index'] == 0
    assert continuation.advance(max_expansions=10, max_candidates=10).alternatives == ()


def test_deepcopy_isolates_real_semantic_cursors_and_source():
    words, tags, lemmas, heads, labels = source()
    family = SentenceAlternative(None, (), provenance='authored:test', metadata=dict(
        tokens=tuple(words), tags=tuple(tags), lemmas=tuple(lemmas), heads=heads, labels=labels,
        syntax_complete=True, token_anchors=({'index': 1, 'token': 'sit', 'char_span': (0, 3)},)))
    adapter = semantic_reader()
    cursor = adapter.start_candidates(words, tags, lemmas, heads, labels)
    first = cursor.advance(max_candidates=1)
    continuation = SentenceContinuation('sit on desk and wait on Tuesday', adapter, (),
        [dict(family=family, cursor=cursor, explored=first.explored, pending=first.pending, emitted=1)], [])
    copied = deepcopy(continuation)
    heads.clear()
    adapter.prepositions = {}
    result = continuation.advance(max_expansions=100, max_candidates=100)
    assert len(result.alternatives) == 3 and result.pending == 0
    assert copied.pending == first.pending
    assert copied.advance(max_expansions=100, max_candidates=100) == result
    result.alternatives[0].metadata['heads'].clear()
    assert copied.states[0]['family'].metadata['heads']


def test_quotation_and_global_source_anchors_survive_continuation():
    text = '  "birds fly."'
    sentence, = reader_with_counter(1).read(text)
    batch = sentence.continuation.advance(max_expansions=100, max_candidates=100)
    assert batch.alternatives and not batch.pending
    assert all(act.kind == 'mention' for alternative in batch.alternatives for act in alternative.acts)
    for alternative in batch.alternatives:
        for anchor in alternative.metadata['token_anchors']:
            start, end = anchor['char_span']
            assert text[start:end] == anchor['token']


@pytest.mark.parametrize('budget', [-1, True, 1.5, None])
def test_invalid_budgets_rejected_without_advancing(budget):
    sentence, = reader_with_counter(1).read('birds fly.')
    cursor = sentence.continuation
    before = cursor.pending
    with pytest.raises(ValueError):
        cursor.advance(max_expansions=budget, max_candidates=1)
    with pytest.raises(ValueError):
        cursor.advance(max_expansions=1, max_candidates=budget)
    assert cursor.pending == before


def test_zero_budgets_preserve_work_and_initial_placeholder_can_complete():
    reader = fixture_reader(2)
    reader.max_sentence_semantic_expansions = 0
    sentence, = reader.read('birds fly.')
    assert all(not a.acts for a in sentence.alternatives)
    cursor = sentence.continuation
    for limits in [(0, 100), (100, 0), (0, 0)]:
        result = cursor.advance(max_expansions=limits[0], max_candidates=limits[1])
        assert result.explored == 0 and result.alternatives == () and result.pending == 4
    result = cursor.advance(max_expansions=100, max_candidates=100)
    assert len(result.alternatives) == 4 and result.pending == 0


def test_act_projection_failure_rolls_back_cursor_and_delivers_on_retry(monkeypatch):
    import tensorcode.agent.understand as module
    sentence, = reader_with_counter(1).read('birds fly.')
    cursor = sentence.continuation
    expected = deepcopy(cursor).advance(max_expansions=100, max_candidates=100)
    before = cursor.pending
    original = module.acts_of
    calls = []
    def failing(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError('authored projection failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'acts_of', failing)
    with pytest.raises(RuntimeError, match='projection failure'):
        cursor.advance(max_expansions=100, max_candidates=100)
    assert cursor.pending == before
    monkeypatch.setattr(module, 'acts_of', original)
    assert cursor.advance(max_expansions=100, max_candidates=100) == expected
