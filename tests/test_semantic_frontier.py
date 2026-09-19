"""Resumption preserves alternative occurrence bindings without repeating work."""
import pytest

from tensorcode.language.deps_semantics import Reader


def source():
    words = ['sit', 'on', 'desk', 'and', 'wait', 'on', 'Tuesday']
    return (words, ['VERB', 'ADP', 'NOUN', 'CCONJ', 'VERB', 'ADP', 'PROPN'], words.copy(),
            dict(enumerate([0, 3, 1, 5, 1, 7, 5], 1)),
            dict(enumerate(['root', 'case', 'obl', 'cc', 'conj', 'case', 'obl'], 1)))


def reader():
    return Reader({'on': [('location', -0.2), ('time', -1.7)]},
                  preposition_provenance='authored:frontier-test')


def test_many_small_advances_equal_one_complete_search_without_replay(monkeypatch):
    adapter = reader()
    expected = adapter.read_candidates(*source())
    calls = []
    original = Reader._read

    def counting(self, *args):
        calls.append(tuple(self._role_bindings.items()))
        return original(self, *args)

    monkeypatch.setattr(Reader, '_read', counting)
    frontier = adapter.start_candidates(*source())
    assert calls == []
    actual = []
    for step in range(1, 8):
        result = frontier.advance(max_expansions=1, max_candidates=1)
        assert result.explored == step
        actual.extend(result.candidates)
        assert result.truncated == bool(result.pending)
    assert tuple(actual) == expected.candidates
    assert result.complete and result.pending == 0
    assert len(calls) == len(set(calls)) == expected.explored == 7
    exhausted = frontier.advance(max_expansions=100, max_candidates=100)
    assert exhausted.candidates == () and exhausted.explored == 7 and exhausted.complete
    assert len(calls) == 7


def test_candidate_cap_retains_every_other_branch():
    adapter = reader()
    frontier = adapter.start_candidates(*source())
    first = frontier.advance(max_candidates=1)
    assert len(first.candidates) == 1 and first.pending == 3
    rest = frontier.advance(max_candidates=10)
    assert first.candidates + rest.candidates == adapter.read_candidates(*source()).candidates
    assert rest.complete
    assignments = {tuple((choice.dependent_token, choice.role) for choice in row.choices)
                   for row in first.candidates + rest.candidates}
    assert len(assignments) == 4
    assert ((3, 'location'), (7, 'time')) in assignments
    assert ((3, 'time'), (7, 'location')) in assignments


def test_zero_budget_inspection_preserves_pending_without_claims():
    frontier = reader().start_candidates(*source())
    for budgets in ({'max_expansions': 0}, {'max_candidates': 0}):
        result = frontier.advance(**budgets)
        assert result.candidates == () and result.explored == 0 and result.pending == 1
        assert result.truncated and not result.complete
    result = frontier.advance(max_expansions=1)
    assert result.explored == 1 and result.pending == 2 and result.candidates == ()
    assert len(frontier.advance().candidates) == 4


def test_frontier_detaches_syntax_and_reader_configuration():
    adapter, inputs = reader(), source()
    expected = adapter.read_candidates(*inputs)
    frontier = adapter.start_candidates(*inputs)
    for value in inputs:
        value.clear()
    adapter.prepositions = {}
    adapter.preposition_provenance = 'changed'
    assert frontier.advance() == expected


def test_unknown_choice_is_a_completed_unresolved_candidate():
    frontier = Reader({}).start_candidates(*source())
    result = frontier.advance()
    assert result.complete and len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.meanings == ()
    assert candidate.unresolved[0].dependent_token == 3
    assert frontier.advance().candidates == ()


def test_empty_source_has_one_explicit_empty_projection():
    frontier = reader().start_candidates([], [], [], {}, {})
    assert frontier.advance(max_expansions=0).pending == 1
    result = frontier.advance()
    assert result.complete and result.explored == 1
    assert len(result.candidates) == 1 and result.candidates[0].meanings == ()


@pytest.mark.parametrize('budget', [-1, True, 1.2])
def test_invalid_advance_budget_does_not_consume_frontier(budget):
    frontier = reader().start_candidates(*source())
    with pytest.raises(ValueError):
        frontier.advance(max_expansions=budget)
    assert frontier.advance(max_expansions=0).explored == 0


def test_projection_error_does_not_discard_pending_branch(monkeypatch):
    frontier = reader().start_candidates(*source())
    original = Reader._read
    def fail(self, *args):
        raise RuntimeError('projection failed')
    monkeypatch.setattr(Reader, '_read', fail)
    with pytest.raises(RuntimeError, match='projection failed'):
        frontier.advance()
    unchanged = frontier.advance(max_expansions=0)
    assert unchanged.pending == 1 and unchanged.explored == 0
    monkeypatch.setattr(Reader, '_read', original)
    assert len(frontier.advance().candidates) == 4


def test_completed_prefix_survives_later_failure_without_replay_or_duplication(monkeypatch):
    words = ['sit', 'on', 'desk']
    inputs = (words, ['VERB', 'ADP', 'NOUN'], words,
              {1: 0, 2: 3, 3: 1}, {1: 'root', 2: 'case', 3: 'obl'})
    adapter = reader()
    expected = adapter.read_candidates(*inputs).candidates
    frontier = adapter.start_candidates(*inputs)
    original = Reader._read
    calls = []
    fail = True

    def sometimes_fails(self, *args):
        roles = tuple(choice.role for choice in self._role_bindings.values())
        calls.append(roles)
        if roles == ('time',) and fail:
            raise RuntimeError('second branch failed')
        return original(self, *args)

    monkeypatch.setattr(Reader, '_read', sometimes_fails)
    with pytest.raises(RuntimeError, match='second branch failed'):
        frontier.advance(max_candidates=2)
    inspected = frontier.advance(max_candidates=0)
    assert inspected.explored == 2 and inspected.pending == 2
    assert inspected.candidates == () and not inspected.complete
    # Repeated failures cannot discard or duplicate the completed prefix.
    with pytest.raises(RuntimeError):
        frontier.advance(max_candidates=2)
    prefix = frontier.advance(max_expansions=0, max_candidates=1)
    assert prefix.candidates == expected[:1]
    assert prefix.explored == 2 and prefix.pending == 1
    assert calls.count(('location',)) == 1
    fail = False
    suffix = frontier.advance(max_candidates=1)
    assert prefix.candidates + suffix.candidates == expected
    assert suffix.explored == 3 and suffix.complete and suffix.pending == 0
    assert calls.count(('location',)) == 1
    assert frontier.advance().candidates == ()


def test_checkpoint_restores_pending_work_without_replaying_completed_branches(monkeypatch):
    from copy import deepcopy

    adapter = reader()
    expected = adapter.read_candidates(*source())
    frontier = adapter.start_candidates(*source())
    prefix = frontier.advance(max_candidates=1)
    snapshot = frontier.snapshot()
    original = Reader._read
    calls = []

    def counting(self, *args):
        calls.append(tuple(self._role_bindings.items()))
        return original(self, *args)

    monkeypatch.setattr(Reader, '_read', counting)
    restored = snapshot.restore()
    assert calls == []
    suffix = restored.advance()
    assert prefix.candidates + suffix.candidates == expected.candidates
    assert len(calls) == expected.explored - prefix.explored
    assert suffix.explored == expected.explored
    # Reading one fork does not consume another or the checkpoint.
    assert snapshot.restore().advance() == suffix
    assert deepcopy(frontier).advance() == suffix
    assert deepcopy(snapshot).restore().advance() == suffix
    assert frontier.advance() == suffix


def test_checkpoint_and_reader_copy_preserve_captured_configuration():
    from copy import deepcopy

    adapter = reader()
    expected = adapter.read_candidates(*source())
    copied = deepcopy(adapter)
    frontier = adapter.start_candidates(*source())
    frontier.advance(max_expansions=1)
    snapshot = frontier.snapshot()
    adapter.prepositions = {'on': [('instrument', 0.0)]}
    adapter.preposition_provenance = 'replacement-policy'
    assert copied.read_candidates(*source()) == expected
    assert snapshot.restore().advance() == expected


def test_checkpoint_detaches_mutable_ready_meanings_after_failure(monkeypatch):
    from tensorcode.language.semantics import Frame

    frontier = reader().start_candidates(*source())
    original = Reader._read
    completed = Frame('test', {'object': {'name': 'original'}})
    calls = []
    fail = True

    def custom(self, *args):
        roles = tuple(choice.role for choice in self._role_bindings.values())
        calls.append(roles)
        if len(roles) == 2:
            if roles == ('location', 'location'):
                return [completed]
            if fail:
                raise RuntimeError('after a completed candidate')
        return original(self, *args)

    monkeypatch.setattr(Reader, '_read', custom)
    with pytest.raises(RuntimeError):
        frontier.advance(max_candidates=4)
    snapshot = frontier.snapshot()
    captured_explored = snapshot.explored
    completed.roles['object']['name'] = 'mutated original'
    first = snapshot.restore()
    ready = first.advance(max_expansions=0, max_candidates=1)
    assert ready.explored == captured_explored
    assert ready.candidates[0].meanings[0].roles['object']['name'] == 'original'
    ready.candidates[0].meanings[0].roles['object']['name'] = 'mutated fork'
    independent = snapshot.restore().advance(max_expansions=0, max_candidates=1)
    assert independent.candidates[0].meanings[0].roles['object']['name'] == 'original'
    fail = False
    remainder = first.advance()
    assert remainder.complete and len(remainder.candidates) == 3
    assert calls.count(('location', 'location')) == 1
