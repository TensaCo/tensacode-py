"""Continuation publication preserves evidence, decisions, and retryable work."""
from dataclasses import FrozenInstanceError, dataclass

import pytest

from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.understand import SentenceAlternative


@dataclass
class Batch:
    alternatives: tuple
    explored: int
    pending: int


class Cursor:
    def __init__(self, *, fail=False, callback=None, progress_only=False):
        self.index = 0
        self.fail = fail
        self.callback = callback
        self.progress_only = progress_only

    @property
    def pending(self):
        return 2 - self.index

    def advance(self, *, max_expansions, max_candidates):
        if not max_expansions or not max_candidates or self.index == 2:
            return Batch((), 0, 2 - self.index)
        self.index += 1
        if self.callback:
            self.callback()
        if self.fail:
            raise RuntimeError('projection failed after progress')
        alternatives = () if self.progress_only else (SentenceAlternative(
            None, (), provenance='supplied cursor', metadata={'index': self.index}),)
        return Batch(alternatives, 1, 2 - self.index)


def setup(cursor=None):
    workspace = InterpretationWorkspace()
    source = workspace.add_source('original evidence')
    group = workspace.create_group(source.id)
    first = workspace.propose(group.id, SentenceAlternative(None, ()))
    if cursor is not None:
        workspace.attach_continuation(group.id, cursor)
    return workspace, group.id, first


def test_expansion_preserves_ids_and_withdraws_selection_without_rewriting_history():
    workspace, group_id, first = setup(Cursor())
    chosen = workspace.select(group_id, first.id, reason='supplied initial choice')
    result = workspace.expand(group_id, max_expansions=1, max_candidates=1)
    assert result.explored == result.pending == 1
    assert result.group.candidates[0] == first
    assert result.group.selected_id is None
    assert result.group.history[:-1] == chosen.history
    assert result.group.history[-1].operation == 'expand'
    assert result.candidate_ids == (result.group.candidates[1].id,)
    next_result = workspace.expand(group_id, max_expansions=1, max_candidates=1)
    assert next_result.pending == 0
    assert next_result.group.candidates[:2] == result.group.candidates
    assert workspace.get_source(result.group.source_id).text == 'original evidence'


def test_cursor_input_snapshots_and_result_payloads_are_isolated():
    supplied = Cursor()
    workspace, group_id, _ = setup(supplied)
    supplied.index = 2
    detached = workspace.get_continuation(group_id)
    detached.index = 2
    result = workspace.expand(group_id, max_expansions=1, max_candidates=1)
    assert result.group.candidates[-1].payload.metadata == {'index': 1}
    result.group.candidates[-1].payload.metadata.clear()
    assert workspace.get(group_id).candidates[-1].payload.metadata == {'index': 1}


def test_failed_projection_does_not_consume_owned_cursor_or_change_group():
    workspace, group_id, _ = setup(Cursor(fail=True))
    before = workspace.get(group_id)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='projection failed'):
            workspace.expand(group_id, max_expansions=1, max_candidates=1)
        assert workspace.get(group_id) == before
        assert workspace.get_continuation(group_id).index == 0


@pytest.mark.parametrize('mutation', ['propose', 'select'])
def test_reentrant_mutation_is_preserved_but_stale_batch_is_not_published(mutation):
    workspace, group_id, first = setup()
    def callback():
        if mutation == 'propose':
            workspace.propose(group_id, 'external candidate')
        else:
            workspace.select(group_id, first.id, reason='external choice')
    workspace.attach_continuation(group_id, Cursor(callback=callback))
    with pytest.raises(RuntimeError, match='changed during expansion'):
        workspace.expand(group_id, max_expansions=1, max_candidates=1)
    group = workspace.get(group_id)
    assert len(group.candidates) == (2 if mutation == 'propose' else 1)
    assert workspace.get_continuation(group_id).index == 0


def test_zero_budget_inspects_pending_without_revision_or_selection_change():
    workspace, group_id, first = setup(Cursor())
    before = workspace.select(group_id, first.id, reason='initial choice')
    result = workspace.expand(group_id, max_expansions=0, max_candidates=0)
    assert result.group == before
    assert result.pending == 2 and result.explored == 0 and result.candidate_ids == ()


def test_progress_without_candidates_records_revision_but_preserves_selection():
    workspace, group_id, first = setup(Cursor(progress_only=True))
    before = workspace.select(group_id, first.id, reason='initial choice')
    result = workspace.expand(group_id, max_expansions=1, max_candidates=1)
    assert result.group.selected_id == first.id
    assert result.group.revision == before.revision + 1
    assert result.group.candidates == before.candidates


@pytest.mark.parametrize('budget', [-1, True, 1.5, None])
def test_invalid_budget_fails_before_cursor_work(budget):
    workspace, group_id, _ = setup(Cursor())
    for keyword in ('max_expansions', 'max_candidates'):
        args = {'max_expansions': 1, 'max_candidates': 1, keyword: budget}
        with pytest.raises(ValueError, match='nonnegative integer'):
            workspace.expand(group_id, **args)
    assert workspace.get_continuation(group_id).index == 0


def test_missing_and_duplicate_continuations_fail_explicitly():
    workspace, group_id, _ = setup()
    assert workspace.get_continuation(group_id) is None
    with pytest.raises(ValueError, match='no continuation'):
        workspace.expand(group_id, max_expansions=1, max_candidates=1)
    with pytest.raises(TypeError, match='advance'):
        workspace.attach_continuation(group_id, object())
    workspace.attach_continuation(group_id, Cursor())
    with pytest.raises(ValueError, match='already has'):
        workspace.attach_continuation(group_id, Cursor())


def test_candidate_copy_failure_leaves_cursor_and_candidates_retryable():
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError('candidate copy failed')

    class BadOutput(Cursor):
        def advance(self, **budgets):
            batch = super().advance(**budgets)
            batch.alternatives[0].metadata['bad'] = Uncopyable()
            return batch

    workspace, group_id, _ = setup(BadOutput())
    before = workspace.get(group_id)
    with pytest.raises(RuntimeError, match='candidate copy failed'):
        workspace.expand(group_id, max_expansions=1, max_candidates=1)
    assert workspace.get(group_id) == before
    assert workspace.get_continuation(group_id).index == 0


@pytest.mark.parametrize('field,value', [('explored', True), ('pending', -1), ('explored', 2)])
def test_invalid_batch_cannot_commit_cursor_progress(field, value):
    class InvalidOutput(Cursor):
        def advance(self, **budgets):
            batch = super().advance(**budgets)
            setattr(batch, field, value)
            return batch

    workspace, group_id, _ = setup(InvalidOutput())
    before = workspace.get(group_id)
    with pytest.raises(ValueError):
        workspace.expand(group_id, max_expansions=1, max_candidates=1)
    assert workspace.get(group_id) == before
    assert workspace.get_continuation(group_id).index == 0


def test_status_distinguishes_absent_attached_and_exhausted_without_copying_cursor(monkeypatch):
    workspace, group_id, _ = setup()
    absent = workspace.continuation_status(group_id)
    assert (absent.available, absent.pending, absent.generation) == (False, 0, 0)
    workspace.attach_continuation(group_id, Cursor())
    attached = workspace.continuation_status(group_id)
    assert (attached.available, attached.pending, attached.generation) == (True, 2, 1)
    with pytest.raises(FrozenInstanceError):
        attached.pending = 0
    workspace.expand(group_id, max_expansions=1, max_candidates=1)
    workspace.expand(group_id, max_expansions=1, max_candidates=1)
    def fail_copy(*args):
        raise AssertionError('status must not copy cursor')
    monkeypatch.setattr('tensorcode.agent.interpretation.deepcopy', fail_copy)
    exhausted = workspace.continuation_status(group_id)
    assert (exhausted.available, exhausted.pending, exhausted.generation) == (True, 0, 3)
    assert attached.pending == 2  # Earlier scalar snapshots do not change.


def test_status_generation_preserves_noops_and_failed_work():
    workspace, group_id, _ = setup(Cursor())
    before = workspace.continuation_status(group_id)
    workspace.expand(group_id, max_expansions=0, max_candidates=0)
    assert workspace.continuation_status(group_id) == before
    failed, failed_id, _ = setup(Cursor(fail=True))
    before_failure = failed.continuation_status(failed_id)
    with pytest.raises(RuntimeError, match='projection failed'):
        failed.expand(failed_id, max_expansions=1, max_candidates=1)
    assert failed.continuation_status(failed_id) == before_failure


def test_generation_detects_progress_with_unchanged_pending_count():
    class ConstantPending(Cursor):
        @property
        def pending(self):
            return 2
        def advance(self, **budgets):
            batch = super().advance(**budgets)
            return Batch(batch.alternatives, batch.explored, self.pending)
    workspace, group_id, _ = setup(ConstantPending(progress_only=True))
    before = workspace.continuation_status(group_id)
    workspace.expand(group_id, max_expansions=1, max_candidates=1)
    after = workspace.continuation_status(group_id)
    assert after.pending == before.pending
    assert after.generation == before.generation + 1


def test_generation_detects_pending_change_without_group_revision_change():
    class FinishesEmpty(Cursor):
        def advance(self, **budgets):
            self.index = 2
            return Batch((), 0, 0)
    workspace, group_id, _ = setup(FinishesEmpty())
    before = workspace.get(group_id)
    status = workspace.continuation_status(group_id)
    workspace.expand(group_id, max_expansions=1, max_candidates=1)
    assert workspace.get(group_id) == before
    assert workspace.continuation_status(group_id).generation == status.generation + 1


@pytest.mark.parametrize('pending', [True, -1, 1.2, None])
def test_invalid_pending_status_is_rejected_without_attachment(pending):
    class InvalidPending(Cursor):
        pass
    InvalidPending.pending = pending
    workspace, group_id, _ = setup()
    with pytest.raises(ValueError, match='pending must be a nonnegative integer'):
        workspace.attach_continuation(group_id, InvalidPending())
    assert not workspace.continuation_status(group_id).available
