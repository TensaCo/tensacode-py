"""Task bindings preserve explicit interpretation authority without creating it."""
from dataclasses import FrozenInstanceError, replace

import pytest

from tensorcode.agent.interpretation import ContinuationStatus, InterpretationWorkspace
from tensorcode.agent.task_dependencies import capture_dependency, validate_dependency, validate_dependencies
from tensorcode.outcomes import Unknown


BASIS = ('Authored mapping from this selected meaning to the supplied goal',)


def setup():
    workspace = InterpretationWorkspace()
    source = workspace.add_source('ambiguous input')
    evidence = workspace.add_source('explicit caller correspondence')
    group = workspace.create_group(source.id)
    candidate = workspace.propose(group.id, {'meaning': ['a']})
    rival = workspace.propose(group.id, {'meaning': ['b']})
    workspace.select(group.id, candidate.id, reason='explicit supplied selection')
    return workspace, group.id, candidate.id, rival.id, evidence.id


def test_capture_is_detached_and_preserves_source_selection_and_authored_basis():
    workspace, group_id, candidate, rival, evidence = setup()
    before = workspace.get(group_id)
    dependency = capture_dependency(workspace, group_id, basis=BASIS, evidence_ids=(evidence,))
    assert dependency.source_id == before.source_id
    assert dependency.candidate_id == candidate
    assert dependency.candidate_ids == (candidate, rival)
    assert dependency.basis == BASIS and dependency.evidence_ids == (evidence,)
    assert dependency.continuation == ContinuationStatus(False, 0, 0)
    assert workspace.get(group_id) == before
    assert validate_dependency(workspace, dependency) is True
    with pytest.raises(FrozenInstanceError):
        dependency.revision = 99
    before.selected.payload['meaning'].clear()
    assert validate_dependency(workspace, dependency) is True


@pytest.mark.parametrize('operation', ['unset', 'reselect', 'reject'])
def test_changed_selection_or_revision_invalidates_without_restoring_authority(operation):
    workspace, group_id, candidate, rival, _ = setup()
    dependency = capture_dependency(workspace, group_id, basis=BASIS)
    if operation == 'unset':
        workspace.unset(group_id, reason='unresolved')
    elif operation == 'reselect':
        workspace.select(group_id, rival, reason='different reading')
    else:
        workspace.reject(group_id, candidate, reason='counterevidence')
    before = workspace.get(group_id)
    result = validate_dependency(workspace, dependency)
    assert isinstance(result, Unknown) and result.reason == 'interpretation_dependency_changed'
    assert workspace.get(group_id) == before


def test_new_rival_invalidates_binding_even_without_revision_change():
    workspace, group_id, *_ = setup()
    dependency = capture_dependency(workspace, group_id, basis=BASIS)
    workspace.propose(group_id, 'new rival')
    assert workspace.get(group_id).revision == dependency.revision
    assert isinstance(validate_dependency(workspace, dependency), Unknown)


class Cursor:
    def __init__(self, pending):
        self.pending = pending
    def advance(self, **kwargs):
        raise AssertionError('dependency capture must not run reader search')


def test_pending_work_blocks_capture_and_new_frontier_invalidates_old_capture():
    workspace, group_id, *_ = setup()
    dependency = capture_dependency(workspace, group_id, basis=BASIS)
    workspace.attach_continuation(group_id, Cursor(1))
    assert isinstance(validate_dependency(workspace, dependency), Unknown)
    with pytest.raises(ValueError, match='pending'):
        capture_dependency(workspace, group_id, basis=BASIS)


def test_attaching_exhausted_cursor_changes_generation_and_invalidates_old_capture():
    workspace, group_id, *_ = setup()
    dependency = capture_dependency(workspace, group_id, basis=BASIS)
    workspace.attach_continuation(group_id, Cursor(0))
    assert isinstance(validate_dependency(workspace, dependency), Unknown)
    fresh = capture_dependency(workspace, group_id, basis=BASIS)
    assert fresh.continuation == ContinuationStatus(True, 0, 1)
    assert validate_dependency(workspace, fresh) is True


def test_unselected_group_and_missing_evidence_cannot_be_bound():
    workspace, group_id, *_ = setup()
    with pytest.raises(KeyError):
        capture_dependency(workspace, group_id, basis=BASIS, evidence_ids=('missing',))
    workspace.unset(group_id, reason='unresolved')
    with pytest.raises(ValueError, match='selected'):
        capture_dependency(workspace, group_id, basis=BASIS)


@pytest.mark.parametrize('basis', [(), '', ('',), ['authored'], (True,)])
def test_basis_requires_explicit_nonempty_tuple(basis):
    workspace, group_id, *_ = setup()
    with pytest.raises(ValueError):
        capture_dependency(workspace, group_id, basis=basis)


def test_forged_fields_and_foreign_workspace_fail_closed():
    workspace, group_id, *_ = setup()
    dependency = capture_dependency(workspace, group_id, basis=BASIS)
    with pytest.raises(ValueError):
        replace(dependency, revision=True)
    with pytest.raises(ValueError):
        replace(dependency, continuation=ContinuationStatus(False, False, 0))
    with pytest.raises(ValueError):
        replace(dependency, candidate_ids=(dependency.candidate_id, dependency.candidate_id))
    object.__setattr__(dependency, 'revision', True)
    assert validate_dependency(workspace, dependency).reason == 'invalid_interpretation_dependency'
    fresh = capture_dependency(workspace, group_id, basis=BASIS)
    assert isinstance(validate_dependency(InterpretationWorkspace(), fresh), Unknown)


def test_batch_checks_every_dependency_and_requires_tuple():
    workspace, group_id, *_ = setup()
    first = capture_dependency(workspace, group_id, basis=BASIS)
    group2 = workspace.create_group(first.source_id)
    candidate = workspace.propose(group2.id, 'second meaning')
    workspace.select(group2.id, candidate.id, reason='explicit supplied selection')
    second = capture_dependency(workspace, group2.id, basis=BASIS)
    assert validate_dependencies(workspace, (first, second)) is True
    assert validate_dependencies(workspace, ()) is True
    assert isinstance(validate_dependencies(workspace, [first]), Unknown)
    workspace.unset(group2.id, reason='counterevidence')
    assert isinstance(validate_dependencies(workspace, (first, second)), Unknown)


def test_evidence_snapshot_callback_cannot_capture_stale_selection(monkeypatch):
    workspace, group_id, _, rival, evidence = setup()
    original = workspace.get_source
    fired = False
    def read(source_id):
        nonlocal fired
        source = original(source_id)
        if source_id == evidence and not fired:
            fired = True
            workspace.select(group_id, rival, reason='correction while copying evidence')
        return source
    monkeypatch.setattr(workspace, 'get_source', read)
    with pytest.raises(ValueError, match='changed while'):
        capture_dependency(workspace, group_id, basis=BASIS, evidence_ids=(evidence,))


def test_later_dependency_evidence_callback_invalidates_earlier_dependency(monkeypatch):
    workspace, group_id, _, rival, evidence = setup()
    first = capture_dependency(workspace, group_id, basis=BASIS)
    other = workspace.create_group(first.source_id)
    selected = workspace.propose(other.id, 'other meaning')
    workspace.select(other.id, selected.id, reason='explicit supplied selection')
    second = capture_dependency(workspace, other.id, basis=BASIS, evidence_ids=(evidence,))
    original = workspace.get_source
    changed = False
    def read(source_id):
        nonlocal changed
        result = original(source_id)
        if source_id == evidence and not changed:
            changed = True
            workspace.select(group_id, rival, reason='later dependency evidence corrected earlier choice')
        return result
    monkeypatch.setattr(workspace, 'get_source', read)
    assert isinstance(validate_dependencies(workspace, (first, second)), Unknown)


def test_final_batch_check_cannot_trigger_second_pass_evidence_revocation(monkeypatch):
    workspace, group_id, candidate, rival, evidence = setup()
    first = capture_dependency(workspace, group_id, basis=BASIS)
    other = workspace.create_group(first.source_id)
    selected = workspace.propose(other.id, 'other meaning')
    workspace.select(other.id, selected.id, reason='explicit supplied selection')
    second = capture_dependency(workspace, other.id, basis=BASIS, evidence_ids=(evidence,))
    original = workspace.get_source
    reads = 0
    def read(source_id):
        nonlocal reads
        result = original(source_id)
        if source_id == evidence:
            reads += 1
            if reads == 2:
                workspace.select(group_id, rival, reason='revocation on last read of repeated validation')
        return result
    monkeypatch.setattr(workspace, 'get_source', read)
    result = validate_dependencies(workspace, (first, second))
    # Old fixed two-pass validation returned True after its final callback revoked
    # the first binding. The final comparison now executes no source callbacks.
    assert result is True
    assert reads == 1
    assert workspace.get(group_id).selected_id == candidate


def test_comparison_basis_does_not_copy_payloads_or_query_cursor_pending(monkeypatch):
    workspace, group_id, candidate, rival, _ = setup()
    workspace.attach_continuation(group_id, Cursor(0))
    expected = workspace.comparison_basis(group_id)
    def fail(*args, **kwargs):
        raise AssertionError('final identity comparison must not invoke callbacks')
    monkeypatch.setattr(workspace, 'get', fail)
    monkeypatch.setattr(workspace, 'get_source', fail)
    monkeypatch.setattr(workspace, '_pending', fail)
    monkeypatch.setattr('tensorcode.agent.interpretation.deepcopy', fail)
    assert workspace.comparison_basis(group_id) == expected
    assert expected[2:5] == (candidate, (candidate, rival), False)
