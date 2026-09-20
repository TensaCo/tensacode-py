"""Goal interpretation links and revision admission survive reentrant copying."""

from types import SimpleNamespace

import pytest

from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.tasks import TaskLedger


def failed_attempt():
    return SimpleNamespace(status='failed', plan='attempted', receipt=None,
                           verified=False, reason='fixture failure', steps=())


def test_goal_links_are_revision_local_and_dependencies_still_default_to_preserved():
    workspace = InterpretationWorkspace()
    source = workspace.add_source('supplied goal evidence')
    group = workspace.create_group(source.id)
    candidate = workspace.propose(group.id, 'supplied goal')
    workspace.select(group.id, candidate.id, reason='authored fixture')
    dep = capture_dependency(workspace, group.id, basis=('authored goal correspondence',))
    ledger = TaskLedger()
    task = ledger.create('request', {'target': 'a'}, dependencies=(dep,), goal_interpretation_id=group.id)
    assert task.goal_interpretation_id == task.history[0].goal_interpretation_id == group.id
    ledger.record(task.id, failed_attempt())
    revised = ledger.revise(task.id, {'target': 'b'}, reason='explicit new goal', expected_revision=1)
    assert revised.goal_interpretation_id is None
    assert revised.history[0].goal_interpretation_id == group.id
    assert revised.history[1].goal_interpretation_id is None
    assert revised.dependencies == (dep,)
    assert len(revised.attempts) == 1 and revised.attempts[0].revision == 1
    adopted = ledger.revise(task.id, {'target': 'c'}, reason='adopt named interpretation',
                            goal_interpretation_id='interpretation:new', expected_revision=2)
    assert adopted.goal_interpretation_id == adopted.history[-1].goal_interpretation_id == 'interpretation:new'
    assert adopted.history[-2].goal_interpretation_id is None


@pytest.mark.parametrize('bad', ['', ' ', 1, False, (), object()])
def test_invalid_goal_links_are_rejected_without_a_revision(bad):
    ledger = TaskLedger()
    with pytest.raises(ValueError, match='goal_interpretation_id'):
        ledger.create('request', goal_interpretation_id=bad)
    assert len(ledger) == 0
    task = ledger.create('request', 'old')
    with pytest.raises(ValueError, match='goal_interpretation_id'):
        ledger.revise(task.id, 'new', reason='change', goal_interpretation_id=bad)
    assert ledger.current_revision(task.id) == 1


class CopyCallback:
    def __init__(self, action, *, state=None, trigger=1):
        self.action = action
        self.state = state if state is not None else {'copies': 0}
        self.trigger = trigger

    def __deepcopy__(self, memo):
        self.state['copies'] += 1
        if self.state['copies'] == self.trigger:
            self.action()
        return CopyCallback(self.action, state=self.state, trigger=self.trigger)


def test_stale_expected_revision_rejects_before_any_payload_copy():
    ledger = TaskLedger()
    task = ledger.create('request', 'old')
    ledger.revise(task.id, 'current', reason='advance')
    value = CopyCallback(lambda: pytest.fail('stale revision reached payload copying'))
    with pytest.raises(ValueError, match='stale expected'):
        ledger.revise(task.id, value, reason='stale adoption', expected_revision=1)
    assert value.state['copies'] == 0
    assert ledger.get(task.id).goal == 'current'


@pytest.mark.parametrize('trigger', [1, 3])
@pytest.mark.parametrize('expected', [None, 1])
def test_reentrant_revision_during_input_or_return_copy_is_never_overwritten(trigger, expected):
    ledger = TaskLedger()
    task = ledger.create('request', 'original', goal_interpretation_id='interpretation:old')
    value = CopyCallback(lambda: ledger.revise(task.id, 'reentrant winner', reason='new evidence',
                                              goal_interpretation_id='interpretation:winner'), trigger=trigger)
    with pytest.raises(ValueError, match='revision'):
        ledger.revise(task.id, value, reason='outer adoption', expected_revision=expected,
                      goal_interpretation_id='interpretation:stale')
    latest = ledger.get(task.id)
    assert latest.revision == 2 and latest.goal == 'reentrant winner'
    assert latest.goal_interpretation_id == 'interpretation:winner'
    assert len(latest.history) == 2


@pytest.mark.parametrize('trigger', [1, 3])
def test_same_revision_attempt_added_during_copy_survives_adoption(trigger):
    ledger = TaskLedger()
    task = ledger.create('request', 'original')
    value = CopyCallback(lambda: ledger.record(task.id, failed_attempt(), revision=1), trigger=trigger)
    revised = ledger.revise(task.id, value, reason='adopt goal', expected_revision=1,
                            goal_interpretation_id='interpretation:adopted')
    latest = ledger.get(task.id)
    assert revised.revision == latest.revision == 2
    assert latest.goal_interpretation_id == 'interpretation:adopted'
    assert len(latest.attempts) == 1
    assert latest.attempts[0].revision == 1 and latest.attempts[0].plan == 'attempted'


@pytest.mark.parametrize('expected', [False, 0, -1, '1'])
def test_invalid_compare_and_swap_version_is_rejected(expected):
    ledger = TaskLedger()
    task = ledger.create('request', 'old')
    with pytest.raises(ValueError, match='expected_revision'):
        ledger.revise(task.id, 'new', reason='adopt', expected_revision=expected)
    assert ledger.current_revision(task.id) == 1


@pytest.mark.parametrize('verdict', [False, None, 1])
def test_commit_guard_requires_exact_true_and_preserves_previous_revision(verdict):
    ledger = TaskLedger()
    task = ledger.create('request', 'old')
    with pytest.raises(ValueError, match='before_commit guard rejected'):
        ledger.revise(task.id, 'new', reason='adopt', expected_revision=1,
                      before_commit=lambda: verdict)
    assert ledger.get(task.id).goal == 'old'
    assert ledger.current_revision(task.id) == 1


def test_commit_guard_runs_after_all_copies_and_before_revision_write():
    ledger = TaskLedger()
    task = ledger.create('request', 'old')
    value = CopyCallback(lambda: None)
    observed = []
    def guard():
        observed.append((value.state['copies'], ledger.current_revision(task.id)))
        return True
    revised = ledger.revise(task.id, value, reason='adopt', expected_revision=1, before_commit=guard)
    assert observed == [(value.state['copies'], 1)]
    assert revised.revision == 2 and value.state['copies'] > 0


def test_commit_guard_cannot_overwrite_revision_it_changes_reentrantly():
    ledger = TaskLedger()
    task = ledger.create('request', 'old')
    def guard():
        ledger.revise(task.id, 'guard winner', reason='newer interpretation')
        return True
    with pytest.raises(ValueError, match='stale expected'):
        ledger.revise(task.id, 'outer', reason='adopt', expected_revision=1, before_commit=guard)
    assert ledger.get(task.id).goal == 'guard winner'
    assert ledger.current_revision(task.id) == 2


def test_commit_guard_same_revision_attempt_is_preserved():
    ledger = TaskLedger()
    task = ledger.create('request', 'old')
    def guard():
        ledger.record(task.id, failed_attempt(), revision=1)
        return True
    ledger.revise(task.id, 'new', reason='adopt', expected_revision=1, before_commit=guard)
    latest = ledger.get(task.id)
    assert latest.goal == 'new' and len(latest.attempts) == 1
    assert latest.attempts[0].revision == 1
