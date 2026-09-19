"""Late outcomes belong to the goal revision dispatched, not the newest goal."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tensorcode.agent.tasks import StepAttempt, TaskLedger
from tensorcode.outcomes import Receipt
from tensorcode.agent.plugin import Call


def outcome(status='done', **kwargs):
    return SimpleNamespace(status=status, plan=kwargs.get('plan'), receipt=kwargs.get('receipt'),
                           verified=kwargs.get('verified'), reason=kwargs.get('reason', ''),
                           steps=kwargs.get('steps', ()))


def test_delayed_old_success_preserves_new_ready_goal_and_applied_receipts():
    ledger = TaskLedger()
    initial = ledger.create('original request', {'target': 'old'})
    revised = ledger.revise(initial.id, {'target': 'new'}, reason='explicit correction')
    call = Call('fixture', 'write', (('target', 'old'),))
    receipt = Receipt(call, 'applied')
    step = StepAttempt('write-old', call, receipt, True)
    recorded = ledger.record(initial.id, outcome(receipt=receipt, steps=(step,), verified=True), revision=1)
    assert recorded.status == 'ready' and recorded.revision == 2
    assert recorded.goal == revised.goal and recorded.source == initial.source
    assert recorded.revisions == revised.revisions
    assert recorded.attempts[0].revision == 1
    assert recorded.attempts[0].receipt == receipt
    assert recorded.attempts[0].steps == (step,)


def test_new_done_status_survives_old_failure_or_success_arrival():
    for status in ('failed', 'done'):
        ledger = TaskLedger()
        initial = ledger.create('request', 'old')
        ledger.revise(initial.id, 'new', reason='correction')
        ledger.record(initial.id, outcome(), revision=2)
        result = ledger.record(initial.id, outcome(status), revision=1)
        assert result.status == 'done' and result.revision == 2
        assert [(attempt.revision, attempt.status) for attempt in result.attempts] == [(2, 'done'), (1, status)]


@pytest.mark.parametrize('revision', [0, -1, True, False, 1.0, '1', 2, 100])
def test_invalid_revision_fails_without_recording(revision):
    ledger = TaskLedger()
    initial = ledger.create('request', 'goal')
    with pytest.raises(ValueError, match='known positive task revision'):
        ledger.record(initial.id, outcome(), revision=revision)
    assert ledger.get(initial.id) == initial


def test_completed_old_revision_cannot_acquire_a_second_attempt():
    ledger = TaskLedger()
    initial = ledger.create('request')
    ledger.record(initial.id, outcome(), revision=1)
    ledger.revise(initial.id, 'new', reason='correction')
    before = ledger.get(initial.id)
    with pytest.raises(ValueError, match='completed'):
        ledger.record(initial.id, outcome('failed'), revision=1)
    assert ledger.get(initial.id) == before
    assert ledger.record(initial.id, outcome(), revision=2).status == 'done'


def test_explicit_revision_payload_and_returned_snapshots_are_detached():
    ledger = TaskLedger()
    initial = ledger.create('request', {'target': []})
    ledger.revise(initial.id, {'target': ['new']}, reason='correction')
    plan = {'steps': ['old']}
    receipt = {'written': ['old']}
    steps = (StepAttempt('one', {'args': ['old']}, receipt, True),)
    result = ledger.record(initial.id, outcome(plan=plan, receipt=receipt, steps=steps), revision=1)
    plan['steps'].clear()
    receipt['written'].clear()
    steps[0].call['args'].clear()
    result.attempts[0].plan['steps'].clear()
    saved = ledger.get(initial.id)
    assert saved.attempts[0].plan == {'steps': ['old']}
    assert saved.attempts[0].receipt == {'written': ['old']}
    assert saved.attempts[0].steps[0].call == {'args': ['old']}


def test_parallel_old_revision_attempts_do_not_lose_receipts_or_change_new_status():
    ledger = TaskLedger()
    initial = ledger.create('request')
    ledger.revise(initial.id, 'new', reason='correction')
    def record(index):
        ledger.record(initial.id, outcome('failed', receipt={'attempt': index}), revision=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tuple(pool.map(record, range(20)))
    final = ledger.get(initial.id)
    assert final.status == 'ready'
    assert len(final.attempts) == 20
    assert {attempt.receipt['attempt'] for attempt in final.attempts} == set(range(20))


def test_revision_during_payload_copy_is_not_overwritten_by_old_attempt():
    ledger = TaskLedger()
    initial = ledger.create('request', 'old')
    class RevisingReceipt:
        def __deepcopy__(self, memo):
            ledger.revise(initial.id, 'new', reason='callback correction')
            return {'applied': 'old'}
    result = ledger.record(initial.id, outcome(receipt=RevisingReceipt()), revision=1)
    assert result.revision == 2 and result.goal == 'new' and result.status == 'ready'
    assert result.attempts[0].revision == 1
    assert result.attempts[0].receipt == {'applied': 'old'}
