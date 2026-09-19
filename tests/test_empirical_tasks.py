"""Actual Gym task identity, bounded continuation, and revision-safe execution."""
from dataclasses import replace

import pytest
pytest.importorskip('gymnasium')

from eval.learning.empirical_planning_fixture import make_setup, GOAL
from tensorcode.agent.empirical_tasks import EmpiricalGoal, pursue
from tensorcode.outcomes import Unknown


@pytest.fixture
def setup():
    value = make_setup()
    try:
        yield value
    finally:
        value.plugin.close()


def goal(s):
    return EmpiricalGoal(GOAL, s.calls, s.model.id, s.model.provider)


def route_chooser(actions):
    remaining = iter(actions)
    def choose(plan):
        action = next(remaining)
        return next(call for call in plan.first_calls if dict(call.args)['action'] == action)
    return choose


def test_one_task_pauses_then_resumes_new_plan_without_replaying_steps(setup):
    s = setup
    before = s.plugin.sequence
    first = pursue(s.agent, s.model, goal(s), max_steps=2, choose=route_chooser((2, 2)), max_depth=4)
    assert first.status == 'suspended' and first.reason == 'step_budget_exhausted'
    assert len(first.steps) == 2 and all(step.receipt.status == 'applied' for step in first.steps)
    assert all(step.verified is True for step in first.steps)
    assert first.verified is False
    assert s.plugin.sequence == before + 2
    assert s.agent.tasks.get(first.task_id).status == 'suspended'
    resumed = pursue(s.agent, s.model, task_id=first.task_id, max_steps=2,
                     choose=route_chooser((1, 1)), max_depth=4)
    assert resumed.status == 'done' and resumed.verified is True
    assert resumed.task_id == first.task_id and s.plugin.sequence == before + 4
    task = s.agent.tasks.get(first.task_id)
    assert task.status == 'done' and task.revision == 1 and len(task.attempts) == 2
    assert set(first.plan.proposal_ids).isdisjoint(resumed.plan.proposal_ids)
    assert len(resumed.steps) == 2
    assert first.interpretation_id is None and resumed.candidate_id is None
    assert not tuple(s.agent.store.claims())
    with pytest.raises(ValueError, match='completed task'):
        pursue(s.agent, s.model, task_id=task.id, max_steps=4)
    assert s.plugin.sequence == before + 4


def test_equal_routes_defer_without_authorized_choice(setup):
    s = setup
    before = s.plugin.sequence
    result = pursue(s.agent, s.model, goal(s), max_steps=4)
    assert result.status == 'unknown' and result.reason == 'first_action_choice_required'
    assert not result.steps and s.plugin.sequence == before
    # A no-action attempt can be resumed with an explicit choice.
    retry = pursue(s.agent, s.model, task_id=result.task_id, max_steps=1, choose=route_chooser((2,)))
    assert retry.status == 'suspended' and len(retry.steps) == 1


def test_revision_during_choice_stops_before_action_and_records_original_revision(setup):
    s = setup
    task = s.agent.tasks.create('authored empirical task', goal(s))
    before = s.plugin.sequence
    def change(plan):
        s.agent.tasks.revise(task.id, goal(s), reason='explicit goal revision during choice')
        return plan.first_calls[0]
    result = pursue(s.agent, s.model, task_id=task.id, max_steps=4, choose=change)
    current = s.agent.tasks.get(task.id)
    assert result.reason == 'task_revision_changed' and result.status == 'unknown'
    assert s.plugin.sequence == before and not result.steps
    assert current.revision == 2 and current.status == 'ready'
    assert current.attempts[-1].revision == 1


def test_revision_during_action_keeps_old_receipt_and_does_not_dispatch_again(setup, monkeypatch):
    s = setup
    task = s.agent.tasks.create('authored empirical task', goal(s))
    original = s.plugin.execute
    def execute(call, *, key=None):
        receipt = original(call, key=key)
        s.agent.tasks.revise(task.id, goal(s), reason='explicit revision while action applied')
        return receipt
    monkeypatch.setattr(s.plugin, 'execute', execute)
    before = s.plugin.sequence
    result = pursue(s.agent, s.model, task_id=task.id, max_steps=4, choose=route_chooser((2, 2, 1, 1)))
    assert result.reason == 'task_revision_changed'
    assert len(result.steps) == 1 and result.steps[0].receipt.status == 'applied'
    assert s.plugin.sequence == before + 1
    current = s.agent.tasks.get(task.id)
    assert current.status == 'ready' and current.revision == 2
    assert current.attempts[-1].revision == 1 and current.attempts[-1].steps == result.steps


def test_revision_in_execution_sensor_guard_blocks_action(setup, monkeypatch):
    s = setup
    task = s.agent.tasks.create('authored empirical task', goal(s))
    original = s.plugin.observe_evidence
    observations = 0
    def observe():
        nonlocal observations
        observations += 1
        if observations == 3:  # Task observation, before_action, final prediction_guard.
            s.agent.tasks.revise(task.id, goal(s), reason='revision from observation callback')
        return original()
    monkeypatch.setattr(s.plugin, 'observe_evidence', observe)
    before = s.plugin.sequence
    result = pursue(s.agent, s.model, task_id=task.id, max_steps=2, choose=route_chooser((2, 2)))
    assert result.reason == 'task_revision_changed' and s.plugin.sequence == before
    assert result.receipt.status == 'rejected'
    assert s.agent.tasks.get(task.id).attempts[-1].revision == 1


def test_applied_but_unobserved_attempt_requires_explicit_revision(setup, monkeypatch):
    s = setup
    original = s.plugin.observe_evidence
    executed = s.plugin.execute
    unavailable = False
    def execute(call, *, key=None):
        nonlocal unavailable
        receipt = executed(call, key=key)
        unavailable = True
        return receipt
    monkeypatch.setattr(s.plugin, 'execute', execute)
    monkeypatch.setattr(s.plugin, 'observe_evidence', lambda: None if unavailable else original())
    result = pursue(s.agent, s.model, goal(s), max_steps=4, choose=route_chooser((2,)))
    assert result.status == 'unverified' and isinstance(result.verified, Unknown)
    assert result.receipt.status == 'applied'
    with pytest.raises(ValueError, match='revise explicitly'):
        pursue(s.agent, s.model, task_id=result.task_id, choose=route_chooser((2,)))


def test_reentrant_pursuit_is_rejected_without_an_extra_attempt(setup):
    s = setup
    task = s.agent.tasks.create('authored empirical task', goal(s))
    def choose(plan):
        with pytest.raises(ValueError, match='active attempt'):
            pursue(s.agent, s.model, task_id=task.id)
        return next(call for call in plan.first_calls if dict(call.args)['action'] == 2)
    result = pursue(s.agent, s.model, task_id=task.id, max_steps=1, choose=choose)
    assert result.status == 'suspended' and len(result.steps) == 1
    assert len(s.agent.tasks.get(task.id).attempts) == 1


def test_task_admission_is_rechecked_after_lock_acquisition(setup):
    from tensorcode.agent.core import Outcome
    from tensorcode.agent.understand import Act
    s = setup
    task = s.agent.tasks.create('authored empirical task', goal(s))
    class CompletedBetweenSnapshotAndLock:
        released = False
        def acquire(self, *, blocking):
            # Another completed attempt becomes visible before admission owns lock.
            s.agent.tasks.record(task.id, Outcome(Act('request', task.goal, None), 'done',
                                                 goal=task.goal, verified=True), revision=task.revision)
            return True
        def release(self):
            self.released = True
    lock = CompletedBetweenSnapshotAndLock()
    s.agent._empirical_task_locks[task.id] = lock
    before = s.plugin.sequence
    with pytest.raises(ValueError, match='completed task'):
        pursue(s.agent, s.model, task_id=task.id, choose=route_chooser((2,)))
    assert lock.released and s.plugin.sequence == before
    assert len(s.agent.tasks.get(task.id).attempts) == 1


def test_initial_observed_goal_cannot_complete_a_revision_changed_by_sensor(setup, monkeypatch):
    s = setup
    initial_goal = replace(goal(s), state=(0, False, False))
    task = s.agent.tasks.create('authored initial-state goal', initial_goal)
    original = s.plugin.observe_evidence
    def observe():
        s.agent.tasks.revise(task.id, goal(s), reason='new goal during fresh observation')
        return original()
    monkeypatch.setattr(s.plugin, 'observe_evidence', observe)
    before = s.plugin.sequence
    result = pursue(s.agent, s.model, task_id=task.id, max_steps=0)
    assert result.status == 'unknown' and result.reason == 'task_revision_changed'
    assert result.receipt is None and s.plugin.sequence == before
    current = s.agent.tasks.get(task.id)
    assert current.status == 'ready' and current.revision == 2
    assert current.attempts[-1].revision == 1


def test_unexpected_post_action_exception_records_uncertainty_and_blocks_retry(setup, monkeypatch):
    import tensorcode.agent.empirical_tasks as tasks
    s = setup
    original = tasks.execute
    def raise_after_execution(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.receipt.status == 'applied'
        raise RuntimeError('authored failure after actual execution returned')
    monkeypatch.setattr(tasks, 'execute', raise_after_execution)
    before = s.plugin.sequence
    result = pursue(s.agent, s.model, goal(s), max_steps=2, choose=route_chooser((2, 2)))
    assert result.status == 'unverified' and result.reason == 'empirical_execution_error'
    assert result.receipt.status == 'indeterminate'
    assert len(result.steps) == 1 and isinstance(result.steps[0].verified, Unknown)
    assert s.plugin.sequence == before + 1
    assert len(s.agent.tasks.get(result.task_id).attempts) == 1
    with pytest.raises(ValueError, match='revise explicitly'):
        pursue(s.agent, s.model, task_id=result.task_id, max_steps=0)


def test_choice_validation_exception_after_earlier_step_keeps_attempt_nonresumable(setup, monkeypatch):
    import tensorcode.agent.empirical_tasks as tasks
    s = setup
    original_same = tasks._same
    marker = object()
    def raises_for_marker(left, right):
        if left is marker:
            raise ValueError('authored invalid choice comparison')
        return original_same(left, right)
    monkeypatch.setattr(tasks, '_same', raises_for_marker)
    choices = 0
    def choose(plan):
        nonlocal choices
        choices += 1
        return plan.first_calls[0] if choices == 1 else marker
    before = s.plugin.sequence
    result = pursue(s.agent, s.model, goal(s), max_steps=4, choose=choose)
    assert result.status == 'unknown' and result.reason == 'empirical_choice_error'
    assert len(result.steps) == 1 and result.steps[0].receipt.status == 'applied'
    assert s.plugin.sequence == before + 1
    with pytest.raises(ValueError, match='revise explicitly'):
        pursue(s.agent, s.model, task_id=result.task_id, max_steps=0)


def test_assessment_publication_failure_preserves_applied_receipt_in_task(setup, monkeypatch):
    s = setup
    add_source = s.agent.interpretations.add_source
    def fail_assessment(*args, **kwargs):
        if kwargs.get('provider') == 'empirical-planning' and kwargs.get('modality') == 'assessment':
            raise RuntimeError('authored assessment writer failure')
        return add_source(*args, **kwargs)
    monkeypatch.setattr(s.agent.interpretations, 'add_source', fail_assessment)
    before = s.plugin.sequence
    result = pursue(s.agent, s.model, goal(s), max_steps=2, choose=route_chooser((2, 2)))
    assert result.status == 'unverified' and result.reason == 'assessment_record_failed'
    assert result.receipt.status == 'applied' and isinstance(result.verified, Unknown)
    assert s.plugin.sequence == before + 1 and len(result.steps) == 1
    with pytest.raises(ValueError, match='revise explicitly'):
        pursue(s.agent, s.model, task_id=result.task_id, max_steps=0)
