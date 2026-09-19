"""A task correction never licenses dispatch or completion for an obsolete goal."""
import pytest
from tensorcode.agent import Agent
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from test_agent_tasks import Devices, desired


@pytest.mark.parametrize('stage', ['precondition', 'execute', 'verification'])
def test_correction_preserves_original_revision_receipts(stage):
    plugin = Devices()
    agent = Agent([plugin])
    task = agent.tasks.create('explicit original request', desired('a'))
    changed = False
    def revise():
        nonlocal changed
        if not changed:
            changed = True
            agent.tasks.revise(task.id, desired('b'), reason='explicit correction')
    if stage == 'precondition':
        original = plugin.precondition_holds
        def checked(condition, args):
            revise()
            return original(condition, args)
        plugin.precondition_holds = checked
    elif stage == 'execute':
        original = plugin.execute
        def execute(call, **kwargs):
            receipt = original(call, **kwargs)
            revise()
            return receipt
        plugin.execute = execute
    else:
        original = plugin.holds
        def holds(cap, args):
            result = original(cap, args)
            revise()
            return result
        plugin.holds = holds
    result = agent.pursue(task_id=task.id)
    current = agent.tasks.get(task.id)
    assert result.status == 'suspended'
    assert isinstance(result.verified, Unknown)
    assert current.revision == 2 and current.status == 'ready'
    assert current.goal == desired('b')
    assert len(current.attempts) == 1 and current.attempts[0].revision == 1
    assert current.attempts[0].receipt == result.receipt
    assert plugin.calls == ([] if stage == 'precondition' else [Ref('device:a')])
    # A new goal revision may subsequently be pursued, without replaying a.
    later = agent.pursue(task_id=task.id)
    assert later.status == 'done'
    assert agent.tasks.get(task.id).attempts[-1].revision == 2
    assert plugin.calls[-1] == Ref('device:b')


def test_correction_between_modeled_steps_stops_old_filesystem_plan(tmp_path):
    from tensorcode.agent.filesystem import FileSystemPlugin
    from tensorcode.goals import GoalSpec, Condition
    plugin = FileSystemPlugin(tmp_path)
    agent = Agent([plugin])
    old = GoalSpec((Condition('content', {'path': 'old/nested/result.txt', 'text': 'obsolete'}),))
    new = GoalSpec((Condition('content', {'path': 'new.txt', 'text': 'current'}),))
    task = agent.tasks.create('supplied filesystem task', old)
    original = plugin.execute
    calls = []
    def execute(call, **kwargs):
        receipt = original(call, **kwargs)
        calls.append(call)
        if len(calls) == 1:
            agent.tasks.revise(task.id, new, reason='corrected target after first real step')
        return receipt
    plugin.execute = execute
    outcome = agent.pursue(task_id=task.id)
    assert outcome.status == 'suspended'
    assert len(calls) == len(outcome.steps) == 1
    assert outcome.receipt.status == 'applied'
    assert not (tmp_path / 'old/nested/result.txt').exists()
    current = agent.tasks.get(task.id)
    assert current.revision == 2 and current.status == 'ready'
    assert current.attempts[0].revision == 1
    assert agent.pursue(task_id=task.id).status == 'done'
    assert (tmp_path / 'new.txt').read_text() == 'current'
    assert not (tmp_path / 'old/nested/result.txt').exists()


def test_reentrant_structured_pursuit_cannot_execute_twice():
    plugin = Devices()
    agent = Agent([plugin])
    task = agent.tasks.create('one task', desired())
    perceive = agent.perceive
    nested = False
    def perception(events):
        nonlocal nested
        if not nested:
            nested = True
            with pytest.raises(ValueError, match='active attempt'):
                agent.pursue(task_id=task.id)
        return perceive(events)
    agent.perceive = perception
    result = agent.pursue(task_id=task.id)
    assert result.status == 'done'
    assert plugin.calls == [Ref('device:a')]
    assert len(agent.tasks.get(task.id).attempts) == 1


def test_post_action_verification_exception_retains_receipt_and_blocks_retry():
    plugin = Devices()
    agent = Agent([plugin])
    def unavailable(*args):
        raise RuntimeError('verification pipeline failed')
    plugin.holds = unavailable
    result = agent.pursue(desired())
    assert result.status == 'unverified'
    assert result.receipt.status == 'applied'
    assert len(result.steps) == 1
    assert isinstance(result.steps[0].verified, Unknown)
    task = agent.tasks.get(result.task_id)
    assert task.attempts[0].receipt == result.receipt
    with pytest.raises(ValueError, match='may have changed'):
        agent.pursue(task_id=result.task_id)
    assert plugin.calls == [Ref('device:a')]
