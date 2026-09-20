"""Authored meaning/goal association; actual request and task execution guards."""
import pytest
from tensorcode.agent import Agent
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import Act, Sentence
from tensorcode.language import Frame, Request, verbnet
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from test_agent_tasks import Devices, desired


def dependency(agent):
    workspace = agent.interpretations
    source = workspace.add_source('Prepare the selected device.', provider='authored test evidence')
    group = workspace.create_group(source.id)
    candidate = workspace.propose(group.id, {'target': 'device:a'}, provenance=('authored interpretation fixture',))
    workspace.select(group.id, candidate.id, reason='explicit test selection')
    dep = agent.capture_task_dependency(group.id, basis=('Authored correspondence to the supplied device goal',))
    return group.id, candidate.id, dep


@pytest.mark.parametrize('mutation', ['unset', 'rival'])
def test_stale_interpretation_blocks_structured_goal(mutation):
    plugin = Devices()
    agent = Agent([plugin])
    group, candidate, dep = dependency(agent)
    if mutation == 'unset':
        agent.interpretations.unset(group, reason='new evidence requires reconsideration')
    else:
        agent.interpretations.propose(group, {'target': 'device:b'})
    outcome = agent.pursue(desired(), dependencies=(dep,))
    task = agent.tasks.get(outcome.task_id)
    assert outcome.status == 'unknown'
    assert outcome.verified.reason == 'interpretation_dependency_changed'
    assert task.status == 'blocked'
    assert task.dependencies == task.revisions[0].dependencies == (dep,)
    assert not plugin.calls


@pytest.mark.parametrize('stage', ['precondition', 'execute'])
def test_withdrawal_during_request_keeps_receipt_without_completion(stage):
    plugin = Devices()
    agent = Agent([plugin])
    group, _, dep = dependency(agent)
    def withdraw():
        agent.interpretations.unset(group, reason='withdraw while task is active')
    if stage == 'precondition':
        original = plugin.precondition_holds
        def check(condition, args):
            withdraw()
            return original(condition, args)
        plugin.precondition_holds = check
    else:
        original = plugin.execute
        def execute(call, **kwargs):
            receipt = original(call, **kwargs)
            withdraw()
            return receipt
        plugin.execute = execute
    outcome = agent.pursue(desired(), dependencies=(dep,))
    assert outcome.status == 'unknown'
    assert outcome.verified.reason == 'interpretation_dependency_changed'
    task = agent.tasks.get(outcome.task_id)
    assert task.status == 'blocked' and task.attempts[0].revision == 1
    assert task.attempts[0].receipt == outcome.receipt
    assert outcome.receipt.status == ('rejected' if stage == 'precondition' else 'applied')
    assert len(plugin.calls) == (0 if stage == 'precondition' else 1)


def test_goal_revision_does_not_silently_drop_its_dependencies():
    agent = Agent([Devices()])
    group, candidate, dep = dependency(agent)
    task = agent.tasks.create('supplied goal', desired(), dependencies=(dep,))
    agent.interpretations.unset(group, reason='reconsider')
    newer = agent.tasks.revise(task.id, desired('b'), reason='change target only')
    assert newer.dependencies == newer.revisions[-1].dependencies == (dep,)
    assert agent.pursue(task_id=task.id).status == 'unknown'
    agent.interpretations.select(group, candidate, reason='fresh explicit assessment')
    fresh = agent.capture_task_dependency(group, basis=('Explicit renewed goal association',))
    revised = agent.tasks.revise(task.id, desired('b'), reason='renew meaning commitment', dependencies=(fresh,))
    assert revised.revisions[0].dependencies == (dep,)
    assert revised.dependencies == (fresh,)
    assert agent.pursue(task_id=task.id).status == 'done'


@pytest.mark.parametrize('withdraw', [False, True])
def test_actual_turn_request_automatically_retains_selected_meaning_dependency(monkeypatch, withdraw):
    from tensorcode.agent import core
    from agent_test_support import selected_agent, select_unique_fixture_goal, supplied_goal_batch
    frame = Frame('enable', {'object': Ref('device:a')})
    act = Act('request', Request(frame), frame)
    sentence = Sentence('enable a', ('enable', 'a'), None, (act,))
    monkeypatch.setattr(core.ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'authored request fixture'))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: supplied_goal_batch(desired(), frame=frame))
    plugin = Devices()
    monkeypatch.setattr(plugin, "refine_goal", lambda _: desired())
    agent = selected_agent([plugin], goal_selector=select_unique_fixture_goal)
    original = plugin.precondition_holds
    def check(condition, args):
        if withdraw:
            group = agent.interpretations.values()[0]
            agent.interpretations.unset(group.id, reason='counterevidence during precondition')
        return original(condition, args)
    plugin.precondition_holds = check
    turn = agent.turn('enable a')
    result = turn.outcomes[0]
    task = agent.tasks.get(result.task_id)
    assert len(task.dependencies) == 2
    assert task.dependencies[0].group_id == result.interpretation_id
    assert task.dependencies[0].candidate_id == result.candidate_id
    assert result.status == ('unknown' if withdraw else 'done')
    assert len(plugin.calls) == (0 if withdraw else 1)


def test_turn_cannot_bind_an_old_act_to_a_new_selection_during_deixis(monkeypatch):
    from tensorcode.agent import core
    from agent_test_support import selected_agent, select_unique_fixture_goal, supplied_goal_batch
    frame = Frame('enable', {'object': Ref('device:a')})
    act = Act('request', Request(frame), frame)
    sentence = Sentence('enable a', ('enable', 'a'), None, (act,))
    monkeypatch.setattr(core.ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'authored request fixture'))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: supplied_goal_batch(desired(), frame=frame))
    plugin = Devices()
    monkeypatch.setattr(plugin, "refine_goal", lambda _: desired())
    agent = selected_agent([plugin], goal_selector=select_unique_fixture_goal)
    original = agent.deixis
    changed = False
    def change(value):
        nonlocal changed
        if not changed:
            changed = True
            group = agent.interpretations.values()[0]
            rival = agent.interpretations.propose(group.id, {'authored': 'new interpretation'})
            agent.interpretations.select(group.id, rival.id, reason='new selection before request handling')
        return original(value)
    agent.deixis = change
    turn = agent.turn('enable a')
    assert turn.outcomes[0].status == 'unknown'
    assert not plugin.calls
    task = agent.tasks.get(turn.outcomes[0].task_id)
    assert task.dependencies[0].candidate_id != agent.interpretations.values()[0].selected_id


@pytest.mark.parametrize('refined', [False, True])
def test_language_request_verification_failure_keeps_goal_receipt_and_dependency(monkeypatch, refined):
    from tensorcode.agent import core
    from agent_test_support import selected_agent, select_unique_fixture_goal, supplied_goal_batch
    frame = Frame('enable', {'object': Ref('device:a')})
    act = Act('request', Request(frame), frame)
    sentence = Sentence('enable a', ('enable', 'a'), None, (act,))
    monkeypatch.setattr(core.ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'authored request fixture'))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: supplied_goal_batch(desired(), frame=frame))
    plugin = Devices()
    expected_goal = desired('b') if refined else desired()
    monkeypatch.setattr(plugin, 'refine_goal', lambda _: expected_goal)
    def failed_verification(*args):
        raise RuntimeError('authored failure after actual action')
    monkeypatch.setattr(plugin, 'holds', failed_verification)
    agent = selected_agent([plugin], goal_selector=select_unique_fixture_goal)
    turn = agent.turn('enable a')
    outcome = turn.outcomes[0]
    task = agent.tasks.get(outcome.task_id)
    assert outcome.status == task.status == 'unverified'
    assert outcome.verified.reason == 'task_attempt_error'
    assert outcome.goal == task.goal == expected_goal
    assert len(task.dependencies) == 2
    assert task.dependencies[0].group_id == outcome.interpretation_id
    assert task.dependencies[0].candidate_id == outcome.candidate_id
    assert outcome.receipt.status == 'applied'
    assert len(outcome.steps) == 1 and outcome.steps[0].receipt == outcome.receipt
    assert task.attempts[0].steps == outcome.steps
    assert task.attempts[0].receipt == outcome.receipt
    assert plugin.calls == [Ref('device:b' if refined else 'device:a')]
    with pytest.raises(ValueError, match='may have changed'):
        agent.pursue(task_id=task.id)
    assert len(plugin.calls) == 1
