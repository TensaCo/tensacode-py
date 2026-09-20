"""Explicit fixture goals test deferred adoption, not learned intent resolution."""
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.core import InterpretationDecision
from tensorcode.agent.goal_interpretation import retain_goal_proposals
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.understand import Act, Sentence
from tensorcode.language import Frame, Request, verbnet
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from agent_test_support import supplied_goal_batch
from test_agent_tasks import Devices, desired


def setup(monkeypatch, *, parent=False):
    plugin = Devices()
    agent = Agent([plugin])
    frame = Frame('enable', {'object': Ref('device:a')})
    batch = supplied_goal_batch(desired(), frame=frame)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *args, **kwargs: batch)
    monkeypatch.setattr(plugin, 'refine_goal', lambda goal: desired())
    dependency = add_parent(agent) if parent else None
    act = Act('request', Request(frame), frame)
    outcome = agent.request(Sentence('enable a', ('enable', 'a'), None, (act,)), act, [],
                            interpretation_dependency=dependency)
    task = agent.tasks.get(outcome.task_id)
    assert task.goal_interpretation_id == outcome.goal_interpretation_id
    assert isinstance(outcome.goal, Unknown) and not plugin.calls
    return agent, plugin, task, dependency


def add_parent(agent):
    source = agent.interpretations.add_source('authored parent meaning')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, {'authored': True})
    agent.interpretations.select(group.id, candidate.id, reason='authored selection')
    return capture_dependency(agent.interpretations, group.id, basis=('supplied correspondence',))


def decision(agent, task):
    group = agent.interpretations.get(task.goal_interpretation_id)
    return InterpretationDecision(group.candidates[0].id, 'explicit fixture goal choice',
        compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))


def test_adopt_retained_goal_without_search_or_action_then_explicitly_pursue(monkeypatch):
    agent, plugin, task, _ = setup(monkeypatch)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('re-enumerated'))
    perceive = agent.perceive
    monkeypatch.setattr(agent, 'perceive', lambda *a: pytest.fail('adoption perceived'))
    adopted = agent.adopt_task_goal(task.id, decision(agent, task), reason='clarified desired result')
    assert adopted.id == task.id and adopted.revision == 2 and adopted.status == 'ready'
    assert adopted.goal == desired() and adopted.attempts == task.attempts
    assert adopted.revisions[0].goal == task.goal and plugin.calls == []
    monkeypatch.setattr(agent, 'perceive', perceive)
    outcome = agent.pursue(task_id=task.id)
    assert outcome.status == 'done' and plugin.calls == [Ref('device:a')]
    assert [attempt.revision for attempt in agent.tasks.get(task.id).attempts] == [1, 2]


def test_stale_explicit_choice_does_not_revise_task(monkeypatch):
    agent, plugin, task, _ = setup(monkeypatch)
    stale = decision(agent, task)
    agent.interpretations.unset(task.goal_interpretation_id, reason='comparison changed')
    result = agent.adopt_task_goal(task.id, stale, reason='late response')
    assert isinstance(result, Unknown) and agent.tasks.current_revision(task.id) == 1
    assert plugin.calls == []


def test_parent_and_unrelated_dependencies_preserved_old_goal_dependency_replaced(monkeypatch):
    agent, _, task, parent = setup(monkeypatch, parent=True)
    unrelated = add_parent(agent)
    task = agent.tasks.revise(task.id, task.goal, reason='add independent prerequisite',
        dependencies=(parent, unrelated), goal_interpretation_id=task.goal_interpretation_id)
    first = agent.adopt_task_goal(task.id, decision(agent, task), reason='first choice')
    second = agent.adopt_task_goal(task.id, decision(agent, first), reason='renew explicit choice')
    assert second.dependencies[:2] == (parent, unrelated)
    assert len(second.dependencies) == 3
    assert second.dependencies[-1].revision > first.dependencies[-1].revision


def test_manual_link_recovers_retained_parent_dependency(monkeypatch):
    agent, _, task, parent = setup(monkeypatch, parent=True)
    manual = agent.tasks.create('manual linked task', goal_interpretation_id=task.goal_interpretation_id)
    adopted = agent.adopt_task_goal(manual.id, decision(agent, manual), reason='adopt retained result')
    assert parent in adopted.dependencies


@pytest.mark.parametrize('mutation', ['parent', 'task'])
def test_refinement_callback_changes_authorization_without_committing(monkeypatch, mutation):
    agent, plugin, task, parent = setup(monkeypatch, parent=True)
    def refine(goal):
        if mutation == 'parent':
            agent.interpretations.unset(parent.group_id, reason='withdraw parent')
        else:
            agent.tasks.revise(task.id, desired('b'), reason='concurrent correction')
        return desired()
    monkeypatch.setattr(plugin, 'refine_goal', refine)
    result = agent.adopt_task_goal(task.id, decision(agent, task), reason='late choice')
    assert isinstance(result, Unknown)
    assert result.reason == ('interpretation_dependency_changed' if mutation == 'parent' else 'task_revision_changed')
    assert agent.tasks.current_revision(task.id) == (1 if mutation == 'parent' else 2)
    assert plugin.calls == []


def test_unconsumed_roles_stay_blocked_until_explicit_refinement(monkeypatch):
    agent, plugin, _, _ = setup(monkeypatch)
    frame = Frame('enable', {'object': Ref('device:a'), 'location': Ref('place:x')})
    lexical = verbnet.Goal('enable', 'authored', desired().conditions, frame, ('location',))
    batch = verbnet.GoalCandidates((verbnet.GoalProposal(lexical, ()),))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: batch)
    group_id = retain_goal_proposals(agent, frame, 'authored qualified request')
    task = agent.tasks.create('qualified', goal_interpretation_id=group_id)
    monkeypatch.setattr(plugin, 'refine_goal', lambda goal: Unknown('no_refinement'))
    rejected = agent.adopt_task_goal(task.id, decision(agent, task), reason='choose goal')
    assert rejected.reason == 'unconsumed_request_semantics'
    assert agent.tasks.current_revision(task.id) == 1
    def authored_consumption(goal):
        assert goal.unmapped_roles == ('location',) and goal.frame.roles['location'] == Ref('place:x')
        return desired()
    monkeypatch.setattr(plugin, 'refine_goal', authored_consumption)
    adopted = agent.adopt_task_goal(task.id, decision(agent, task), reason='explicit domain binding')
    assert adopted.goal == desired() and not plugin.calls


def test_missing_link_and_invalid_reason(monkeypatch):
    agent, _, task, _ = setup(monkeypatch)
    manual = agent.tasks.create('unlinked')
    assert agent.adopt_task_goal(manual.id, decision(agent, task), reason='choose').reason == 'no_retained_goal_interpretation'
    with pytest.raises(ValueError, match='reason'):
        agent.adopt_task_goal(task.id, decision(agent, task), reason=' ')


def test_request_description_failure_keeps_already_refined_goal_attribution(monkeypatch):
    from tensorcode.goals import GoalSpec
    from agent_test_support import select_unique_fixture_goal
    plugin = Devices()
    agent = Agent([plugin], goal_selector=select_unique_fixture_goal)
    frame = Frame('enable', {'object': Ref('device:a')})
    batch = supplied_goal_batch(desired(), frame=frame)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: batch)
    refined = desired('b')
    monkeypatch.setattr(plugin, 'refine_goal', lambda goal: refined)
    def broken_description(self):
        raise RuntimeError('fixture description failure')
    monkeypatch.setattr(GoalSpec, 'describe', broken_description)
    act = Act('request', Request(frame), frame)
    result = agent.request(Sentence('enable a', ('enable', 'a'), None, (act,)), act, [])
    assert result.goal == refined
    assert agent.tasks.get(result.task_id).goal == refined
    assert not plugin.calls


def test_two_retained_goal_choices_correct_completed_task_without_replaying_receipt(monkeypatch):
    plugin = Devices()
    agent = Agent([plugin])
    frame = Frame('enable', {'object': Ref('device:a')})
    proposals = tuple(supplied_goal_batch(desired(name), frame=frame).proposals[0] for name in ('a', 'b'))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: verbnet.GoalCandidates(proposals))
    monkeypatch.setattr(plugin, 'refine_goal', lambda goal: desired('a') if goal.conditions == desired('a').conditions else desired('b'))
    group_id = retain_goal_proposals(agent, frame, 'authored two-goal fixture')
    task = agent.tasks.create('choose device', goal_interpretation_id=group_id)
    def choose(index):
        group = agent.interpretations.get(group_id)
        return InterpretationDecision(group.candidates[index].id, 'explicit device choice',
            compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('re-enumerated retained alternatives'))
    first = agent.adopt_task_goal(task.id, choose(0), reason='prepare device a')
    completed = agent.pursue(task_id=first.id)
    assert completed.status == 'done' and plugin.calls == [Ref('device:a')]
    previous = agent.tasks.get(task.id)
    corrected = agent.adopt_task_goal(task.id, choose(1), reason='now prepare device b')
    assert corrected.status == 'ready' and corrected.goal == desired('b')
    assert corrected.attempts == previous.attempts
    assert corrected.attempts[-1].receipt == completed.receipt
    assert corrected.revisions[-2].goal == desired('a')
    assert plugin.calls == [Ref('device:a')]
    assert agent.pursue(task_id=task.id).status == 'done'
    assert plugin.calls == [Ref('device:a'), Ref('device:b')]
    assert [attempt.revision for attempt in agent.tasks.get(task.id).attempts] == [2, 3]


def test_goal_payload_copy_withdrawal_blocks_final_adoption_commit(monkeypatch):
    from tensorcode.goals import GoalSpec
    agent, plugin, task, parent = setup(monkeypatch, parent=True)
    copied = []
    class WithdrawingGoal(GoalSpec):
        def __deepcopy__(self, memo):
            copied.append(True)
            agent.interpretations.unset(parent.group_id, reason='withdrawal during ledger payload copy')
            return desired()
    monkeypatch.setattr(plugin, 'refine_goal', lambda goal: WithdrawingGoal(desired().conditions))
    result = agent.adopt_task_goal(task.id, decision(agent, task), reason='choice before copy callback')
    assert copied
    assert isinstance(result, Unknown) and result.reason == 'interpretation_dependency_changed'
    retained = agent.tasks.get(task.id)
    assert retained.revision == task.revision and retained.attempts == task.attempts
    assert not plugin.calls
