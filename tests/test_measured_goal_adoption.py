"""Explicit measured-goal teaching stays declarative through learned adoption."""
from dataclasses import replace
import pytest

from tensorcode.agent import Agent, InterpretationDecision, Plugin
from tensorcode.agent.goal_interpretation import retain_taught_goal, retain_goal_proposals
from tensorcode.agent.goal_learning import extract_goal_example, fit_goal_model, admit_goal_model
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.understand import Act, Sentence
from tensorcode.goals import MeasuredActionGoal, GoalSpec, Condition
from tensorcode.language import Frame, Request, verbnet
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref


def parent(agent, name):
    frame = Frame('prepare', {'object': Ref('node:' + name)}, {'mood': 'imperative'})
    source = agent.interpretations.add_source('authored prepare ' + name)
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, frame)
    agent.interpretations.select(group.id, candidate.id, reason='explicit authored reading')
    return frame, capture_dependency(agent.interpretations, group.id, basis=('explicit selected frame',))


def decision(agent, group_id):
    group = agent.interpretations.get(group_id)
    assert len(group.candidates) == 1
    return InterpretationDecision(group.candidates[0].id, 'explicit supplied goal choice',
        compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))


def teaching(agent, name, *, state=False):
    frame, dependency = parent(agent, name)
    target = frame.roles['object']
    goal = (GoalSpec((Condition('ready', {'target': target}),)) if state else
            MeasuredActionGoal(target, 'activate_node', 'inputChecked', True, ('explicit authored goal label',)))
    gid = retain_taught_goal(agent, frame, goal, 'authored goal teaching',
                            parent_dependency=dependency, reason='teacher supplied desired outcome')
    assert not isinstance(gid, Unknown), gid
    assert agent.interpretations.get(gid).selected_id is None
    source = agent.interpretations.get_source(agent.interpretations.get(gid).source_id)
    assert source.provider == 'explicit-goal-teaching'
    task = agent.tasks.create('authored instruction', Unknown('goal choice pending'),
                             dependencies=(dependency,), goal_interpretation_id=gid)
    adopted = agent.adopt_task_goal(task.id, decision(agent, gid), reason='explicit teaching adoption')
    assert not isinstance(adopted, Unknown), adopted
    assert adopted.goal == goal and adopted.status == 'ready'
    return adopted


def learned(monkeypatch):
    agent = Agent()
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *args, **kwargs: pytest.fail('teaching must not bootstrap lexical semantics'))
    tasks = [teaching(agent, name) for name in ('training-one', 'training-two', 'heldout')]
    handle = fit_goal_model(agent, [task.id for task in tasks[:2]], [tasks[2].id])
    assert not isinstance(handle, Unknown), handle
    admitted = admit_goal_model(agent, handle, reason='explicit learned correspondence admission')
    assert not isinstance(admitted, Unknown), admitted
    agent.goal_model = admitted
    return agent, tasks, admitted


def test_teacher_measured_and_state_goals_retain_distinct_exact_shapes_without_lexical_seed(monkeypatch):
    agent = Agent()
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *args, **kwargs: pytest.fail('lexical bootstrap'))
    for state in (False, True):
        task = teaching(agent, 'state' if state else 'measured', state=state)
        extracted = extract_goal_example(agent, task.id)
        assert not isinstance(extracted, Unknown), extracted
        assert extracted.example.goal == task.goal
        assert type(extracted.example.goal) is (GoalSpec if state else MeasuredActionGoal)


def test_learned_measured_adoption_keeps_type_dependencies_and_never_runs_state_planner(monkeypatch):
    agent, _, admitted = learned(monkeypatch)
    frame, dependency = parent(agent, 'novel')
    gid = retain_goal_proposals(agent, frame, 'novel authored reading', parent_dependency=dependency)
    task = agent.tasks.create('novel instruction', Unknown('awaiting explicit goal choice'),
                             dependencies=(dependency,), goal_interpretation_id=gid)
    monkeypatch.setattr(agent, '_execute_goal', lambda *a, **kw: pytest.fail('symbolic planner invoked'))
    adopted = agent.adopt_task_goal(task.id, decision(agent, gid), reason='explicit learned goal adoption')
    assert not isinstance(adopted, Unknown), adopted
    assert type(adopted.goal) is MeasuredActionGoal
    assert adopted.goal.target == Ref('node:novel')
    assert (adopted.goal.operation, adopted.goal.measurement, adopted.goal.desired_outcome) == ('activate_node', 'inputChecked', True)
    assert adopted.goal_interpretation_id == gid and dependency in adopted.dependencies
    assert admitted.dependency in adopted.dependencies
    assert not adopted.attempts and not agent.store.propositions()


def test_selected_request_defers_measured_execution_without_refinement(monkeypatch):
    agent, _, _ = learned(monkeypatch)
    frame, dependency = parent(agent, 'request-target')
    agent.goal_selector = lambda group: InterpretationDecision(group.candidates[0].id, 'explicit sole fixture proposal')
    plugin = Plugin('must-not-refine')
    monkeypatch.setattr(plugin, 'refine_goal', lambda *args: pytest.fail('measured goal sent to lexical refiner'))
    agent.plugins = [plugin]
    monkeypatch.setattr(agent, '_execute_goal', lambda *a, **kw: pytest.fail('measured goal sent to state planner'))
    act = Act('request', Request(frame), frame)
    outcome = agent.request(Sentence('supplied instruction', (), None, (act,)), act, [], interpretation_dependency=dependency)
    assert outcome.status == 'suspended' and outcome.reason == 'measured_goal_requires_materialization'
    assert type(outcome.goal) is MeasuredActionGoal and outcome.receipt is None
    task = agent.tasks.get(outcome.task_id)
    assert task.goal == outcome.goal and task.goal_interpretation_id == outcome.goal_interpretation_id


def test_teacher_frame_mismatch_and_unselected_parent_are_not_labels():
    agent = Agent()
    frame, dependency = parent(agent, 'one')
    goal = MeasuredActionGoal(Ref('node:one'), 'activate_node', 'inputChecked', True)
    result = retain_taught_goal(agent, replace(frame, predicate='invented'), goal, 'teaching',
        parent_dependency=dependency, reason='explicit teacher')
    assert isinstance(result, Unknown)
    agent.interpretations.unset(dependency.group_id, reason='withdraw reading')
    result = retain_taught_goal(agent, frame, goal, 'teaching', parent_dependency=dependency, reason='explicit teacher')
    assert isinstance(result, Unknown)


def test_tampered_teacher_parent_source_prevents_extract():
    agent = Agent()
    task = teaching(agent, 'source-target')
    parent_dependency = task.dependencies[0]
    group = agent.interpretations.get(parent_dependency.group_id)
    agent.interpretations._sources[group.source_id].metadata['tampered'] = True
    assert isinstance(extract_goal_example(agent, task.id), Unknown)


def test_arbitrary_unlinked_measured_task_cannot_teach():
    agent = Agent()
    task = agent.tasks.create('unlinked', MeasuredActionGoal(Ref('node:arbitrary'), 'activate_node', 'inputChecked', True))
    assert isinstance(extract_goal_example(agent, task.id), Unknown)
