"""Whole-request goal supervision; semantics and grounding are supplied fixtures."""
from dataclasses import replace

import pytest

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent import goal_interpretation as goals
from tensorcode.agent.goal_learning import (
    admit_goal_model, extract_goal_example, fit_goal_model, get_goal_model,
)
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame, Request
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref


def example(agent, name, *, skipped=()):
    item, destination, protected = (Ref(f'{name}:{role}') for role in ('item', 'destination', 'protected'))
    frames = (Frame('create', {'object': item, 'destination': destination}),
              Frame('keep', {'object': protected}, {'qualifier': 'existing'}))
    goal = GoalSpec((Condition('located', {'item': item, 'destination': destination}),),
                    invariants=(Condition('unchanged', {'item': protected}),))
    source = agent.interpretations.add_source('supplied complete request', provider='authored fixture')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, SentenceAlternative(
        None, tuple(Act('request', Request(frame), frame) for frame in frames), skipped=skipped))
    agent.interpretations.select(group.id, candidate.id, reason='supplied complete reading')
    dependency = capture_dependency(agent.interpretations, group.id, basis=('supplied reading',))
    return frames, goal, source, group, candidate, dependency


def teach(agent, row):
    assert hasattr(goals, 'retain_taught_sentence_goal'), 'whole-sentence goal teaching is absent'
    return goals.retain_taught_sentence_goal(agent, row[1], row[2].text,
        parent_dependency=row[5], reason='supplied whole-request goal')


def select(agent, group_id):
    group = agent.interpretations.get(group_id)
    return goals.select_goal(agent, group_id, decision=InterpretationDecision(
        group.candidates[0].id, 'supplied goal selection', compared_revision=group.revision,
        compared_candidate_ids=tuple(c.id for c in group.candidates)))


def trained_agent():
    agent = Agent()
    tasks = []
    for name in ('runtime-alpha', 'runtime-beta', 'runtime-heldout'):
        row = example(agent, name)
        group_id = teach(agent, row)
        result = select(agent, group_id)
        tasks.append(agent.tasks.create(row[2].text, result.goal,
            dependencies=(row[5], *result.supporting_dependencies, result.dependency),
            goal_interpretation_id=group_id))
    fitted = fit_goal_model(agent, [t.id for t in tasks[:2]], [tasks[2].id])
    assert not isinstance(fitted, Unknown), fitted
    agent.goal_model = admit_goal_model(agent, fitted, reason='supplied runtime model admission')
    assert not isinstance(agent.goal_model, Unknown), agent.goal_model
    return agent


def runtime_proposal(agent, row):
    assert hasattr(goals, 'retain_sentence_goal_proposals'), 'whole-sentence runtime goal projection is absent'
    return goals.retain_sentence_goal_proposals(agent, row[3].id, row[4].id,
        basis=('explicit complete input for goal interpretation',))


def test_retains_every_ordered_request_frame_and_all_taught_constraints():
    agent = Agent(); row = example(agent, 'first')
    group_id = teach(agent, row)
    assert not isinstance(group_id, Unknown), group_id
    source = agent.interpretations.get_source(agent.interpretations.get(group_id).source_id)
    assert source.payload['frame'].roles['requests'] == row[0]
    assert source.metadata['input_kind'] == 'complete-request-sequence'
    result = select(agent, group_id)
    assert result.goal == row[1]
    assert not agent.tasks.values()


def test_partial_sentence_cannot_teach_a_complete_goal():
    agent = Agent(); row = example(agent, 'partial', skipped=('but',))
    assert isinstance(teach(agent, row), Unknown)


def test_admitted_whole_sentence_correspondence_requires_both_clauses():
    agent = Agent(); rows = [example(agent, name) for name in ('alpha', 'beta', 'heldout')]
    tasks = []
    for row in rows:
        group_id = teach(agent, row)
        assert not isinstance(group_id, Unknown), group_id
        result = select(agent, group_id)
        assert not isinstance(result.goal, Unknown), result
        tasks.append(agent.tasks.create(row[2].text, result.goal,
            dependencies=(row[5], *result.supporting_dependencies, result.dependency),
            goal_interpretation_id=group_id))
    retained = extract_goal_example(agent, tasks[0].id)
    assert not isinstance(retained, Unknown), retained
    assert retained.example.frame.roles['requests'] == rows[0][0]
    model = fit_goal_model(agent, [task.id for task in tasks[:2]], [tasks[2].id])
    assert not isinstance(model, Unknown), model
    admitted = admit_goal_model(agent, model, reason='supplied whole-request model admission')
    assert not isinstance(admitted, Unknown), admitted
    learned = get_goal_model(agent, admitted)
    fresh = example(agent, 'fresh')
    full = replace(retained.example.frame, roles={'requests': fresh[0]})
    predicted = learned.propose(full)
    assert len(predicted.proposals) == 1
    assert predicted.proposals[0].goal.conditions == fresh[1].conditions
    assert predicted.proposals[0].goal.invariants == fresh[1].invariants
    assert not learned.propose(replace(full, roles={'requests': fresh[0][:1]})).proposals


@pytest.mark.parametrize('change', ['withdraw', 'mutate'])
def test_parent_change_prevents_goal_selection(change):
    agent = Agent(); row = example(agent, 'changing')
    group_id = teach(agent, row)
    assert not isinstance(group_id, Unknown), group_id
    if change == 'withdraw':
        agent.interpretations.unset(row[3].id, reason='withdraw reading')
    else:
        group = agent.interpretations._groups[row[3].id]
        forged = replace(group.candidates[0], payload=replace(row[4].payload, acts=row[4].payload.acts[:1]))
        agent.interpretations._groups[group.id] = replace(group, candidates=(forged,))
    assert isinstance(select(agent, group_id).goal, Unknown)


def test_runtime_goal_proposal_consumes_complete_sentence_and_admitted_model():
    agent = trained_agent(); row = example(agent, 'runtime-fresh')
    before = agent.tasks.values()
    group_id = runtime_proposal(agent, row)
    assert not isinstance(group_id, Unknown), group_id
    group = agent.interpretations.get(group_id)
    assert group.selected_id is None
    source = agent.interpretations.get_source(group.source_id)
    assert source.payload['frame'].roles['requests'] == row[0]
    result = select(agent, group_id)
    assert result.goal.conditions == row[1].conditions
    assert result.goal.invariants == row[1].invariants
    assert agent.goal_model.dependency in result.supporting_dependencies
    assert agent.tasks.values() == before


def test_runtime_goal_projection_has_no_unconfigured_model_fallback():
    agent = Agent(); row = example(agent, 'no-model')
    assert isinstance(runtime_proposal(agent, row), Unknown)


@pytest.mark.parametrize('change', ['withdraw', 'mutate', 'model'])
def test_runtime_goal_selection_revalidates_source_and_model(change):
    agent = trained_agent(); row = example(agent, 'runtime-changing')
    group_id = runtime_proposal(agent, row)
    assert not isinstance(group_id, Unknown), group_id
    if change == 'withdraw':
        agent.interpretations.unset(row[3].id, reason='withdraw input')
    elif change == 'model':
        agent.interpretations.unset(agent.goal_model.group_id, reason='withdraw model')
    else:
        group = agent.interpretations._groups[row[3].id]
        forged = replace(group.candidates[0], payload=replace(row[4].payload, acts=row[4].payload.acts[:1]))
        agent.interpretations._groups[group.id] = replace(group, candidates=(forged,))
    assert isinstance(select(agent, group_id).goal, Unknown)


def test_runtime_retention_rechecks_reading_after_publication_callbacks(monkeypatch):
    agent = trained_agent(); row = example(agent, 'runtime-race')
    propose = agent.interpretations.propose
    def withdraw(*args, **kwargs):
        result = propose(*args, **kwargs)
        agent.interpretations.unset(row[3].id, reason='changed during goal publication')
        return result
    monkeypatch.setattr(agent.interpretations, 'propose', withdraw)
    assert isinstance(runtime_proposal(agent, row), Unknown)


@pytest.mark.parametrize('mode', ['taught', 'runtime'])
def test_selection_rechecks_sentence_after_final_selection_callback(monkeypatch, mode):
    agent = Agent() if mode == 'taught' else trained_agent()
    row = example(agent, 'late-selection-' + mode)
    group_id = teach(agent, row) if mode == 'taught' else runtime_proposal(agent, row)
    original = agent.interpretations.select
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[0] == group_id:
            parent = agent.interpretations._groups[row[3].id]
            forged = replace(parent.candidates[0], payload=replace(row[4].payload, acts=row[4].payload.acts[:1]))
            agent.interpretations._groups[parent.id] = replace(parent, candidates=(forged,))
        return result
    monkeypatch.setattr(agent.interpretations, 'select', changed)
    assert isinstance(select(agent, group_id).goal, Unknown)


def test_teaching_rechecks_parent_after_final_evidence_copy(monkeypatch):
    agent = Agent(); row = example(agent, 'late-teaching')
    original = agent.interpretations.get_source
    def withdrawn(source_id):
        result = original(source_id)
        if result.provider == 'explicit-goal-teaching':
            agent.interpretations.unset(row[3].id, reason='withdraw during final source copy')
        return result
    monkeypatch.setattr(agent.interpretations, 'get_source', withdrawn)
    assert isinstance(teach(agent, row), Unknown)


@pytest.mark.parametrize('change', ['select', 'rival'])
def test_runtime_publication_cannot_change_proposal_comparison(monkeypatch, change):
    agent = trained_agent(); row = example(agent, 'publication-' + change)
    original = agent.interpretations.propose
    def changed(*args, **kwargs):
        candidate = original(*args, **kwargs)
        if change == 'select':
            agent.interpretations.select(candidate.group_id, candidate.id, reason='publication callback')
        else:
            original(candidate.group_id, Unknown('additional rival'), provenance=('unexpected',))
        return candidate
    monkeypatch.setattr(agent.interpretations, 'propose', changed)
    assert isinstance(runtime_proposal(agent, row), Unknown)
