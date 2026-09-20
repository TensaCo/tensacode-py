"""Authentic retained teaching and explicit model admission, not inferred intent."""
from dataclasses import replace

import pytest

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.goal_interpretation import retain_goal_proposals, select_goal
from tensorcode.agent.goal_learning import (
    GoalModelHandle, admit_goal_model, extract_goal_example, fit_goal_model, get_goal_model,
)
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame, verbnet
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref


def taught_task(agent, monkeypatch, name, *, predicate="ready"):
    identity = Ref(f'device:{name}')
    frame = Frame('prepare', {'object': identity}, {'mood': 'imperative'})
    goal = GoalSpec((Condition(predicate, {'target': identity}),), basis=('explicit fixture teaching',))
    workspace = agent.interpretations
    source = workspace.add_source(f'prepare {name}', provider='authored input fixture')
    parent = workspace.create_group(source.id)
    candidate = workspace.propose(parent.id, frame, provenance=('supplied frame, not inferred text',))
    workspace.select(parent.id, candidate.id, reason='explicit teaching interpretation')
    dependency = capture_dependency(workspace, parent.id, basis=('authored teaching sentence',))
    lexical = verbnet.Goal('prepare', 'authored:prepare', goal.conditions, frame)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw:
                        verbnet.GoalCandidates((verbnet.GoalProposal(lexical, ()),)))
    group_id = retain_goal_proposals(agent, frame, source.text, parent_dependency=dependency)
    group = workspace.get(group_id)
    resolution = select_goal(agent, group_id, decision=InterpretationDecision(
        group.candidates[0].id, 'explicit teaching goal', compared_revision=group.revision,
        compared_candidate_ids=tuple(c.id for c in group.candidates)))
    assert not isinstance(resolution.goal, Unknown)
    task = agent.tasks.create(source.text, goal, dependencies=(dependency, resolution.dependency),
                              goal_interpretation_id=group_id)
    return task


def dataset(monkeypatch):
    agent = Agent()
    tasks = [taught_task(agent, monkeypatch, name) for name in ('train-a', 'train-b', 'heldout-c')]
    return agent, tasks


def fit(agent, tasks, **kwargs):
    return fit_goal_model(agent, [tasks[0].id, tasks[1].id], [tasks[2].id], **kwargs)


def test_extracts_exact_task_revision_and_original_frame(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    record = extract_goal_example(agent, tasks[0].id)
    assert not isinstance(record, Unknown)
    assert record.task_revision == 1 and record.dependencies == tasks[0].dependencies
    assert record.example.frame.roles == {'object': Ref('device:train-a')}
    assert record.example.goal == tasks[0].goal
    source = agent.interpretations.get_source(record.evidence_source_id)
    assert source.provider == 'retained-task-goal-supervision'
    assert source.payload == record.example


def test_arbitrary_mimicked_goal_source_cannot_teach(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    original = agent.interpretations.get(tasks[0].goal_interpretation_id)
    source = agent.interpretations.get_source(original.source_id)
    clone_source = agent.interpretations.add_source(source.text, modality=source.modality,
        provider=source.provider, payload=source.payload, metadata=source.metadata)
    clone_group = agent.interpretations.create_group(clone_source.id, provenance=original.provenance)
    candidate = agent.interpretations.propose(clone_group.id, original.selected.payload)
    agent.interpretations.select(clone_group.id, candidate.id, reason='mimic')
    dep = capture_dependency(agent.interpretations, clone_group.id, basis=('mimic',))
    fake = agent.tasks.create('mimicked teaching', tasks[0].goal,
                             dependencies=(tasks[0].dependencies[0], dep), goal_interpretation_id=clone_group.id)
    result = extract_goal_example(agent, fake.id)
    assert isinstance(result, Unknown) and 'unrecognized' in result.detail


def test_unknown_goal_and_stale_dependencies_cannot_teach(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    unknown = agent.tasks.create('unknown', Unknown('not_understood'))
    assert isinstance(extract_goal_example(agent, unknown.id), Unknown)
    agent.interpretations.unset(tasks[0].goal_interpretation_id, reason='withdraw interpretation')
    assert isinstance(extract_goal_example(agent, tasks[0].id), Unknown)


def test_fit_requires_admission_and_admitted_model_generalizes_to_new_reference(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    handle = fit(agent, tasks)
    assert isinstance(handle, GoalModelHandle), handle
    assert handle.dependency is None
    assert isinstance(get_goal_model(agent, handle), Unknown)
    admitted = admit_goal_model(agent, handle, reason='explicit model version admission')
    assert isinstance(admitted, GoalModelHandle), admitted
    model = get_goal_model(agent, admitted)
    assert not isinstance(model, Unknown), model
    prediction = model.propose(Frame('prepare', {'object': Ref('device:unseen')}, {'mood': 'imperative'}))
    assert prediction.complete and len(prediction.proposals) == 1
    assert prediction.proposals[0].goal.conditions[0].args == {'target': Ref('device:unseen')}
    assert not agent.store.propositions() and not agent.store.claims()


def test_overlapping_task_splits_cannot_publish_a_model(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    result = fit_goal_model(agent, [tasks[0].id, tasks[1].id], [tasks[1].id])
    assert isinstance(result, Unknown) and 'disjoint' in result.detail
    assert not getattr(agent, '_goal_learning_models', {})


def test_task_revision_changed_by_fitting_prevents_publication(monkeypatch):
    from tensorcode.agent import goal_learning
    agent, tasks = dataset(monkeypatch)
    original = goal_learning.fit_correspondences
    def changes(*args, **kwargs):
        model = original(*args, **kwargs)
        agent.tasks.revise(tasks[0].id, tasks[0].goal, reason='changed teaching during fitting')
        return model
    monkeypatch.setattr(goal_learning, 'fit_correspondences', changes)
    result = fit(agent, tasks)
    assert isinstance(result, Unknown)
    assert not getattr(agent, '_goal_learning_models', {})


def test_historical_model_survives_later_task_revision_but_refit_withdraws_admission(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    handle = fit(agent, tasks)
    admitted = admit_goal_model(agent, handle, reason='admit first')
    historical = agent.interpretations.get_source(handle.evidence_source_id)
    # Later teaching changes do not rewrite successful historical fitting evidence.
    agent.tasks.revise(tasks[0].id, tasks[0].goal, reason='new task revision',
                       goal_interpretation_id=tasks[0].goal_interpretation_id)
    assert not isinstance(get_goal_model(agent, admitted), Unknown)
    newer = fit(agent, tasks, group_id=handle.group_id)
    assert isinstance(newer, GoalModelHandle), newer
    assert newer.group_id == handle.group_id and newer.candidate_id != handle.candidate_id
    assert agent.interpretations.get(newer.group_id).selected_id is None
    assert isinstance(get_goal_model(agent, admitted), Unknown)
    assert agent.interpretations.get_source(handle.evidence_source_id) == historical


def test_model_candidate_or_handle_tampering_cannot_authorize_a_model(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    handle = admit_goal_model(agent, fit(agent, tasks), reason='admit')
    assert isinstance(get_goal_model(agent, replace(handle, model_id='fabricated')), Unknown)
    # Preserve identity while changing the retained state: ID equality is insufficient.
    candidate = agent.interpretations._groups[handle.group_id].selected
    candidate.payload.snapshot['_complete'] = False
    assert isinstance(get_goal_model(agent, handle), Unknown)


def test_task_change_during_extraction_cannot_register_teaching(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    original = agent.tasks.get
    changed = False
    def get(identity):
        nonlocal changed
        value = original(identity)
        if not changed:
            changed = True
            agent.tasks.revise(identity, value.goal, reason='change during extraction',
                               goal_interpretation_id=value.goal_interpretation_id)
        return value
    monkeypatch.setattr(agent.tasks, 'get', get)
    assert isinstance(extract_goal_example(agent, tasks[0].id), Unknown)
    assert not getattr(agent, '_goal_learning_examples', {})


def test_failed_refit_publication_keeps_prior_versions_authentic_and_readmittable(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    original_handle = fit(agent, tasks)
    admitted = admit_goal_model(agent, original_handle, reason='admit original')
    assert not isinstance(admitted, Unknown)
    original = agent.interpretations.propose
    changed = False
    def propose(group_id, payload, **kwargs):
        nonlocal changed
        candidate = original(group_id, payload, **kwargs)
        if group_id == admitted.group_id and not changed:
            changed = True
            agent.tasks.revise(tasks[0].id, tasks[0].goal, reason='teaching changed during publication')
        return candidate
    monkeypatch.setattr(agent.interpretations, 'propose', propose)
    result = fit(agent, tasks, group_id=admitted.group_id)
    assert isinstance(result, Unknown)
    group = agent.interpretations.get(admitted.group_id)
    assert len(group.candidates) == 2 and group.candidates[-1].rejected
    failed = group.candidates[-1]
    rejected_handle = GoalModelHandle(group.id, failed.id, failed.payload.snapshot['_id'],
                                      failed.payload.evidence_source_id)
    assert isinstance(admit_goal_model(agent, rejected_handle, reason='cannot admit failed fit'), Unknown)
    readmitted = admit_goal_model(agent, original_handle, reason='readmit previous historical version')
    assert isinstance(readmitted, GoalModelHandle), readmitted
    assert not isinstance(get_goal_model(agent, readmitted), Unknown)


def test_admission_cannot_capture_a_different_selection_epoch(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    handle = fit(agent, tasks)
    original = agent.interpretations.select
    def select(group_id, candidate_id, **kwargs):
        result = original(group_id, candidate_id, **kwargs)
        original(group_id, candidate_id, reason='reentrant selection of same model')
        return result
    monkeypatch.setattr(agent.interpretations, 'select', select)
    result = admit_goal_model(agent, handle, reason='original admission')
    assert isinstance(result, Unknown) and 'changed during selection' in result.detail


def test_later_example_validation_cannot_change_an_already_checked_teacher(monkeypatch):
    from tensorcode.agent import goal_learning
    agent, tasks = dataset(monkeypatch)
    original_fit, original_get = goal_learning.fit_correspondences, agent.tasks.get
    state = {'fitted': False, 'changed': False}
    def fitting(*args, **kwargs):
        model = original_fit(*args, **kwargs)
        state['fitted'] = True
        return model
    def get(identity):
        task = original_get(identity)
        if state['fitted'] and identity == tasks[2].id and not state['changed']:
            state['changed'] = True
            agent.tasks.revise(tasks[0].id, tasks[0].goal, reason='last check changed first teacher',
                               goal_interpretation_id=tasks[0].goal_interpretation_id)
        return task
    monkeypatch.setattr(goal_learning, 'fit_correspondences', fitting)
    monkeypatch.setattr(agent.tasks, 'get', get)
    assert isinstance(fit(agent, tasks), Unknown)
    assert not getattr(agent, '_goal_learning_models', {})


def test_admitted_singleton_rival_cannot_be_skipped_by_execution_policy(monkeypatch):
    from tensorcode.agent.understand import Act, Sentence
    from tensorcode.language import Request
    agent, tasks = dataset(monkeypatch)
    rival = taught_task(agent, monkeypatch, 'rival', predicate='not-ready')
    handle = fit_goal_model(agent, [tasks[0].id, tasks[1].id, rival.id], [tasks[2].id])
    agent.goal_model = admit_goal_model(agent, handle, reason='explicitly admit retained teaching')
    assert not isinstance(agent.goal_model, Unknown)
    selections, dispatches = [], []
    def choose(group):
        selections.append(group.id)
        return InterpretationDecision(group.candidates[0].id, 'prefer ready')
    agent.goal_selector = choose
    monkeypatch.setattr(agent, '_invoke', lambda *a, **kw: dispatches.append(a))
    frame = Frame('prepare', {'object': Ref('device:fresh')}, {'mood': 'imperative'})
    act = Act('request', Request(frame), frame)
    sentence = Sentence('prepare fresh', ('prepare', 'fresh'), None, (act,))
    outcome = agent.request(sentence, act, [])
    assert outcome.status == 'unknown' and isinstance(outcome.goal, Unknown)
    assert outcome.goal.reason == 'goal_correspondence_unresolved'
    assert not selections and not dispatches and outcome.receipt is None
    group = agent.interpretations.get(outcome.goal_interpretation_id)
    assert group.selected is None
    proposal = group.candidates[0].payload
    assert proposal.conflicting_training_example_ids
    assert any(str(c.payload).startswith('unrepresented_training_rival:') for c in group.candidates)
    # A delayed explicit choice cannot bypass the same obligation either.
    decision = InterpretationDecision(group.candidates[0].id, 'force preferred mapping',
        compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))
    result = select_goal(agent, group.id, decision=decision)
    assert isinstance(result.goal, Unknown) and result.goal.reason == 'goal_correspondence_unresolved'
    assert agent.interpretations.get(group.id).selected is None


def test_all_supported_rivals_remain_explicitly_selectable(monkeypatch):
    agent, tasks = dataset(monkeypatch)
    rivals = [taught_task(agent, monkeypatch, name, predicate='not-ready')
              for name in ('rival-a', 'rival-b', 'rival-held')]
    handle = fit_goal_model(agent, [tasks[0].id, tasks[1].id, rivals[0].id, rivals[1].id],
                            [tasks[2].id, rivals[2].id])
    agent.goal_model = admit_goal_model(agent, handle, reason='admit both supported readings')
    frame = Frame('prepare', {'object': Ref('device:fresh')}, {'mood': 'imperative'})
    group_id = retain_goal_proposals(agent, frame, 'prepare fresh')
    group = agent.interpretations.get(group_id)
    assert len(group.candidates) == 2
    assert all(c.payload.conflicting_training_example_ids for c in group.candidates)
    candidate = next(c for c in group.candidates if c.payload.goal.conditions[0].pred == 'not-ready')
    resolution = select_goal(agent, group_id, decision=InterpretationDecision(candidate.id,
        'explicitly choose taught rival', compared_revision=group.revision,
        compared_candidate_ids=tuple(c.id for c in group.candidates)))
    assert isinstance(resolution.goal, GoalSpec)
    assert resolution.goal.conditions[0].pred == 'not-ready'
