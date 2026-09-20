"""Explicit contextual labels and authentic, separately admitted model versions."""
from dataclasses import replace
import importlib
import pytest
from tensorcode.agent import Agent
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame
from tensorcode.language.semantics import Request
from tensorcode.records import Ref
from tensorcode.outcomes import Unknown


def api():
    import importlib.util
    assert importlib.util.find_spec('tensorcode.agent.task_revision_learning') is not None, 'retained contextual learning API missing'
    return importlib.import_module('tensorcode.agent.task_revision_learning')


def sample(agent, name, *, malformed=None):
    item, old, new, protected = (Ref(name + ':' + r) for r in ('item', 'old', 'new', 'protected'))
    previous = GoalSpec((Condition('located', {'item': item, 'destination': old}),),
                        invariants=(Condition('unchanged', {'item': protected}),))
    revised = replace(previous, conditions=(Condition('located', {'item': item, 'destination': new}),))
    task = agent.tasks.create('supplied original goal', previous)
    frames = (Frame('change', {'destination': new}), Frame('preserve', {'item': protected}, {'qualifier': 'existing'}))
    acts = tuple(Act('request', Request(f), f) for f in frames)
    if malformed == 'fragment': acts = (*acts[:1], Act('fragment', frames[1], frames[1]))
    if malformed == 'mismatch': acts = (Act('request', Request(frames[1]), frames[0]), acts[1])
    source = agent.interpretations.add_source('supplied full correction', provider='authored structural fixture')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, SentenceAlternative(None, acts, skipped=('extra',) if malformed == 'skipped' else ()))
    agent.interpretations.select(group.id, candidate.id, reason='supplied interpretation')
    return task, group, candidate, revised, frames


def retain(agent, data):
    task, group, candidate, revised, _ = data
    result = api().retain_task_revision_example(agent, task.id, group.id, candidate.id, revised, basis=('explicit teacher',))
    assert not isinstance(result, Unknown), result
    return result


def dataset():
    agent = Agent()
    data = [sample(agent, name) for name in ('a', 'b', 'held')]
    return agent, data, [retain(agent, row) for row in data]


def test_retains_all_ordered_frames_and_stale_context_fails():
    agent = Agent(); data = sample(agent, 'a'); record = retain(agent, data)
    assert record.example.corrections == data[4]
    assert record.example.previous == data[0].goal
    assert api().validate_task_revision_context(agent, record.context) is True
    agent.tasks.revise(data[0].id, data[3], reason='intervening correction')
    assert isinstance(api().validate_task_revision_context(agent, record.context), Unknown)


@pytest.mark.parametrize('malformed', ['fragment', 'mismatch', 'skipped'])
def test_rejects_partial_or_inconsistent_reading(malformed):
    agent = Agent(); task, group, candidate, revised, _ = sample(agent, 'bad', malformed=malformed)
    result = api().retain_task_revision_example(agent, task.id, group.id, candidate.id, revised, basis=('teacher',))
    assert isinstance(result, Unknown)


def test_fit_admission_reference_transfer_and_historical_supervision():
    agent, data, records = dataset(); module = api()
    handle = module.fit_task_revision_model(agent, records[:2], records[2:])
    assert not isinstance(handle, Unknown), handle
    assert isinstance(module.get_task_revision_model(agent, handle), Unknown)
    admitted = module.admit_task_revision_model(agent, handle, reason='explicit admission')
    assert not isinstance(admitted, Unknown), admitted
    agent.tasks.revise(data[0][0].id, data[0][3], reason='later teaching task revision')
    model = module.get_task_revision_model(agent, admitted)
    assert not isinstance(model, Unknown), model
    fresh = sample(agent, 'fresh')
    proposed = model.propose(fresh[0].goal, fresh[4])
    assert len(proposed.proposals) == 1
    assert proposed.proposals[0].goal.conditions == fresh[3].conditions
    assert proposed.proposals[0].goal.invariants == fresh[3].invariants
    assert isinstance(module.fit_task_revision_model(agent, records[:2], records[2:]), Unknown)


def test_forged_record_rejected_and_refit_requires_readmission():
    agent, data, records = dataset(); module = api()
    forged = replace(records[0], example=replace(records[0].example, revised=replace(data[0][3], invariants=())))
    assert isinstance(module.fit_task_revision_model(agent, (forged, records[1]), records[2:]), Unknown)
    original = module.fit_task_revision_model(agent, records[:2], records[2:])
    admitted = module.admit_task_revision_model(agent, original, reason='first')
    refit = module.fit_task_revision_model(agent, records[:2], records[2:], group_id=original.group_id)
    assert not isinstance(refit, Unknown), refit
    assert isinstance(module.get_task_revision_model(agent, admitted), Unknown)
    readmitted = module.admit_task_revision_model(agent, original, reason='old authentic version')
    assert not isinstance(module.get_task_revision_model(agent, readmitted), Unknown)


def test_explicitly_incomplete_projection_cannot_be_retained():
    agent = Agent(); task, group, candidate, revised, _ = sample(agent, 'incomplete')
    alternative = replace(candidate.payload, metadata={'semantic_projection_complete': False})
    child = agent.interpretations.propose(group.id, alternative)
    agent.interpretations.select(group.id, child.id, reason='explicit but incomplete reading')
    assert isinstance(api().retain_task_revision_example(agent, task.id, group.id, child.id, revised, basis=('teacher',)), Unknown)


@pytest.mark.parametrize('covered', [True, False])
def test_learned_source_anchors_cover_every_nonwhitespace_character(covered):
    agent = Agent(); task, group, candidate, revised, _ = sample(agent, 'anchored')
    text = 'change preserve' + ('' if covered else ' omitted')
    source = agent.interpretations.add_source(text, provider='learned-reader')
    new_group = agent.interpretations.create_group(source.id)
    alternative = replace(candidate.payload, provenance='learned-speech-acts', metadata={
        'tokens': ('change', 'preserve'), 'semantic_projection_complete': None,
        'token_anchors': ({'index': 1, 'token': 'change', 'char_span': (0, 6)},
                          {'index': 2, 'token': 'preserve', 'char_span': (7, 15)})})
    child = agent.interpretations.propose(new_group.id, alternative)
    agent.interpretations.select(new_group.id, child.id, reason='explicit anchored reading')
    result = api().retain_task_revision_example(agent, task.id, new_group.id, child.id, revised, basis=('teacher',))
    assert isinstance(result, Unknown) is not covered


def test_source_and_nested_model_forgery_are_rejected():
    agent, data, records = dataset(); module = api()
    evidence = agent.interpretations._sources[records[0].evidence_source_id]
    agent.interpretations._sources[evidence.id] = replace(evidence, text='tampered teaching')
    assert isinstance(module.fit_task_revision_model(agent, records[:2], records[2:]), Unknown)
    agent.interpretations._sources[evidence.id] = evidence
    handle = module.fit_task_revision_model(agent, records[:2], records[2:])
    admitted = module.admit_task_revision_model(agent, handle, reason='explicit admission')
    assert not isinstance(module.get_task_revision_model(agent, admitted), Unknown)
    retained_model = agent._task_revision_models[handle.group_id].versions[0][3]
    retained_model._correspondence._templates = ()
    assert isinstance(module.get_task_revision_model(agent, admitted), Unknown)


def test_later_record_read_cannot_stale_earlier_task_in_batch(monkeypatch):
    agent, data, records = dataset(); workspace = agent.interpretations
    original = workspace.get_source
    fired = []
    def changing_read(identity):
        result = original(identity)
        if identity == records[-1].evidence_source_id and not fired:
            fired.append(True)
            agent.tasks.revise(data[0][0].id, data[0][3], reason='changed while reading later teaching')
        return result
    monkeypatch.setattr(workspace, 'get_source', changing_read)
    assert isinstance(api().fit_task_revision_model(agent, records[:2], records[2:]), Unknown)
    assert fired


def test_failed_publication_keeps_older_authentic_version_readmittable(monkeypatch):
    agent, data, records = dataset(); module = api(); workspace = agent.interpretations
    original_handle = module.fit_task_revision_model(agent, records[:2], records[2:])
    original = workspace.propose
    def changing_publication(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[0] == original_handle.group_id:
            agent.tasks.revise(data[0][0].id, data[0][3], reason='changed during publication')
        return result
    monkeypatch.setattr(workspace, 'propose', changing_publication)
    result = module.fit_task_revision_model(agent, records[:2], records[2:], group_id=original_handle.group_id)
    assert isinstance(result, Unknown)
    group = workspace.get(original_handle.group_id)
    assert len(group.candidates) == 2 and group.candidates[-1].rejected
    admitted = module.admit_task_revision_model(agent, original_handle, reason='readmit authentic older historical version')
    assert not isinstance(module.get_task_revision_model(agent, admitted), Unknown)


def test_unrecognized_goal_group_cannot_authorize_context():
    agent = Agent(); task, group, candidate, revised, _ = sample(agent, 'forged-group')
    from tensorcode.agent.task_dependencies import capture_dependency
    dependency = capture_dependency(agent.interpretations, group.id, basis=('unrelated group',))
    forged = agent.tasks.create('forged linked task', task.goal, dependencies=(dependency,), goal_interpretation_id=group.id)
    assert isinstance(api().capture_task_revision_context(agent, forged.id, group.id, candidate.id, basis=('teacher',)), Unknown)


def test_changed_correction_and_shared_heldout_sources_cannot_fit():
    agent, data, records = dataset(); module = api()
    task, group, candidate, revised, _ = data[0]
    duplicate_task = agent.tasks.create('different task, same correction source', task.goal)
    duplicate_record = module.retain_task_revision_example(agent, duplicate_task.id, group.id, candidate.id, revised, basis=('teacher',))
    assert not isinstance(duplicate_record, Unknown), duplicate_record
    assert isinstance(module.fit_task_revision_model(agent, records[:2], (duplicate_record,)), Unknown)
    agent.interpretations.unset(group.id, reason='withdraw correction')
    assert isinstance(module.fit_task_revision_model(agent, records[:2], records[2:]), Unknown)


def test_returned_context_is_detached_and_forged_admission_is_rejected():
    agent, data, records = dataset(); module = api()
    records[0].context.previous.conditions[0].args.clear()
    assert agent.tasks.get(data[0][0].id).goal.conditions[0].args
    assert isinstance(module.fit_task_revision_model(agent, records[:2], records[2:]), Unknown)
    agent, data, records = dataset()
    handle = module.fit_task_revision_model(agent, records[:2], records[2:])
    admitted = module.admit_task_revision_model(agent, handle, reason='explicit')
    forged = replace(admitted, dependency=replace(admitted.dependency, basis=('forged',)))
    assert isinstance(module.get_task_revision_model(agent, forged), Unknown)
