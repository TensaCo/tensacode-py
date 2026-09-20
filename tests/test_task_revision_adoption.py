"""Contextual adoption mechanics with supplied structural teaching and task identity."""
import importlib
import importlib.util
from dataclasses import replace
from types import SimpleNamespace
import pytest
from tensorcode.agent import InterpretationDecision
from tensorcode.outcomes import Unknown
from test_task_revision_learning import dataset, sample
from tensorcode.agent.task_revision_learning import (fit_task_revision_model, admit_task_revision_model,
    capture_task_revision_context)


def api():
    assert importlib.util.find_spec('tensorcode.agent.task_revision') is not None, 'task correction proposal/adoption API missing'
    return importlib.import_module('tensorcode.agent.task_revision')


def setup():
    agent, _, records = dataset()
    handle = fit_task_revision_model(agent, records[:2], records[2:])
    admitted = admit_task_revision_model(agent, handle, reason='supplied model admission')
    assert not isinstance(admitted, Unknown), admitted
    fresh = sample(agent, 'fresh')
    return agent, admitted, fresh


def propose(agent, model, row):
    task, correction, candidate, _, _ = row
    group_id = api().propose_task_revision(agent, task.id, correction.id, candidate.id,
                                         model=model, basis=('supplied task association',))
    assert not isinstance(group_id, Unknown), group_id
    return group_id


def decision(agent, group_id, index=0):
    group = agent.interpretations.get(group_id)
    return InterpretationDecision(group.candidates[index].id, 'explicit compared correction',
        compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))


def test_adopts_same_task_retaining_invariants_history_receipts_without_actions(monkeypatch):
    agent, model, row = setup(); task = row[0]
    receipt = {'completed': 'original step'}
    agent.tasks.record(task.id, SimpleNamespace(status='unverified', plan='old plan', receipt=receipt, verified=None, reason='partial'))
    monkeypatch.setattr(agent, 'perceive', lambda *a, **kw: pytest.fail('adoption must not perceive'))
    monkeypatch.setattr(agent, 'pursue', lambda *a, **kw: pytest.fail('adoption must not execute'))
    group_id = propose(agent, model, row)
    assert agent.tasks.current_revision(task.id) == 1
    assert agent.interpretations.get(group_id).selected_id is None
    adopted = api().adopt_task_revision(agent, task.id, group_id, decision(agent, group_id), reason='correct destination')
    assert not isinstance(adopted, Unknown), adopted
    assert adopted.id == task.id and adopted.revision == 2
    assert adopted.goal.conditions == row[3].conditions and adopted.goal.invariants == row[3].invariants
    assert adopted.revisions[0].goal == task.goal and adopted.attempts[0].receipt == receipt
    assert adopted.attempts[0].revision == 1 and adopted.attempts[0].plan == 'old plan'
    assert adopted.goal_interpretation_id == group_id
    assert model.dependency in adopted.dependencies
    assert api().validate_retained_revision_group(agent, group_id) is True
    next_context = capture_task_revision_context(agent, task.id, row[1].id, row[2].id, basis=('next explicit correction',))
    assert not isinstance(next_context, Unknown), next_context
    assert next_context.task_revision == 2 and next_context.previous == adopted.goal


@pytest.mark.parametrize('changed', ['task', 'comparison', 'correction', 'model'])
def test_stale_or_withdrawn_evidence_cannot_revise(changed):
    agent, model, row = setup(); task = row[0]; group_id = propose(agent, model, row)
    choice = decision(agent, group_id)
    if changed == 'task': agent.tasks.revise(task.id, task.goal, reason='intervening task revision')
    if changed == 'comparison': agent.interpretations.unset(group_id, reason='comparison changed')
    if changed == 'correction': agent.interpretations.unset(row[1].id, reason='withdraw correction')
    if changed == 'model': agent.interpretations.unset(model.group_id, reason='withdraw model')
    before = agent.tasks.current_revision(task.id)
    result = api().adopt_task_revision(agent, task.id, group_id, choice, reason='late correction')
    assert isinstance(result, Unknown)
    assert agent.tasks.current_revision(task.id) == before


@pytest.mark.parametrize('changed', ['source', 'candidate', 'registry'])
def test_tampered_retained_group_cannot_adopt(changed):
    agent, model, row = setup(); group_id = propose(agent, model, row); choice = decision(agent, group_id)
    workspace = agent.interpretations; group = workspace.get(group_id)
    if changed == 'source':
        source = workspace._sources[group.source_id]
        workspace._sources[source.id] = replace(source, text='forged')
    if changed == 'candidate':
        candidate = group.candidates[0]
        forged = replace(candidate, payload=replace(candidate.payload, goal=replace(row[3], invariants=())))
        workspace._groups[group_id] = replace(group, candidates=(forged,))
    if changed == 'registry': del agent._task_revision_groups[group_id]
    assert isinstance(api().adopt_task_revision(agent, row[0].id, group_id, choice, reason='forged'), Unknown)
    assert agent.tasks.current_revision(row[0].id) == 1


def test_reentrant_task_revision_during_selection_cannot_be_overwritten(monkeypatch):
    agent, model, row = setup(); group_id = propose(agent, model, row)
    original = agent.interpretations.select
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[0] == group_id:
            agent.tasks.revise(row[0].id, row[0].goal, reason='concurrent revision')
        return result
    monkeypatch.setattr(agent.interpretations, 'select', changed)
    assert isinstance(api().adopt_task_revision(agent, row[0].id, group_id, decision(agent, group_id), reason='stale'), Unknown)
    current = agent.tasks.get(row[0].id)
    assert current.revision == 2 and current.goal == row[0].goal


def test_final_ledger_copy_callback_withdraws_correction_before_commit(monkeypatch):
    from tensorcode.agent import tasks as ledger
    agent, model, row = setup(); group_id = propose(agent, model, row)
    original = ledger.deepcopy
    fired = []
    def changing_copy(value):
        result = original(value)
        if type(value) is ledger.Task and value.id == row[0].id and value.revision == 2 and not fired:
            fired.append(True)
            agent.interpretations.unset(row[1].id, reason='withdraw during final returned-task copy')
        return result
    monkeypatch.setattr(ledger, 'deepcopy', changing_copy)
    result = api().adopt_task_revision(agent, row[0].id, group_id, decision(agent, group_id), reason='late')
    assert isinstance(result, Unknown) and fired
    assert agent.tasks.current_revision(row[0].id) == 1


def test_wrong_task_and_unspecified_comparison_do_not_adopt():
    agent, model, row = setup(); group_id = propose(agent, model, row)
    wrong = agent.tasks.create('other task', row[0].goal)
    assert isinstance(api().adopt_task_revision(agent, wrong.id, group_id, decision(agent, group_id), reason='wrong task'), Unknown)
    unspecified = InterpretationDecision(agent.interpretations.get(group_id).candidates[0].id, 'missing comparison')
    assert isinstance(api().adopt_task_revision(agent, row[0].id, group_id, unspecified, reason='missing comparison'), Unknown)
    assert agent.tasks.current_revision(row[0].id) == agent.tasks.current_revision(wrong.id) == 1


def test_unsupported_context_retains_unresolved_evidence_without_selecting():
    agent, model, row = setup()
    changed_goal = replace(row[0].goal, invariants=())
    task = agent.tasks.revise(row[0].id, changed_goal, reason='unsupported prior constraint')
    row = (task, *row[1:])
    group_id = propose(agent, model, row)
    group = agent.interpretations.get(group_id)
    source = agent.interpretations.get_source(group.source_id)
    assert not source.payload.proposals and source.payload.unresolved
    assert len(group.candidates) == len(source.payload.unresolved) and group.selected_id is None
    assert isinstance(api().adopt_task_revision(agent, task.id, group_id, decision(agent, group_id), reason='try unresolved'), Unknown)
    assert agent.tasks.current_revision(task.id) == 2


def test_complete_competing_teaching_retains_every_goal_until_explicit_choice():
    from tensorcode.agent import Agent
    from tensorcode.agent.task_revision_learning import retain_task_revision_example
    agent = Agent(); records = []
    for name, keep in (('a', True), ('b', True), ('c', False), ('d', False), ('held-a', True), ('held-b', False)):
        row = sample(agent, name)
        revised = row[3] if keep else replace(row[3], invariants=())
        record = retain_task_revision_example(agent, row[0].id, row[1].id, row[2].id, revised, basis=('competing explicit teacher',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    model = admit_task_revision_model(agent, fit_task_revision_model(agent, records[:4], records[4:]), reason='admit competing supplied teaching')
    assert not isinstance(model, Unknown), model
    row = sample(agent, 'fresh'); group_id = propose(agent, model, row)
    group = agent.interpretations.get(group_id)
    assert len(group.candidates) == 2 and group.selected_id is None
    assert {len(c.payload.goal.invariants) for c in group.candidates} == {0, 1}
    assert agent.tasks.current_revision(row[0].id) == 1
    keep_index = next(i for i, c in enumerate(group.candidates) if c.payload.goal.invariants)
    adopted = api().adopt_task_revision(agent, row[0].id, group_id, decision(agent, group_id, keep_index), reason='explicit preservation choice')
    assert not isinstance(adopted, Unknown), adopted
    assert adopted.goal.invariants == row[3].invariants


def test_successive_adoption_preserves_prior_goal_dependencies():
    agent, model, row = setup(); first_group = propose(agent, model, row)
    first = api().adopt_task_revision(agent, row[0].id, first_group, decision(agent, first_group), reason='first correction')
    assert not isinstance(first, Unknown), first
    # A distinct correction source supplies a fresh destination while preserving
    # the prior goal shape learned from independent references.
    from tensorcode.agent.understand import Act, SentenceAlternative
    from tensorcode.language import Frame, Request
    from tensorcode.records import Ref
    source = agent.interpretations.add_source('second supplied correction', provider='authored structural fixture')
    group = agent.interpretations.create_group(source.id)
    frame = Frame('change', {'destination': Ref('fresh:third')})
    frames = (frame, row[4][1])
    candidate = agent.interpretations.propose(group.id, SentenceAlternative(None, tuple(Act('request', Request(f), f) for f in frames)))
    agent.interpretations.select(group.id, candidate.id, reason='second supplied reading')
    second_group = api().propose_task_revision(agent, first.id, group.id, candidate.id, model=model, basis=('same explicit task',))
    assert not isinstance(second_group, Unknown), second_group
    second = api().adopt_task_revision(agent, first.id, second_group, decision(agent, second_group), reason='second correction')
    assert not isinstance(second, Unknown), second
    assert second.revision == 3 and second.goal.conditions[0].args['destination'] == Ref('fresh:third')
    assert all(dependency in second.dependencies for dependency in first.dependencies)
    assert api().validate_retained_revision_group(agent, first_group) is True
    assert api().validate_retained_revision_group(agent, second_group) is True
