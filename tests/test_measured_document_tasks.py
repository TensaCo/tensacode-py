"""Explicit teaching/model fixtures isolate declarative intent materialization."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

from tensorcode.agent.document_transition_evidence import browser_transition_projection
from tensorcode.agent.goal_interpretation import retain_goal_proposals, LearnedGoalProposal, LearnedGoalCandidates
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.measured_document_tasks import pursue_measured_document_task
from tensorcode.goals import MeasuredActionGoal
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from test_document_action_evidence import prepared, PATH
import tensorcode.agent.goal_interpretation as interpretation
import tensorcode.agent.measured_document_tasks as materializer
import tensorcode.agent.document_tasks as runner


@pytest.fixture
def measured(monkeypatch, request):
    agent, provider, document, gid, cid = prepared()
    reading = agent.interpretations.get(gid).selected.payload
    target = reading.acts[0].frame.roles['object'].ref
    goal = MeasuredActionGoal(target, 'activate_node', browser_transition_projection().name, True)
    goal = replace(goal, **getattr(request, 'param', {}))
    parent = capture_dependency(agent.interpretations, gid, basis=('authored selected reading',))
    model_source = agent.interpretations.add_source('explicit fixture model admission')
    model_group = agent.interpretations.create_group(model_source.id)
    model_candidate = agent.interpretations.propose(model_group.id, 'authored model authority fixture')
    agent.interpretations.select(model_group.id, model_candidate.id, reason='explicit fixture admission')
    model_dependency = capture_dependency(agent.interpretations, model_group.id, basis=('authored model fixture',))
    agent.goal_model = object()
    monkeypatch.setattr(interpretation, '_learned_candidates', lambda *_:
        (LearnedGoalCandidates((LearnedGoalProposal(goal, ('fixture-template',), ('train',), ('heldout',)),), (), True), (model_dependency,)))
    group_id = retain_goal_proposals(agent, reading.acts[0].frame, 'authored teaching fixture', parent_dependency=parent)
    group = agent.interpretations.get(group_id)
    agent.interpretations.select(group.id, group.candidates[0].id, reason='explicit fixture goal selection')
    dependency = capture_dependency(agent.interpretations, group.id, basis=('authored selected goal',))
    task = agent.tasks.create('authored intent', goal, dependencies=(parent, model_dependency, dependency), goal_interpretation_id=group.id)
    model = SimpleNamespace(id='authored-model', provider='plugin:' + provider.name, projection=browser_transition_projection())
    monkeypatch.setattr(materializer, '_model_contract', lambda a, p, m: None if m is model and p is provider else (_ for _ in ()).throw(ValueError('foreign model')))
    def predict(a, p, m, token):
        from tensorcode.agent.plugin import Call
        return SimpleNamespace(id='prediction', evidence_source_id='prediction-source',
            action=Call(p.name, 'activate_node', (('target', token),)), prediction=SimpleNamespace(outcome=True))
    monkeypatch.setattr(runner, 'predict_document_transition', predict)
    monkeypatch.setattr(runner, 'validate_document_prediction', lambda *_: True)
    monkeypatch.setattr(runner, 'validate_document_prediction_authority', lambda *_: True)
    monkeypatch.setattr(runner, 'assess_document_transition', lambda *_: SimpleNamespace(
        outcome=True, source_ids=('before', 'after'), evidence_source_id='feedback'))
    args = dict(task_id=task.id, language_group_id=gid, candidate_id=cid, path=PATH,
                document_group_id=document.group_id, document_candidate_id=document.candidate_id)
    return agent, provider, model, task, args, group.id


def test_same_task_preserves_selected_declaration_and_retains_realization(measured):
    agent, provider, model, task, args, goal_group = measured
    result = pursue_measured_document_task(agent, provider, model, **args)
    assert not isinstance(result, Unknown), result
    assert result.status == 'done', result
    current = agent.tasks.get(task.id)
    assert result.task_id == task.id and current.revision == task.revision
    assert current.goal == task.goal == result.goal
    assert current.goal_interpretation_id == goal_group and len(current.attempts) == 1
    assert len(provider.calls) == 1
    sources = [s for s in agent.interpretations.sources() if s.modality == 'measured-document-realization']
    assert len(sources) == 1
    assert sources[0].metadata['task_id'] == task.id
    assert sources[0].payload['goal'] == task.goal
    assert sources[0].id in result.plan.observation_source_ids
    assert isinstance(pursue_measured_document_task(agent, provider, model, **args), Unknown)
    assert len(provider.calls) == 1


@pytest.mark.parametrize('field,value', [('measurement', 'checkbox'), ('operation', 'click'), ('target', Ref('node:other'))])
def test_task_rewrite_cannot_override_selected_goal(measured, field, value):
    agent, provider, model, task, args, _ = measured
    agent.tasks.revise(task.id, replace(task.goal, **{field: value}), reason='unselected supplied rewrite')
    assert isinstance(pursue_measured_document_task(agent, provider, model, **args), Unknown)
    assert not provider.calls


def test_forged_goal_group_and_wrong_reading_are_not_runtime_authority(measured):
    agent, provider, model, task, args, _ = measured
    forged = agent.tasks.create('forged goal', task.goal, dependencies=task.dependencies,
                                goal_interpretation_id='invented-group')
    assert isinstance(pursue_measured_document_task(agent, provider, model, **{**args, 'task_id': forged.id}), Unknown)
    assert isinstance(pursue_measured_document_task(agent, provider, model, **{**args, 'candidate_id': 'other'}), Unknown)
    assert not provider.calls


def test_goal_withdrawal_during_before_observation_blocks_action(measured):
    agent, provider, model, task, args, goal_group = measured
    provider.before_observation = lambda: agent.interpretations.unset(goal_group, reason='withdrawn selected goal')
    result = pursue_measured_document_task(agent, provider, model, **args)
    assert result.status == 'unknown' and not provider.calls
    assert agent.tasks.get(task.id).goal == task.goal


def test_postaction_goal_withdrawal_retains_receipt_without_completing(measured):
    agent, provider, model, task, args, goal_group = measured
    provider.before_execution = lambda: agent.interpretations.unset(goal_group, reason='withdrawn during execution')
    result = pursue_measured_document_task(agent, provider, model, **args)
    assert result.status == 'unknown' and result.receipt.status == 'applied'
    assert len(provider.calls) == 1 and agent.tasks.get(task.id).goal == task.goal


@pytest.mark.parametrize('measured', [dict(measurement='checkbox'), dict(operation='click'),
                                     dict(target=Ref('node:other'))], indirect=True)
def test_selected_but_unsupported_contract_or_mismatched_grounding_never_acts(measured):
    agent, provider, model, _, args, _ = measured
    assert isinstance(pursue_measured_document_task(agent, provider, model, **args), Unknown)
    assert not provider.calls


def test_declarative_handoff_attempt_is_materialized_without_goal_revision(measured):
    from tensorcode.agent import InterpretationDecision
    from tensorcode.agent.understand import Sentence
    agent, provider, model, task, args, _ = measured
    reading = agent.interpretations.get(args['language_group_id']).selected.payload
    agent.goal_selector = lambda group: InterpretationDecision(group.candidates[0].id,
        'explicit selection of authored fixture goal', compared_revision=group.revision,
        compared_candidate_ids=tuple(candidate.id for candidate in group.candidates))
    parent = capture_dependency(agent.interpretations, args['language_group_id'], basis=('explicit request reading',))
    request = agent.request(Sentence('authored measured request', (), None, reading.acts),
        reading.acts[0], [], interpretation_dependency=parent)
    assert request.status == 'suspended' and request.reason == 'measured_goal_requires_materialization', (request.reason, request.verified)
    task = agent.tasks.get(request.task_id)
    args = {**args, 'task_id': task.id}
    result = pursue_measured_document_task(agent, provider, model, **args)
    assert result.status == 'done' and len(provider.calls) == 1
    current = agent.tasks.get(task.id)
    assert current.revision == task.revision and current.goal == task.goal
    assert len(current.attempts) == 2
