"""Ledger semantics for explicit post-activation goals, with authored transport fixtures."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.document_actions import DocumentActionProposal, DocumentActionContext
from tensorcode.agent.document_tasks import create_document_task, pursue_document_task, DocumentGoal
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.plugin import Call
from tensorcode.outcomes import Unknown, Receipt
from tensorcode.records import Ref
import tensorcode.agent.document_tasks as runner


@pytest.fixture
def case(monkeypatch):
    agent = Agent([])
    provider = SimpleNamespace(name='fixture-provider')
    model = SimpleNamespace(id='fixture-model', provider='plugin:fixture-provider')
    source = agent.interpretations.add_source('authored task dependency')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, 'authored selected interpretation')
    agent.interpretations.select(group.id, candidate.id, reason='explicit fixture selection')
    dependency = capture_dependency(agent.interpretations, group.id, basis=('authored fixture dependency',))
    proposal = DocumentActionProposal('proposal', 'proposal-source', Ref('node:target'))
    action = Call(provider.name, 'activate_node', (('target', 'opaque-fixture-token'),))
    context = DocumentActionContext(action, (dependency,), proposal.target)
    state = SimpleNamespace(calls=0, predicted=True, observed=True, prediction_hook=lambda: None,
                            before_hook=lambda: None, after_hook=lambda: None, feedback_hook=lambda: None,
                            guard_result=True, suspended=False, validations=0, final_hook=lambda: None, final_authority=True)
    monkeypatch.setattr(runner, 'document_action_context', lambda *args: context)
    def prediction(*args):
        state.prediction_hook()
        return SimpleNamespace(id='prediction', evidence_source_id='prediction-source', action=action,
                               prediction=state.predicted if isinstance(state.predicted, Unknown)
                               else SimpleNamespace(outcome=state.predicted))
    monkeypatch.setattr(runner, 'predict_document_transition', prediction)
    def validate(*args):
        state.validations += 1
        return state.guard_result
    monkeypatch.setattr(runner, 'validate_document_prediction', validate)
    monkeypatch.setattr(runner, 'validate_document_prediction_authority', lambda *args: state.final_authority)
    def execute(*args, before_dispatch, task_revision, final_dispatch_check):
        state.before_hook()
        allowed = before_dispatch(('before-source',))
        state.final_hook()
        if allowed is True:
            allowed = final_dispatch_check()
        if allowed is True and agent.tasks.current_revision(task_revision[0]) == task_revision[1]:
            state.calls += 1
            receipt = Receipt(action, 'applied')
            state.after_hook()
        else:
            receipt = Receipt(action, 'rejected', error='guard declined')
        return SimpleNamespace(id='execution', evidence_source_id='execution-source', receipt=receipt,
            events=({'type': 'receipt', 'attempt_id': 'attempt'},))
    monkeypatch.setattr(runner, 'execute_document_action', execute)
    def assess(*args):
        state.feedback_hook()
        if isinstance(state.observed, Unknown):
            return state.observed
        state.suspended = state.observed != state.predicted
        return SimpleNamespace(outcome=state.observed, receipt=Receipt(action, 'applied'),
            attempt_id='attempt', source_ids=('before-source', 'after-source'), suspension=state.suspended,
            evidence_source_id='feedback-source')
    monkeypatch.setattr(runner, 'assess_document_transition', assess)
    return agent, provider, model, proposal, state, group


def create(case, desired=True):
    agent, provider, model, proposal, _, _ = case
    task = create_document_task(agent, provider, model, proposal, desired)
    assert not isinstance(task, Unknown), task
    return task


def pursue(case, task):
    agent, provider, model, _, _, _ = case
    return pursue_document_task(agent, provider, model, task_id=task.id)


def test_actual_outcome_alone_completes_and_completed_revision_cannot_replay(case):
    task = create(case)
    result = pursue(case, task)
    agent, _, _, _, state, _ = case
    assert result.status == 'done' and result.verified is True and result.receipt.status == 'applied'
    assert state.calls == 1 and state.validations == 1
    assert 'feedback-source' in result.plan.observation_source_ids
    assert agent.tasks.get(task.id).status == 'done'
    assert pursue(case, task).reason == 'document_task_revision_consumed'
    assert state.calls == 1 and len(agent.tasks.get(task.id).attempts) == 1


@pytest.mark.parametrize('predicted', [Unknown('unsupported'), False])
def test_unknown_or_mismatched_prediction_leaves_browser_untouched(case, predicted):
    task = create(case)
    case[4].predicted = predicted
    result = pursue(case, task)
    assert result.status == 'unknown' and case[4].calls == 0
    assert case[0].tasks.get(task.id).status == 'blocked'
    assert pursue(case, task).reason == 'document_task_revision_consumed'


def test_counterexample_is_unverified_and_suspends_rule(case):
    task = create(case)
    case[4].observed = False
    result = pursue(case, task)
    assert result.status == 'unverified' and result.verified is False
    assert case[4].calls == 1 and case[4].suspended
    assert result.steps[0].receipt.status == 'applied'


def test_missing_observed_outcome_cannot_complete_applied_action(case):
    task = create(case)
    case[4].observed = Unknown('observation_unavailable')
    result = pursue(case, task)
    assert result.status == 'unverified' and isinstance(result.verified, Unknown)
    assert result.receipt.status == 'applied'


@pytest.mark.parametrize('phase', ['prediction_hook', 'before_hook'])
def test_withdrawal_before_dispatch_blocks_operation(case, phase):
    task = create(case)
    agent, _, _, _, state, group = case
    setattr(state, phase, lambda: agent.interpretations.unset(group.id, reason='withdrawn interpretation'))
    result = pursue(case, task)
    assert result.status == 'unknown' and state.calls == 0


@pytest.mark.parametrize('phase', ['prediction_hook', 'before_hook', 'after_hook', 'feedback_hook'])
def test_revision_during_attempt_retains_old_history_without_completing_new_revision(case, phase):
    task = create(case)
    agent, _, _, _, state, _ = case
    setattr(state, phase, lambda: agent.tasks.revise(task.id, task.goal, reason='explicit revised request'))
    result = pursue(case, task)
    assert result.status == 'unknown' and result.reason == 'task_revision_changed'
    latest = agent.tasks.get(task.id)
    assert latest.revision == 2 and latest.status == 'ready'
    assert len(latest.attempts) == 1 and latest.attempts[0].revision == 1
    if phase in ('after_hook', 'feedback_hook'):
        assert state.calls == 1 and latest.attempts[0].receipt.status == 'applied'
    else:
        assert state.calls == 0


def test_reentrant_pursuit_does_not_create_second_attempt(case):
    task = create(case)
    state = case[4]
    def recurse():
        assert pursue(case, task).reason == 'document_task_in_progress'
    state.prediction_hook = recurse
    assert pursue(case, task).status == 'done'
    assert state.calls == 1 and len(case[0].tasks.get(task.id).attempts) == 1


def test_explicit_revision_can_retry_unacted_unknown(case):
    task = create(case)
    state = case[4]
    state.predicted = Unknown('unsupported')
    assert pursue(case, task).status == 'unknown'
    state.predicted = True
    revised = case[0].tasks.revise(task.id, task.goal, reason='explicit retry after fresh model evidence')
    assert pursue(case, revised).status == 'done'
    assert [attempt.revision for attempt in case[0].tasks.get(task.id).attempts] == [1, 2]


def test_dispatch_prediction_guard_declines_without_action(case):
    task = create(case)
    case[4].guard_result = Unknown('model_rule_suspended')
    result = pursue(case, task)
    assert result.status == 'unknown' and result.receipt.status == 'rejected'
    assert case[4].calls == 0


def test_late_model_withdrawal_after_observation_guard_declines_dispatch(case):
    task = create(case)
    state = case[4]
    def withdraw():
        state.final_authority = Unknown('transition_model_changed')
    state.final_hook = withdraw
    result = pursue(case, task)
    assert state.validations == 1 and state.calls == 0
    assert result.status == 'unknown' and result.receipt.status == 'rejected'


def test_new_task_requires_explicit_post_activation_outcome(case):
    agent, provider, model, proposal, _, _ = case
    with pytest.raises(ValueError, match='must be explicit'):
        pursue_document_task(agent, provider, model, proposal=proposal)
