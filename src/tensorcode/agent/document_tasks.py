"""One revision-bound activation pursuing an explicit measured postcondition.

The caller supplies the operation, bound reading, model and desired post-activation
outcome. This is not state-goal search, lexical goal inference, or a skip-if-already
satisfied planner. Predicted effects authorize attempts; only actual paired
observations can establish completion.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import Lock

from ..learning.experience import _same
from ..outcomes import Unknown, Receipt
from .document_actions import document_action_context, execute_document_action
from .document_transition_evidence import (
    predict_document_transition, validate_document_prediction, assess_document_transition,
    validate_document_prediction_authority,
)
from .task_dependencies import validate_dependencies, _expected_basis
from .tasks import StepAttempt
from .understand import Act


_UNSET = object()


@dataclass(frozen=True)
class DocumentGoal:
    desired_outcome: object
    proposal: object
    model_id: str
    provider: str


@dataclass(frozen=True)
class DocumentTaskTrace:
    proposal_id: str
    prediction_id: str | None
    execution_id: str | None
    observation_source_ids: tuple[str, ...]
    operation: str = 'explicit activate_node with desired measured post-activation outcome'


def create_document_task(agent, provider, model, proposal, desired_outcome, *, source='document'):
    """Capture selected reading/document dependencies for a caller-supplied goal."""
    try:
        if isinstance(desired_outcome, Unknown):
            raise ValueError('desired outcome cannot be Unknown')
        hash(desired_outcome)
        if model.provider != 'plugin:' + provider.name:
            raise ValueError('document model belongs to another provider')
        context = document_action_context(agent, provider, proposal)
        if isinstance(context, Unknown):
            return context
        goal = DocumentGoal(deepcopy(desired_outcome), deepcopy(proposal), model.id, model.provider)
        task = agent.tasks.create(source, goal, dependencies=context.dependencies)
        return task
    except Exception as error:
        return Unknown('document_task_unavailable', f'{type(error).__name__}: {error}')


def pursue_document_task(agent, provider, model, *, proposal=None, desired_outcome=_UNSET,
                         task_id=None, source='document'):
    """Attempt one explicit task revision once, without automatic replay or choice."""
    from .core import Outcome
    if (proposal is None) == (task_id is None):
        raise ValueError('supply either proposal and desired_outcome, or task_id')
    if proposal is not None and desired_outcome is _UNSET:
        raise ValueError('desired post-activation outcome must be explicit')
    if task_id is not None and desired_outcome is not _UNSET:
        raise ValueError('change desired outcome through an explicit task revision')
    if task_id is None:
        task = create_document_task(agent, provider, model, proposal, desired_outcome, source=source)
        if isinstance(task, Unknown):
            return Outcome(Act('request', desired_outcome, None), 'unknown', verified=task, reason=task.reason)
    else:
        task = agent.tasks.get(task_id)
    locks = getattr(agent, '_document_task_locks', None)
    if locks is None:
        locks = agent._document_task_locks = {}
    lock = locks.setdefault(task.id, Lock())
    if not lock.acquire(blocking=False):
        return Outcome(Act('request', task.goal, None), 'unknown', task_id=task.id,
                       verified=Unknown('document_task_in_progress'), reason='document_task_in_progress')
    try:
        task = agent.tasks.get(task.id)
        revision, goal = task.revision, deepcopy(task.goal)
        act = Act('request', goal, None)
        if task.status == 'done' or any(attempt.revision == revision for attempt in task.attempts):
            return Outcome(act, 'unknown', task_id=task.id, verified=Unknown('document_task_revision_consumed'),
                           reason='document_task_revision_consumed')
        prediction = execution = None
        receipt = None
        source_ids = []
        steps = []

        def unchanged():
            current = agent.tasks.get(task.id)
            if current.revision != revision or not _same(current.goal, goal) or current.dependencies != task.dependencies:
                return Unknown('task_revision_changed')
            valid = validate_dependencies(agent.interpretations, task.dependencies)
            if valid is not True:
                return valid
            if agent.tasks.current_revision(task.id) != revision:
                return Unknown('task_revision_changed')
            for dependency in task.dependencies:
                if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
                    return Unknown('task_interpretation_changed')
            return True

        def finish(status, verified, reason):
            trace = DocumentTaskTrace(getattr(goal.proposal, 'id', ''), getattr(prediction, 'id', None),
                                      getattr(execution, 'id', None), tuple(source_ids))
            outcome = Outcome(act, status, goal=deepcopy(goal), plan=trace, receipt=deepcopy(receipt),
                verified=deepcopy(verified), reason=reason, task_id=task.id, steps=tuple(deepcopy(steps)))
            validity = unchanged()
            if validity is not True:
                outcome = replace(outcome, status='unknown', verified=validity, reason=validity.reason)
            agent.tasks.record(task.id, outcome, revision=revision)
            return outcome

        if type(goal) is not DocumentGoal or goal.model_id != model.id or goal.provider != model.provider or model.provider != 'plugin:' + provider.name:
            # Malformed supplied goals carry no executable authority.
            if type(goal) is not DocumentGoal:
                outcome = Outcome(act, 'unknown', task_id=task.id, verified=Unknown('invalid_document_goal'), reason='invalid_document_goal')
                agent.tasks.record(task.id, outcome, revision=revision)
                return outcome
            return finish('unknown', Unknown('document_model_mismatch'), 'document_model_mismatch')
        validity = unchanged()
        if validity is not True:
            return finish('unknown', validity, validity.reason)
        context = document_action_context(agent, provider, goal.proposal)
        if isinstance(context, Unknown):
            return finish('unknown', context, context.reason)
        if context.dependencies != task.dependencies:
            return finish('unknown', Unknown('document_task_dependencies_changed'), 'document_task_dependencies_changed')
        prediction = predict_document_transition(agent, provider, model, context.action.arg('target'))
        validity = unchanged()
        if validity is not True:
            return finish('unknown', validity, validity.reason)
        if isinstance(prediction, Unknown):
            return finish('unknown', prediction, prediction.reason)
        source_ids.append(prediction.evidence_source_id)
        if isinstance(prediction.prediction, Unknown):
            return finish('unknown', prediction.prediction, prediction.prediction.reason)
        if not _same(prediction.action, context.action):
            return finish('unknown', Unknown('document_prediction_action_changed'), 'document_prediction_action_changed')
        if not _same(prediction.prediction.outcome, goal.desired_outcome):
            return finish('unknown', False, 'predicted_document_outcome_differs')

        def guard(before_source_ids):
            validity = unchanged()
            if validity is not True:
                return validity
            result = validate_document_prediction(agent, provider, model, prediction, before_source_ids)
            if result is not True:
                return result
            return unchanged()

        try:
            execution = execute_document_action(agent, provider, goal.proposal, before_dispatch=guard,
                                                task_revision=(task.id, revision),
                final_dispatch_check=lambda: validate_document_prediction_authority(agent, provider, model, prediction))
        except Exception as error:
            receipt = Receipt(context.action, 'indeterminate', error=f'{type(error).__name__}: {error}')
            steps.append(StepAttempt(goal.proposal.id, context.action, receipt, Unknown('document_execution_error')))
            return finish('unverified', Unknown('document_execution_error', receipt.error), 'document_execution_error')
        if isinstance(execution, Unknown):
            return finish('unknown', execution, execution.reason)
        receipt = execution.receipt
        source_ids.append(execution.evidence_source_id)
        steps.append(StepAttempt(execution.id, context.action, receipt, Unknown('document_outcome_unassessed')))
        validity = unchanged()
        if validity is not True:
            return finish('unknown', validity, validity.reason)
        if receipt.status != 'applied':
            return finish('unknown', Unknown('document_action_not_applied', receipt.error or receipt.status), 'document_action_not_applied')
        attempts = {event['attempt_id'] for event in execution.events if event.get('type') == 'receipt'}
        if len(attempts) != 1:
            return finish('unverified', Unknown('document_attempt_ambiguous'), 'document_attempt_ambiguous')
        feedback = assess_document_transition(agent, provider, model, prediction, attempts.pop())
        if isinstance(feedback, Unknown):
            return finish('unverified', feedback, feedback.reason)
        source_ids.extend((*feedback.source_ids, feedback.evidence_source_id))
        verified = _same(feedback.outcome, goal.desired_outcome)
        steps[-1] = replace(steps[-1], verified=verified)
        return finish('done' if verified else 'unverified', verified,
                      'document_outcome_observed' if verified else 'document_outcome_differs')
    finally:
        lock.release()
