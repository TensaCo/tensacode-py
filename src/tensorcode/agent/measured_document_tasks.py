"""Materialize selected measured intent without replacing its declarative goal.

Only exact registered browser activation/measurement contracts are supported.
Runtime binding supplies a provider and fitted transition model, never new intent.
"""
from copy import deepcopy
from uuid import uuid4

from ..goals import MeasuredActionGoal
from ..learning.experience import _same
from ..outcomes import Unknown
from .document_actions import prepare_document_action, document_action_context
from .document_tasks import DocumentGoal, _pursue_realized_document_task
from .document_transition_evidence import browser_transition_projection, _model_contract
from .goal_learning import _task_state
from .task_dependencies import validate_dependencies, _expected_basis


def _registry(agent):
    if not hasattr(agent, '_measured_document_realizations'):
        agent._measured_document_realizations = {}
    return agent._measured_document_realizations


def _has_execution_attempt(task):
    """A declarative handoff contains no executable attempt to replay."""
    return any(attempt.revision == task.revision and not (
        attempt.status == 'suspended' and attempt.reason == 'measured_goal_requires_materialization'
        and attempt.receipt is None and not attempt.steps and attempt.plan is None)
        for attempt in task.attempts)


def _selected_goal(agent, task_id):
    task, group, source = _task_state(agent, task_id)
    if type(task.goal) is not MeasuredActionGoal or not _same(group.selected.payload.goal, task.goal):
        raise ValueError('task must retain exactly its selected measured goal')
    return task, group, source


def _resolve_realization(agent, provider, model, task, realization_id):
    """Authenticate the registered plan even after its action token is consumed."""
    try:
        entry = _registry(agent).get(realization_id)
        if entry is None or entry[0] is not provider or entry[1] is not model:
            raise ValueError('unrecognized measured document realization')
        _, _, original, source, goal_source, realized, dependencies = entry
        current, _, current_goal_source = _selected_goal(agent, task.id)
        if (current.id != original.id or current.revision != original.revision
                or not _same(current.goal, original.goal) or current.dependencies != original.dependencies
                or not _same(task.goal, original.goal) or task.revision != original.revision
                or not _same(current_goal_source, goal_source)
                or not _same(agent.interpretations.get_source(source.id), source)):
            raise ValueError('measured goal or realization evidence changed')
        _model_contract(agent, provider, model)
        valid = validate_dependencies(agent.interpretations, dependencies)
        if valid is not True:
            return valid
        result = deepcopy((realized, dependencies, (source.id,)))
        if agent.tasks.current_revision(task.id) != original.revision:
            raise ValueError('task revision changed')
        for dependency in dependencies:
            if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
                raise ValueError('measured goal interpretation changed')
        return result
    except Exception as error:
        return Unknown('measured_document_realization_unavailable', f'{type(error).__name__}: {error}')


def pursue_measured_document_task(agent, provider, model, *, task_id, language_group_id,
                                 candidate_id, path, document_group_id, document_candidate_id):
    """Bind an adopted goal to selected evidence, then pursue that same revision.

    The caller supplies runtime connections and retained context identities. The
    selected goal supplies target, operation, measurement, and desired outcome.
    """
    try:
        task, _, goal_source = _selected_goal(agent, task_id)
        if task.status == 'done' or _has_execution_attempt(task):
            raise ValueError('task revision already attempted')
        goal = task.goal
        projection = browser_transition_projection()
        _model_contract(agent, provider, model)
        if goal.operation != 'activate_node' or goal.measurement != projection.name:
            raise ValueError('unsupported exact operation or measurement contract')
        if model.projection.name != goal.measurement:
            raise ValueError('transition model measurement does not match selected goal')
        parent = goal_source.metadata.get('parent_dependency')
        if parent is None or parent.group_id != language_group_id or parent.candidate_id != candidate_id:
            raise ValueError('runtime reading differs from selected goal parent')
        proposal = prepare_document_action(agent, provider, language_group_id, candidate_id, path,
                                           document_group_id, document_candidate_id)
        if isinstance(proposal, Unknown):
            return proposal
        context = document_action_context(agent, provider, proposal)
        if isinstance(context, Unknown):
            return context
        if not _same(context.target, goal.target) or context.action.capability != goal.operation:
            raise ValueError('selected goal target differs from grounded runtime target')
        dependencies = tuple(dict.fromkeys((*task.dependencies, *context.dependencies)))
        realized = DocumentGoal(deepcopy(goal.desired_outcome), proposal, model.id, model.provider)
        source = agent.interpretations.add_source('Runtime realization of selected measured intent',
            modality='measured-document-realization', provider=provider.name,
            payload={'goal': deepcopy(goal), 'realized_goal': deepcopy(realized), 'action': deepcopy(context.action)},
            metadata={'task_id': task.id, 'task_revision': task.revision,
                      'goal_source_id': goal_source.id, 'goal_interpretation_id': task.goal_interpretation_id,
                      'dependencies': dependencies, 'model_id': model.id,
                      'proposal_source_id': proposal.evidence_source_id})
        realization_id = 'measured-document:' + uuid4().hex
        _registry(agent)[realization_id] = (provider, model, deepcopy(task), deepcopy(source),
                                           deepcopy(goal_source), deepcopy(realized), deepcopy(dependencies))
        valid = _resolve_realization(agent, provider, model, task, realization_id)
        if isinstance(valid, Unknown):
            return valid
        return _pursue_realized_document_task(agent, provider, model, task.id, realization_id)
    except Exception as error:
        return Unknown('measured_document_task_unavailable', f'{type(error).__name__}: {error}')
