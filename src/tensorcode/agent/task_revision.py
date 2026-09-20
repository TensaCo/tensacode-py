"""Explicit contextual correction proposals and guarded same-task adoption.

Task association, teaching labels, model admission, and proposal selection are
supplied decisions. Proposal and adoption never parse, perceive, plan, or execute.
Earlier dependencies remain live; receipts stay attached to their old revisions.
"""
from copy import deepcopy
from dataclasses import dataclass

from ..goals import GoalSpec
from ..learning.experience import _same
from ..learning.goal_correspondence import CorrespondenceCandidates, CorrespondenceProposal
from ..outcomes import Unknown
from .core import InterpretationDecision
from .scene_grounding import _validate_snapshot, _final_comparisons, grounding_dependencies
from .task_dependencies import capture_dependency, validate_dependencies, _expected_basis
from .task_revision_learning import (TaskRevisionContext, TaskRevisionModelHandle,
    capture_task_revision_context, validate_task_revision_context, get_task_revision_model,
    _dependencies)
from .understand import SentenceAlternative


@dataclass(frozen=True)
class _RetainedRevisionGroup:
    source: object
    provenance: tuple[str, ...]
    candidates: tuple


def _registry(agent):
    if not hasattr(agent, '_task_revision_groups'):
        agent._task_revision_groups = {}
    return agent._task_revision_groups


def _required(result):
    if isinstance(result, Unknown):
        raise ValueError(f'{result.reason}: {result.detail}')
    return result


def _supports(context, model):
    return tuple(dict.fromkeys((*_dependencies(context), model.dependency)))


def _final(agent, context, model, *, group_id=None, comparison=None, extra=(), current=False):
    dependencies = (*_supports(context, model), *extra)
    _required(validate_dependencies(agent.interpretations, dependencies))
    _final_comparisons(agent.interpretations, (context.correction,))
    for dependency in dependencies:
        if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
            raise ValueError('correction dependency changed during validation')
    if group_id is not None and agent.interpretations.comparison_basis(group_id) != comparison:
        raise ValueError('task revision proposal comparison changed')
    if current and agent.tasks.current_revision(context.task_id) != context.task_revision:
        raise ValueError('task revision changed during validation')


def _group_state(agent, group_id):
    """Authenticate content and live support against a historical task revision."""
    retained = _registry(agent).get(group_id)
    if retained is None:
        raise ValueError('unrecognized retained task revision group')
    workspace = agent.interpretations
    comparison = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    source = workspace.get_source(group.source_id)
    if (group.source_id != retained.source.id or group.provenance != retained.provenance
            or not _same(source, retained.source)
            or tuple(c.id for c in group.candidates) != tuple(c.id for c in retained.candidates)):
        raise ValueError('retained task revision group content changed')
    for candidate, original in zip(group.candidates, retained.candidates):
        if (candidate.group_id != group_id or candidate.provenance != original.provenance
                or not _same(candidate.payload, original.payload)):
            raise ValueError('retained task revision candidate content changed')
    context, model = source.metadata['context'], source.metadata['model']
    if (type(context) is not TaskRevisionContext or type(model) is not TaskRevisionModelHandle
            or type(source.payload) is not CorrespondenceCandidates
            or source.provider != 'contextual-task-revision'):
        raise ValueError('invalid retained task revision source')
    expected = (*source.payload.proposals, *source.payload.unresolved)
    if len(expected) != len(group.candidates) or any(not _same(c.payload, p) for c, p in zip(group.candidates, expected)):
        raise ValueError('retained task revision batch changed')
    revision = agent.tasks.current_revision(context.task_id)
    task = agent.tasks.get(context.task_id)
    historical = next((r for r in task.history if r.revision == context.task_revision), None)
    if (historical is None or not _same(historical.goal, context.previous)
            or historical.dependencies != context.prior_dependencies
            or historical.goal_interpretation_id != context.goal_group_id):
        raise ValueError('historical task revision context changed')
    _validate_snapshot(workspace, context.correction, SentenceAlternative)
    inherited = grounding_dependencies(agent, context.correction.group_id, context.correction.candidate_id)
    if isinstance(inherited, Unknown) or inherited != context.supporting_dependencies:
        raise ValueError('correction support changed')
    _required(get_task_revision_model(agent, model))
    _final(agent, context, model, group_id=group_id, comparison=comparison)
    if agent.tasks.current_revision(context.task_id) != revision:
        raise ValueError('task changed while validating retained revision group')
    return group, source, context, model, comparison


def validate_retained_revision_group(agent, group_id):
    """Validate authentic historical context and still-live correction/model support.

    Successful adoption advances the task; the old contextual input is thereafter
    authenticated against its retained ledger history rather than current goal.
    """
    try:
        _group_state(agent, group_id)
        return True
    except Exception as error:
        return Unknown('task_revision_group_changed', f'{type(error).__name__}: {error}')


def propose_task_revision(agent, task_id, correction_group_id, correction_candidate_id, *, model, basis):
    """Retain every learned alternative without selecting or changing the task."""
    published = None
    try:
        context = _required(capture_task_revision_context(agent, task_id, correction_group_id,
                                                          correction_candidate_id, basis=basis))
        learner = _required(get_task_revision_model(agent, model))
        batch = learner.propose(deepcopy(context.previous), deepcopy(context.corrections))
        if type(batch) is not CorrespondenceCandidates:
            raise ValueError('contextual model returned an unsupported proposal batch')
        workspace = agent.interpretations
        source = workspace.add_source('Contextual correction proposals for an explicitly associated task',
            modality='goal-projection', provider='contextual-task-revision', payload=deepcopy(batch),
            metadata={'context': deepcopy(context), 'model': deepcopy(model)})
        cached_source = deepcopy(source)
        _required(validate_task_revision_context(agent, context))
        _required(get_task_revision_model(agent, model))
        _final(agent, context, model, current=True)
        group = workspace.create_group(source.id, provenance=('contextual task revision alternatives',))
        published = group.id
        candidates = []
        for payload in (*batch.proposals, *batch.unresolved):
            candidate = workspace.propose(group.id, payload, provenance=(
                'contextual-task-revision-proposal' if type(payload) is CorrespondenceProposal
                else 'unresolved-contextual-task-revision',))
            candidates.append(deepcopy(candidate))
        retained = _RetainedRevisionGroup(cached_source, group.provenance, tuple(candidates))
        _registry(agent)[group.id] = retained
        _, _, _, _, comparison = _group_state(agent, group.id)
        if comparison != (source.id, 0, None, tuple(c.id for c in candidates), None, False, 0):
            raise ValueError('proposal comparison changed during publication')
        _required(validate_task_revision_context(agent, context))
        _required(get_task_revision_model(agent, model))
        _final(agent, context, model, group_id=group.id, comparison=comparison, current=True)
        return group.id
    except Exception as error:
        if published is not None:
            workspace = agent.interpretations
            for candidate in workspace.get(published).candidates:
                workspace.reject(published, candidate.id, reason='task correction evidence changed during publication')
        return Unknown('task_revision_proposal_unavailable', f'{type(error).__name__}: {error}')


def adopt_task_revision(agent, task_id, group_id, decision: InterpretationDecision, *, reason):
    """Adopt an explicitly compared proposal through the ledger's guarded revision."""
    try:
        if type(reason) is not str or not reason.strip():
            raise ValueError('task revision adoption requires an explicit reason')
        if (type(decision) is not InterpretationDecision or type(decision.reason) is not str
                or not decision.reason.strip() or type(decision.evidence_ids) is not tuple
                or any(type(x) is not str or not x.strip() for x in decision.evidence_ids)):
            raise ValueError('explicit interpretation decision and evidence IDs required')
        group, source, context, model, comparison = _group_state(agent, group_id)
        if context.task_id != task_id:
            raise ValueError('proposal belongs to another task')
        if (type(decision.compared_revision) is not int or decision.compared_revision != group.revision
                or decision.compared_candidate_ids != tuple(c.id for c in group.candidates)):
            raise ValueError('decision does not identify the exact compared proposal set')
        _required(validate_task_revision_context(agent, context))
        batch = source.payload
        if not batch.complete or batch.unresolved:
            raise ValueError('contextual revision search remains incomplete or unresolved')
        candidate = next((c for c in group.candidates if c.id == decision.candidate_id), None)
        if (candidate is None or candidate.rejected or type(candidate.payload) is not CorrespondenceProposal
                or type(candidate.payload.goal) is not GoalSpec):
            raise ValueError('decision must select an available contextual goal proposal')
        workspace = agent.interpretations
        for identity in decision.evidence_ids:
            workspace.get_source(identity)
        assessment = workspace.add_source('Explicit contextual task revision selection', modality='assessment',
            provider='task-revision-selection', payload=deepcopy(decision),
            metadata={'task_id': task_id, 'proposal_group_id': group_id, 'reason': reason})
        goal = deepcopy(candidate.payload.goal)
        _group_state(agent, group_id)
        _required(validate_task_revision_context(agent, context))
        _required(get_task_revision_model(agent, model))
        _final(agent, context, model, group_id=group_id, comparison=comparison, current=True)
        selected_basis = (comparison[0], group.revision + 1, candidate.id, comparison[3],
                          False, comparison[5], comparison[6])
        workspace.select(group_id, candidate.id, reason=decision.reason,
                         evidence_ids=(*decision.evidence_ids, assessment.id))
        if workspace.comparison_basis(group_id) != selected_basis:
            raise ValueError('proposal comparison changed during selection')
        dependency = capture_dependency(workspace, group_id,
            basis=('explicit contextual task revision selection', decision.reason, reason),
            evidence_ids=tuple(dict.fromkeys((source.id, assessment.id, *decision.evidence_ids))))
        dependencies = tuple(dict.fromkeys((*_supports(context, model), dependency)))

        def before_commit():
            _group_state(agent, group_id)
            _required(validate_task_revision_context(agent, context))
            _required(get_task_revision_model(agent, model))
            _final(agent, context, model, group_id=group_id, comparison=selected_basis,
                   extra=(dependency,), current=True)
            return True

        before_commit()
        return agent.tasks.revise(task_id, goal, reason=reason, dependencies=dependencies,
            goal_interpretation_id=group_id, expected_revision=context.task_revision,
            before_commit=before_commit)
    except Exception as error:
        return Unknown('task_revision_adoption_unavailable', f'{type(error).__name__}: {error}')
