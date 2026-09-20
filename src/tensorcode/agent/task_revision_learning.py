"""Authenticated whole-correction supervision and explicit contextual model admission.

Task association and revised goals are supplied teaching. Successful model fits
retain historical supervision; live proposal contexts still require current tasks
and the complete selected correction, including all ordered request frames.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from uuid import uuid4

from ..goals import GoalSpec
from ..language import Frame
from ..language.semantics import Request
from ..learning.experience import _same
from ..learning.goal_revision import GoalRevisionExample, GoalRevisionModel, fit_goal_revisions
from ..outcomes import Unknown
from .scene_grounding import (_CandidateSnapshot, _capture, _candidate, _validate_snapshot,
                              _final_comparisons, grounding_dependencies)
from .task_dependencies import InterpretationDependency, capture_dependency, validate_dependencies, _expected_basis
from .understand import SentenceAlternative


@dataclass(frozen=True)
class TaskRevisionContext:
    task_id: str
    task_revision: int
    previous: GoalSpec
    goal_group_id: str | None
    prior_dependencies: tuple[InterpretationDependency, ...]
    correction: _CandidateSnapshot
    correction_dependency: InterpretationDependency
    supporting_dependencies: tuple[InterpretationDependency, ...]
    corrections: tuple[Frame, ...]
    basis: tuple[str, ...]


@dataclass(frozen=True)
class RetainedTaskRevisionExample:
    example: GoalRevisionExample
    context: TaskRevisionContext
    evidence_source_id: str


@dataclass(frozen=True)
class TaskRevisionModelHandle:
    group_id: str
    candidate_id: str
    model_id: str
    evidence_source_id: str
    dependency: InterpretationDependency | None = None


@dataclass(frozen=True)
class _ModelVersion:
    snapshot: dict
    evidence_source_id: str


@dataclass(frozen=True)
class _ModelGroup:
    source: object
    provenance: tuple[str, ...]
    versions: tuple


def _registry(agent, name):
    if not hasattr(agent, name):
        setattr(agent, name, {})
    return getattr(agent, name)


def _models(agent):
    return _registry(agent, '_task_revision_models')


def _examples(agent):
    return _registry(agent, '_task_revision_examples')


def _basis(basis):
    if type(basis) is not tuple or not basis or any(type(x) is not str or not x.strip() for x in basis):
        raise ValueError('explicit nonempty basis tuple required')


def _task_state(agent, task_id):
    revision = agent.tasks.current_revision(task_id)
    task = agent.tasks.get(task_id)
    if type(task.goal) is not GoalSpec:
        raise ValueError('context requires a structured GoalSpec')
    group_id = task.goal_interpretation_id
    if group_id is not None:
        if group_id in getattr(agent, '_goal_proposal_groups', {}):
            from .goal_learning import _task_state as original_task_state
            original_task_state(agent, task_id)
        elif group_id in getattr(agent, '_task_revision_groups', {}):
            from .task_revision import validate_retained_revision_group
            result = validate_retained_revision_group(agent, group_id)
            if result is not True:
                raise ValueError(result.detail)
            group = agent.interpretations.get(group_id)
            dependencies = tuple(d for d in task.dependencies if d.group_id == group_id)
            if (group.selected is None or len(dependencies) != 1
                    or dependencies[0].candidate_id != group.selected_id
                    or not _same(group.selected.payload.goal, task.goal)):
                raise ValueError('task does not match retained revision goal')
        else:
            raise ValueError('unrecognized retained goal group')
    valid = validate_dependencies(agent.interpretations, task.dependencies)
    if valid is not True:
        raise ValueError(valid.reason)
    if task.revision != revision or agent.tasks.current_revision(task_id) != revision:
        raise ValueError('task changed during context capture')
    return task


def _frames(snapshot):
    alternative = _candidate(snapshot).payload
    if (not alternative.acts or alternative.skipped or alternative.metadata.get('unresolved')
            or alternative.metadata.get('semantic_unresolved')
            or alternative.metadata.get('syntax_complete') is False
            or alternative.metadata.get('semantic_projection_complete') is False):
        raise ValueError('correction reading contains skipped or unresolved content')
    frames = []
    for act in alternative.acts:
        if (act.kind != 'request' or type(act.meaning) is not Request
                or type(act.frame) is not Frame or not _same(act.meaning.frame, act.frame)):
            raise ValueError('every correction act must be an exact matching Request and Frame')
        frames.append(act.frame)
    metadata = alternative.metadata
    if ('tokens' in metadata or 'token_anchors' in metadata
            or 'learned' in alternative.provenance):
        tokens, anchors = metadata.get('tokens'), metadata.get('token_anchors')
        if type(tokens) not in (tuple, list) or not tokens or type(anchors) not in (tuple, list) or len(tokens) != len(anchors):
            raise ValueError('learned correction requires complete token anchors')
        previous = 0
        for index, (token, anchor) in enumerate(zip(tokens, anchors), 1):
            if type(anchor) is not dict or anchor.get('index') != index or anchor.get('token') != token:
                raise ValueError('correction token anchor identity mismatch')
            span = anchor.get('char_span')
            if (type(span) not in (tuple, list) or len(span) != 2 or any(type(x) is not int for x in span)
                    or not previous <= span[0] < span[1] <= len(snapshot.source.text)
                    or snapshot.source.text[span[0]:span[1]] != token
                    or snapshot.source.text[previous:span[0]].strip()):
                raise ValueError('correction token anchors do not cover retained source')
            previous = span[1]
        if snapshot.source.text[previous:].strip():
            raise ValueError('correction token anchors omit source content')
    return deepcopy(tuple(frames))


def _dependencies(context):
    return (*context.prior_dependencies, context.correction_dependency, *context.supporting_dependencies)


def _validate_context(agent, context):
    if type(context) is not TaskRevisionContext:
        raise ValueError('expected retained task revision context')
    _basis(context.basis)
    task = _task_state(agent, context.task_id)
    if (task.revision != context.task_revision or not _same(task.goal, context.previous)
            or task.goal_interpretation_id != context.goal_group_id
            or task.dependencies != context.prior_dependencies):
        raise ValueError('task revision context changed')
    snapshot = context.correction
    _validate_snapshot(agent.interpretations, snapshot, SentenceAlternative)
    if (snapshot.group.selected_id != snapshot.candidate_id
            or snapshot.comparison != _expected_basis(context.correction_dependency)
            or not _same(_frames(snapshot), context.corrections)):
        raise ValueError('correction context changed')
    supporting = grounding_dependencies(agent, snapshot.group_id, snapshot.candidate_id)
    if isinstance(supporting, Unknown) or supporting != context.supporting_dependencies:
        raise ValueError('correction support changed')


def _final_contexts(agent, contexts):
    dependencies = tuple(d for c in contexts for d in _dependencies(c))
    valid = validate_dependencies(agent.interpretations, dependencies)
    if valid is not True:
        raise ValueError(valid.reason)
    _final_comparisons(agent.interpretations, tuple(c.correction for c in contexts))
    for dependency in dependencies:
        if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
            raise ValueError('context dependency changed during validation')
    if any(agent.tasks.current_revision(c.task_id) != c.task_revision for c in contexts):
        raise ValueError('task changed during context validation')


def capture_task_revision_context(agent, task_id, correction_group_id, correction_candidate_id, *, basis):
    """Capture explicit task association and a complete selected correction reading."""
    try:
        _basis(basis)
        task = _task_state(agent, task_id)
        snapshot = _capture(agent.interpretations, correction_group_id, correction_candidate_id, SentenceAlternative)
        if snapshot.group.selected_id != correction_candidate_id:
            raise ValueError('correction must be explicitly selected')
        dependency = capture_dependency(agent.interpretations, correction_group_id, basis=basis,
                                        evidence_ids=(snapshot.source.id,))
        supporting = grounding_dependencies(agent, correction_group_id, correction_candidate_id)
        if isinstance(supporting, Unknown):
            raise ValueError(supporting.detail)
        context = deepcopy(TaskRevisionContext(task.id, task.revision, task.goal, task.goal_interpretation_id,
            task.dependencies, snapshot, dependency, supporting, _frames(snapshot), basis))
        _validate_context(agent, context)
        _final_contexts(agent, (context,))
        return context
    except Exception as error:
        return Unknown('task_revision_context_unavailable', f'{type(error).__name__}: {error}')


def validate_task_revision_context(agent, context):
    try:
        _validate_context(agent, context)
        _final_contexts(agent, (context,))
        return True
    except Exception as error:
        return Unknown('task_revision_context_changed', f'{type(error).__name__}: {error}')


def retain_task_revision_example(agent, task_id, correction_group_id, correction_candidate_id, revised, *, basis):
    """Retain teacher-supplied contextual revision labels without changing the task."""
    try:
        if type(revised) is not GoalSpec:
            raise ValueError('teaching requires a revised GoalSpec')
        context = capture_task_revision_context(agent, task_id, correction_group_id, correction_candidate_id, basis=basis)
        if isinstance(context, Unknown):
            raise ValueError(context.detail)
        example = GoalRevisionExample('task-revision-example:' + uuid4().hex, deepcopy(context.previous),
                                      deepcopy(context.corrections), deepcopy(revised), basis)
        evidence = agent.interpretations.add_source('Historical explicit contextual task revision supervision',
            modality='teaching', provider='retained-task-revision-supervision', payload=deepcopy(example),
            metadata={'context': deepcopy(context)})
        record = RetainedTaskRevisionExample(example, context, evidence.id)
        cached, result = deepcopy((record, evidence)), deepcopy(record)
        _validate_context(agent, context)
        _final_contexts(agent, (context,))
        _examples(agent)[evidence.id] = cached
        return result
    except Exception as error:
        return Unknown('task_revision_teaching_unavailable', f'{type(error).__name__}: {error}')


def _validate_examples(agent, records):
    for record in records:
        if type(record) is not RetainedTaskRevisionExample:
            raise ValueError('expected retained contextual teaching')
        cached = _examples(agent).get(record.evidence_source_id)
        if cached is None or not _same(record, cached[0]):
            raise ValueError('unrecognized teaching snapshot')
        if not _same(agent.interpretations.get_source(record.evidence_source_id), cached[1]):
            raise ValueError('teaching evidence changed')
        _validate_context(agent, record.context)
    _final_contexts(agent, tuple(r.context for r in records))


def _model_snapshot(model):
    from .goal_learning import _model_snapshot as correspondence_snapshot
    if type(model) is not GoalRevisionModel or set(vars(model)) != {'_id', '_training', '_validation', '_correspondence'}:
        raise ValueError('unrecognized contextual model implementation')
    return deepcopy({'_id': model._id, '_training': model._training, '_validation': model._validation,
                     '_correspondence': correspondence_snapshot(model._correspondence)})


def _model_state(agent, group_id):
    retained = _models(agent).get(group_id)
    if retained is None:
        raise ValueError('unrecognized goal model group')
    workspace = agent.interpretations
    basis = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    if (group.source_id != retained.source.id or group.provenance != retained.provenance
            or not _same(workspace.get_source(group.source_id), retained.source)
            or tuple(c.id for c in group.candidates) != tuple(row[0] for row in retained.versions)):
        raise ValueError('goal model group content changed')
    for candidate, (candidate_id, version, source, model) in zip(group.candidates, retained.versions):
        if (candidate.group_id != group_id or candidate.provenance != ('retained-task-revision-model',)
                or not _same(candidate.payload, version)
                or not _same(_model_snapshot(model), version.snapshot)
                or not _same(workspace.get_source(version.evidence_source_id), source)):
            raise ValueError('goal model version content changed')
    if workspace.comparison_basis(group_id) != basis:
        raise ValueError('goal model comparison changed')
    return retained, group, basis


def fit_task_revision_model(agent, training_records, validation_records, *, group_id=None, max_pairs=256):
    """Fit and retain an unadmitted model; explicit refit withdraws old admission."""
    published = None
    try:
        training, validation = tuple(training_records), tuple(validation_records)
        records = (*training, *validation)
        if not training or not validation or any(type(r) is not RetainedTaskRevisionExample for r in records):
            raise ValueError('nonempty retained training and heldout records required')
        for identities in (tuple(r.evidence_source_id for r in records),
                           tuple(r.context.task_id for r in records),
                           tuple(r.context.correction.source.id for r in records)):
            if len(set(identities)) != len(identities):
                raise ValueError('training and heldout tasks and correction sources must be unique and disjoint')
        previous = _model_state(agent, group_id) if group_id is not None else None
        _validate_examples(agent, records)
        model = fit_goal_revisions(tuple(r.example for r in training), tuple(r.example for r in validation), max_pairs=max_pairs)
        snapshot = _model_snapshot(model)
        if not model.complete:
            raise ValueError('contextual goal revision fit incomplete')
        evidence = agent.interpretations.add_source(
            'Fitted contextual goal revision from retained explicit supervision', modality='model-fit',
            provider='task-revision-learning', payload=deepcopy(snapshot),
            metadata={'training_evidence_ids': tuple(r.evidence_source_id for r in training),
                      'heldout_evidence_ids': tuple(r.evidence_source_id for r in validation),
                      'max_pairs': max_pairs, 'historical_supervision': True})
        version = _ModelVersion(snapshot, evidence.id)
        cached_version, cached_source, cached_model = deepcopy(version), deepcopy(evidence), deepcopy(model)
        _validate_examples(agent, records)
        workspace = agent.interpretations
        if previous is None:
            group = workspace.create_group(evidence.id, provenance=('contextual goal revision model versions',))
            retained = _ModelGroup(deepcopy(evidence), group.provenance, ())
        else:
            retained, group, basis = _model_state(agent, group_id)
            if basis != previous[2]:
                raise ValueError('model admission changed during refit')
        candidate = workspace.propose(group.id, version, provenance=('retained-task-revision-model',))
        # Publication itself can invoke payload-copy callbacks. Register its
        # authentic historical content before post-publication checks; a failed
        # fit becomes a rejected version, never an unregistered row that poisons
        # the entire prior model group. Earlier versions can be readmitted.
        _models(agent)[group.id] = replace(retained, versions=(*retained.versions,
                                          (candidate.id, cached_version, cached_source, cached_model)))
        published = (group.id, candidate.id)
        workspace.unset(group.id, reason='new fitted model requires explicit admission')
        _validate_examples(agent, records)
        return TaskRevisionModelHandle(group.id, candidate.id, model.id, evidence.id)
    except Exception as error:
        if published is not None:
            agent.interpretations.reject(*published, reason='teaching changed during model publication')
        return Unknown('task_revision_model_fit_unavailable', f'{type(error).__name__}: {error}')


def admit_task_revision_model(agent, handle, *, reason):
    """Explicitly admit an authentic version; this does not decide any request."""
    try:
        if not isinstance(handle, TaskRevisionModelHandle) or not isinstance(reason, str) or not reason.strip():
            raise ValueError('model admission requires a retained handle and explicit reason')
        _, group, basis = _model_state(agent, handle.group_id)
        candidate = next(c for c in group.candidates if c.id == handle.candidate_id)
        if (candidate.rejected or candidate.payload.snapshot['_id'] != handle.model_id
                or candidate.payload.evidence_source_id != handle.evidence_source_id):
            raise ValueError('model handle does not identify the retained version')
        if agent.interpretations.comparison_basis(group.id) != basis:
            raise ValueError('model comparison changed before admission')
        selected_basis = (basis[0], group.revision + 1, candidate.id, basis[3], False, basis[5], basis[6])
        agent.interpretations.select(group.id, candidate.id, reason=reason,
                                     evidence_ids=(handle.evidence_source_id,))
        if agent.interpretations.comparison_basis(group.id) != selected_basis:
            raise ValueError('model admission changed during selection')
        dependency = capture_dependency(agent.interpretations, group.id,
            basis=('explicit contextual task revision model admission', reason), evidence_ids=(handle.evidence_source_id,))
        if agent.interpretations.comparison_basis(group.id) != selected_basis:
            raise ValueError('model admission changed during dependency capture')
        _model_state(agent, group.id)
        if validate_dependencies(agent.interpretations, (dependency,)) is not True:
            raise ValueError('model admission changed')
        if agent.interpretations.comparison_basis(group.id) != selected_basis:
            raise ValueError('model admission changed after dependency capture')
        result = replace(handle, dependency=dependency)
        cached = deepcopy(result)
        _model_state(agent, group.id)
        if (validate_dependencies(agent.interpretations, (dependency,)) is not True
                or agent.interpretations.comparison_basis(group.id) != selected_basis):
            raise ValueError('model admission changed after copying')
        _registry(agent, '_task_revision_admissions')[(group.id, candidate.id)] = cached
        return result
    except Exception as error:
        return Unknown('task_revision_model_admission_unavailable', f'{type(error).__name__}: {error}')


def get_task_revision_model(agent, handle):
    """Read only an admitted authentic model version with a live dependency."""
    try:
        if not isinstance(handle, TaskRevisionModelHandle) or handle.dependency is None:
            raise ValueError('model version has not been explicitly admitted')
        cached = _registry(agent, '_task_revision_admissions').get((handle.group_id, handle.candidate_id))
        if cached is None or not _same(handle, cached):
            raise ValueError('unrecognized contextual model admission')
        _, group, comparison = _model_state(agent, handle.group_id)
        candidate = group.selected
        if (candidate is None or candidate.id != handle.candidate_id
                or handle.dependency.group_id != group.id or handle.dependency.candidate_id != candidate.id
                or candidate.payload.snapshot['_id'] != handle.model_id
                or candidate.payload.evidence_source_id != handle.evidence_source_id):
            raise ValueError('model handle does not identify current admission')
        retained = _models(agent)[group.id]
        model = deepcopy(next(row[3] for row in retained.versions if row[0] == candidate.id))
        _model_state(agent, group.id)
        valid = validate_dependencies(agent.interpretations, (handle.dependency,))
        if valid is not True:
            raise ValueError(valid.reason)
        if agent.interpretations.comparison_basis(group.id) != comparison:
            raise ValueError('model comparison changed during retrieval')
        return model
    except Exception as error:
        return Unknown('task_revision_model_unavailable', f'{type(error).__name__}: {error}')
