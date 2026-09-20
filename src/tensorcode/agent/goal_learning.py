"""Learn goal correspondences from authenticated, explicitly supervised tasks.

A current task goal is supplied teaching, not proof of intended meaning or world
truth. Successful fits retain historical teaching snapshots: later task revisions
require explicit refitting and do not retroactively rewrite that evidence. A model
version needs separate admission; admission is not selection of a future intent.
"""
from copy import deepcopy
from dataclasses import dataclass, replace

from ..goals import GoalSpec
from ..language.semantics import Frame
from ..learning.experience import _same
from ..learning.goal_correspondence import GoalExample, GoalCorrespondenceModel, fit_correspondences
from ..outcomes import Unknown
from .task_dependencies import InterpretationDependency, capture_dependency, validate_dependencies


@dataclass(frozen=True)
class RetainedGoalExample:
    example: GoalExample
    task_id: str
    task_revision: int
    goal_group_id: str
    goal_source_id: str
    goal_candidate_id: str
    dependencies: tuple[InterpretationDependency, ...]
    evidence_source_id: str


@dataclass(frozen=True)
class GoalModelHandle:
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
    versions: tuple[tuple[str, _ModelVersion, object, GoalCorrespondenceModel], ...]


def _model_snapshot(model):
    """Authenticate exact pure-model state, including unresolved search evidence."""
    if type(model) is not GoalCorrespondenceModel or set(vars(model)) != {
            '_id', '_training', '_validation', '_templates', '_complete', '_unresolved'}:
        raise ValueError('unrecognized goal correspondence model implementation')
    return deepcopy(vars(model))


def _models(agent):
    if not hasattr(agent, '_goal_learning_models'):
        agent._goal_learning_models = {}
    return agent._goal_learning_models


def _examples(agent):
    if not hasattr(agent, '_goal_learning_examples'):
        agent._goal_learning_examples = {}
    return agent._goal_learning_examples


def _task_state(agent, task_id):
    """Read an authentic linked task without selecting or changing interpretation."""
    revision = agent.tasks.current_revision(task_id)
    task = agent.tasks.get(task_id)
    if not isinstance(task.goal, GoalSpec):
        raise ValueError('teaching requires a current structured goal')
    retained = getattr(agent, '_goal_proposal_groups', {}).get(task.goal_interpretation_id)
    if retained is None:
        raise ValueError('unrecognized goal interpretation group')
    workspace = agent.interpretations
    group = workspace.get(task.goal_interpretation_id)
    source = workspace.get_source(group.source_id)
    if (not _same(source, retained.source) or group.provenance != retained.provenance
            or group.source_id != retained.source.id
            or tuple(c.id for c in group.candidates) != retained.candidate_ids):
        raise ValueError('retained goal group content changed')
    if not isinstance(source.payload, dict) or not isinstance(source.payload.get('frame'), Frame):
        raise ValueError('goal source has no retained frame')
    batch = source.payload.get('batch')
    expected = (*batch.proposals, *batch.unresolved)
    if len(group.candidates) != len(expected) or any(
            c.group_id != group.id or not _same(c.payload, value)
            for c, value in zip(group.candidates, expected)):
        raise ValueError('retained goal proposal content changed')
    selected = group.selected
    if selected is None or selected.rejected or not hasattr(selected.payload, 'goal'):
        raise ValueError('task goal interpretation is not selected')
    goal_dependencies = [dep for dep in task.dependencies if dep.group_id == group.id]
    if len(goal_dependencies) != 1 or goal_dependencies[0].candidate_id != selected.id:
        raise ValueError('task lacks exact selected goal dependency')
    parent = source.metadata.get('parent_dependency')
    required = (parent, *source.metadata.get('supporting_dependencies', ()))
    if parent is None or any(dependency not in task.dependencies for dependency in required):
        raise ValueError('teaching requires its original sentence and supporting dependencies')
    valid = validate_dependencies(workspace, task.dependencies)
    if valid is not True:
        raise ValueError(valid.reason)
    if agent.tasks.current_revision(task_id) != revision or task.revision != revision:
        raise ValueError('task changed while reading teaching')
    return task, group, source


def _validate_example(agent, record):
    retained = _examples(agent).get(record.evidence_source_id)
    if retained is None or not _same(record, retained[0]):
        raise ValueError('unrecognized teaching snapshot')
    if not _same(agent.interpretations.get_source(record.evidence_source_id), retained[1]):
        raise ValueError('teaching evidence changed')
    task, group, source = _task_state(agent, record.task_id)
    if (task.revision != record.task_revision or group.id != record.goal_group_id
            or source.id != record.goal_source_id or group.selected_id != record.goal_candidate_id
            or task.dependencies != record.dependencies or not _same(task.goal, record.example.goal)
            or not _same(source.payload['frame'], record.example.frame)):
        raise ValueError('task supervision changed')
    if agent.tasks.current_revision(record.task_id) != record.task_revision:
        raise ValueError('task supervision changed during comparison')


def _validate_examples(agent, records):
    for record in records:
        _validate_example(agent, record)
    # A later example's copy callback can revise an earlier example. Validate
    # all interpretation commitments together, then finish with callback-free
    # task revision reads instead of trusting per-example checks in isolation.
    dependencies = tuple(dep for record in records for dep in record.dependencies)
    valid = validate_dependencies(agent.interpretations, dependencies)
    if valid is not True:
        raise ValueError(valid.reason)
    if any(agent.tasks.current_revision(record.task_id) != record.task_revision for record in records):
        raise ValueError('task supervision changed during batch validation')


def extract_goal_example(agent, task_id):
    """Retain authenticated supervision from a selected, linked task revision."""
    try:
        task, group, source = _task_state(agent, task_id)
        example = GoalExample(f'{task.id}:revision:{task.revision}', deepcopy(source.payload['frame']),
                              deepcopy(task.goal), ('explicit retained task goal supervision',))
        evidence = agent.interpretations.add_source(
            'Historical explicit task-to-goal supervision', modality='teaching',
            provider='retained-task-goal-supervision', payload=deepcopy(example),
            metadata={'task_id': task.id, 'task_revision': task.revision,
                      'goal_group_id': group.id, 'goal_source_id': source.id,
                      'dependencies': deepcopy(task.dependencies)})
        record = RetainedGoalExample(example, task.id, task.revision, group.id, source.id,
                                     group.selected_id, deepcopy(task.dependencies), evidence.id)
        cached = deepcopy((record, evidence))
        result = deepcopy(record)
        current, now_group, now_source = _task_state(agent, task_id)
        if (current.revision != task.revision or current.dependencies != task.dependencies
                or now_group.selected_id != group.selected_id or not _same(current.goal, task.goal)
                or not _same(now_source, source)):
            raise ValueError('task changed while retaining teaching')
        if agent.tasks.current_revision(task_id) != task.revision:
            raise ValueError('task changed while retaining teaching')
        _examples(agent)[evidence.id] = cached
        return result
    except Exception as error:
        return Unknown('goal_teaching_unavailable', f'{type(error).__name__}: {error}')


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
        if (candidate.group_id != group_id or candidate.provenance != ('retained-goal-correspondence-model',)
                or not _same(candidate.payload, version)
                or not _same(_model_snapshot(model), version.snapshot)
                or not _same(workspace.get_source(version.evidence_source_id), source)):
            raise ValueError('goal model version content changed')
    if workspace.comparison_basis(group_id) != basis:
        raise ValueError('goal model comparison changed')
    return retained, group, basis


def fit_goal_model(agent, training_task_ids, heldout_task_ids, *, group_id=None, max_pairs=256):
    """Fit and retain an unadmitted model; explicit refit withdraws old admission."""
    published = None
    try:
        train_ids, heldout_ids = tuple(training_task_ids), tuple(heldout_task_ids)
        if (not train_ids or not heldout_ids or len(set(train_ids)) != len(train_ids)
                or len(set(heldout_ids)) != len(heldout_ids) or set(train_ids) & set(heldout_ids)):
            raise ValueError('training and heldout task IDs must be nonempty, unique and disjoint')
        previous = _model_state(agent, group_id) if group_id is not None else None
        records = tuple(extract_goal_example(agent, identity) for identity in (*train_ids, *heldout_ids))
        if any(isinstance(record, Unknown) for record in records):
            raise ValueError(next(record.detail for record in records if isinstance(record, Unknown)))
        _validate_examples(agent, records)
        model = fit_correspondences(tuple(r.example for r in records[:len(train_ids)]),
                                    tuple(r.example for r in records[len(train_ids):]), max_pairs=max_pairs)
        snapshot = _model_snapshot(model)
        if not model.complete:
            raise ValueError('goal correspondence fit incomplete')
        evidence = agent.interpretations.add_source(
            'Fitted goal correspondence from retained explicit supervision', modality='model-fit',
            provider='goal-correspondence-learning', payload=deepcopy(snapshot),
            metadata={'training_evidence_ids': tuple(r.evidence_source_id for r in records[:len(train_ids)]),
                      'heldout_evidence_ids': tuple(r.evidence_source_id for r in records[len(train_ids):]),
                      'max_pairs': max_pairs, 'historical_supervision': True})
        version = _ModelVersion(snapshot, evidence.id)
        cached_version, cached_source, cached_model = deepcopy(version), deepcopy(evidence), deepcopy(model)
        _validate_examples(agent, records)
        workspace = agent.interpretations
        if previous is None:
            group = workspace.create_group(evidence.id, provenance=('goal correspondence model versions',))
            retained = _ModelGroup(deepcopy(evidence), group.provenance, ())
        else:
            retained, group, basis = _model_state(agent, group_id)
            if basis != previous[2]:
                raise ValueError('model admission changed during refit')
        candidate = workspace.propose(group.id, version, provenance=('retained-goal-correspondence-model',))
        # Publication itself can invoke payload-copy callbacks. Register its
        # authentic historical content before post-publication checks; a failed
        # fit becomes a rejected version, never an unregistered row that poisons
        # the entire prior model group. Earlier versions can be readmitted.
        _models(agent)[group.id] = replace(retained, versions=(*retained.versions,
                                          (candidate.id, cached_version, cached_source, cached_model)))
        published = (group.id, candidate.id)
        workspace.unset(group.id, reason='new fitted model requires explicit admission')
        _validate_examples(agent, records)
        return GoalModelHandle(group.id, candidate.id, model.id, evidence.id)
    except Exception as error:
        if published is not None:
            agent.interpretations.reject(*published, reason='teaching changed during model publication')
        return Unknown('goal_model_fit_unavailable', f'{type(error).__name__}: {error}')


def admit_goal_model(agent, handle, *, reason):
    """Explicitly admit an authentic version; this does not decide any request."""
    try:
        if not isinstance(handle, GoalModelHandle) or not isinstance(reason, str) or not reason.strip():
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
            basis=('explicit goal model admission', reason), evidence_ids=(handle.evidence_source_id,))
        if agent.interpretations.comparison_basis(group.id) != selected_basis:
            raise ValueError('model admission changed during dependency capture')
        _model_state(agent, group.id)
        if validate_dependencies(agent.interpretations, (dependency,)) is not True:
            raise ValueError('model admission changed')
        if agent.interpretations.comparison_basis(group.id) != selected_basis:
            raise ValueError('model admission changed after dependency capture')
        return replace(handle, dependency=dependency)
    except Exception as error:
        return Unknown('goal_model_admission_unavailable', f'{type(error).__name__}: {error}')


def get_goal_model(agent, handle):
    """Read only an admitted authentic model version with a live dependency."""
    try:
        if not isinstance(handle, GoalModelHandle) or handle.dependency is None:
            raise ValueError('model version has not been explicitly admitted')
        _, group, _ = _model_state(agent, handle.group_id)
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
        return model
    except Exception as error:
        return Unknown('goal_model_unavailable', f'{type(error).__name__}: {error}')
