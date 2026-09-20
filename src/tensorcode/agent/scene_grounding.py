"""Retained supervision and explicitly admitted scene-to-mention grounding.

Teachers supply alignments, never a query or an automatic choice of scene. Fits
retain historical snapshots. Publication adds every distinct inferred reference as
an alternative and does not select language, assert beliefs, or execute actions.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from uuid import uuid4

from ..language import Entity, Frame
from ..learning.experience import _same
from ..learning.scene_grounding import GroundingExample, SceneGroundingModel, fit_scene_grounding
from ..outcomes import Unknown
from ..records import Ref
from .grounding import MentionBinding, _with_frame, propose_grounding
from .scene import SceneProposal
from .evidence_graph import GraphProposal, graph_root

GRAPH_PROPOSALS = (SceneProposal, GraphProposal)
from .task_dependencies import InterpretationDependency, capture_dependency, validate_dependencies
from .understand import SentenceAlternative


@dataclass(frozen=True)
class _CandidateSnapshot:
    group_id: str
    source: object
    group: object
    candidate_id: str
    comparison: tuple
    frontier: object


@dataclass(frozen=True)
class RetainedGroundingExample:
    example: GroundingExample
    language: _CandidateSnapshot
    scene: _CandidateSnapshot
    path: tuple
    evidence_source_id: str


@dataclass(frozen=True)
class SceneGroundingModelHandle:
    group_id: str
    candidate_id: str
    model_id: str
    evidence_source_id: str
    dependency: InterpretationDependency | None = None


@dataclass(frozen=True)
class UnresolvedGrounding:
    """A retained possible referent with no executable binding.

    ``reference=None`` records that the observed graph may omit referents. A
    known reference records unanswered query evidence, not a negative match.
    """
    reference: Ref | None
    query_ids: tuple[str, ...]
    reason: str
    evidence_source_id: str


@dataclass(frozen=True)
class SceneGroundingReport:
    candidate_ids: tuple[str, ...]
    complete: bool
    source_id: str
    unresolved: tuple[str, ...]
    unresolved_candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ModelVersion:
    snapshot: dict
    evidence_source_id: str


@dataclass(frozen=True)
class _ModelGroup:
    source: object
    provenance: tuple[str, ...]
    versions: tuple


@dataclass(frozen=True)
class _GroundedCandidate:
    source: object
    group_provenance: tuple[str, ...]
    candidate: object
    parent: object
    model: SceneGroundingModelHandle
    scene: _CandidateSnapshot
    dependencies: tuple[InterpretationDependency, ...]
    report_source: object


def _registry(agent, name):
    if not hasattr(agent, name):
        setattr(agent, name, {})
    return getattr(agent, name)


def _capture(workspace, group_id, candidate_id, expected):
    comparison = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    source = workspace.get_source(group.source_id)
    frontier = workspace.continuation_status(group_id)
    candidate = next(c for c in group.candidates if c.id == candidate_id)
    if candidate.group_id != group.id or candidate.rejected or type(candidate.payload) not in (expected if type(expected) is tuple else (expected,)):
        raise ValueError('candidate is rejected, mismatched, or has an unsupported payload')
    if type(candidate.payload) is SceneProposal:
        candidate.payload.validate()
        if source.modality != 'image' or source.metadata.get('image_ref') != candidate.payload.graph.image.id:
            raise ValueError('scene graph does not identify its retained image source')
    elif type(candidate.payload) is GraphProposal:
        candidate.payload.validate()
        if candidate.payload.source_id != source.id or source.metadata.get('root_ref') != graph_root(candidate.payload.graph).id:
            raise ValueError('evidence graph does not identify its exact retained source and root')
    if workspace.comparison_basis(group_id) != comparison:
        raise ValueError('interpretation comparison changed while reading evidence')
    return _CandidateSnapshot(group_id, source, group, candidate_id, comparison, frontier)


def _candidate(snapshot):
    return next(c for c in snapshot.group.candidates if c.id == snapshot.candidate_id)


def _validate_snapshot(workspace, snapshot, expected):
    current = _capture(workspace, snapshot.group_id, snapshot.candidate_id, expected)
    if not _same(current, snapshot):
        raise ValueError('retained interpretation content or comparison changed')


def _final_comparisons(workspace, snapshots):
    for snapshot in snapshots:
        if workspace.comparison_basis(snapshot.group_id) != snapshot.comparison:
            raise ValueError('interpretation comparison changed during validation')


def _description(payload, path):
    if (type(path) is not tuple or len(path) < 4 or path[0] != 'acts'
            or type(path[1]) is not int or path[2] != 'frame'
            or any(type(part) not in (str, int) for part in path)):
        raise ValueError('description requires an explicit Entity occurrence path')
    if not 0 <= path[1] < len(payload.acts):
        raise ValueError('description act index is out of range')
    act = payload.acts[path[1]]
    _with_frame(act, act.frame)  # Check agreement between meaning and frame.
    value = act.frame
    for part in path[3:]:
        if isinstance(value, Mapping):
            value = value[part]
        elif type(value) is tuple and type(part) is int and 0 <= part < len(value):
            value = value[part]
        elif isinstance(value, Frame) and part in ('roles', 'features'):
            value = getattr(value, part)
        elif isinstance(value, Entity) and part in ('features', 'candidates'):
            value = getattr(value, part)
        else:
            raise ValueError('description path does not traverse a semantic structure')
    if type(value) is not Entity:
        raise ValueError('description path must identify an Entity occurrence')
    return deepcopy(value)


def _validate_example(agent, record):
    if type(record) is not RetainedGroundingExample:
        raise ValueError('teaching requires retained grounding records')
    cached = _registry(agent, '_scene_grounding_examples').get(record.evidence_source_id)
    if cached is None or not _same(record, cached[0]):
        raise ValueError('unrecognized or modified grounding teaching record')
    workspace = agent.interpretations
    if not _same(workspace.get_source(record.evidence_source_id), cached[1]):
        raise ValueError('grounding teaching source changed')
    _validate_snapshot(workspace, record.language, SentenceAlternative)
    _validate_snapshot(workspace, record.scene, GRAPH_PROPOSALS)
    _final_comparisons(workspace, (record.language, record.scene))


def _validate_examples(agent, records):
    for record in records:
        _validate_example(agent, record)
    _final_comparisons(agent.interpretations, tuple(s for r in records for s in (r.language, r.scene)))


def retain_grounding_example(agent, language_group_id, candidate_id, path,
                             scene_group_id, scene_candidate_id, positive_refs,
                             negative_refs=(), *, basis):
    """Retain explicit teaching labels and exact language/scene comparisons.

    Teaching candidates need not be selected. Labels and basis are teacher input;
    this operation does not infer reference identity or claim visual truth.
    """
    try:
        if type(basis) is not tuple or not basis or any(type(x) is not str or not x.strip() for x in basis):
            raise ValueError('teaching requires an explicit nonempty basis tuple')
        workspace = agent.interpretations
        language = _capture(workspace, language_group_id, candidate_id, SentenceAlternative)
        scene = _capture(workspace, scene_group_id, scene_candidate_id, GRAPH_PROPOSALS)
        description = _description(_candidate(language).payload, path)
        graph = _candidate(scene).payload.graph
        for labels in (positive_refs, negative_refs):
            if type(labels) is not tuple or any(type(ref) is not Ref or ref not in (graph_root(graph), *graph.nodes) for ref in labels):
                raise ValueError('teaching labels require explicit declared scene node Refs')
            if len(set(labels)) != len(labels):
                raise ValueError('duplicate teaching alignment')
        if (not positive_refs and not negative_refs) or set(positive_refs) & set(negative_refs):
            raise ValueError('at least one explicit positive or negative teaching label is required without conflicts')
        example = GroundingExample('grounding-example:' + uuid4().hex, description,
                                   deepcopy(graph), positive_refs, negative_refs, basis)
        evidence = workspace.add_source('Explicit historical language-to-scene supervision',
            modality='teaching', provider='retained-scene-grounding-supervision',
            payload=deepcopy(example), metadata={'language': language, 'scene': scene, 'path': path})
        record = RetainedGroundingExample(example, language, scene, path, evidence.id)
        cached, result = deepcopy((record, evidence)), deepcopy(record)
        _validate_snapshot(workspace, language, SentenceAlternative)
        _validate_snapshot(workspace, scene, GRAPH_PROPOSALS)
        _final_comparisons(workspace, (language, scene))
        _registry(agent, '_scene_grounding_examples')[evidence.id] = cached
        return result
    except Exception as error:
        return Unknown('scene_grounding_teaching_unavailable', f'{type(error).__name__}: {error}')


def _model_snapshot(model):
    if type(model) is not SceneGroundingModel or set(vars(model)) != {
            '_id', '_training', '_validation', '_queries', '_complete', '_unresolved', '_max_matches'}:
        raise ValueError('unrecognized scene grounding model implementation')
    return deepcopy(vars(model))


def _model_state(agent, group_id):
    retained = _registry(agent, '_scene_grounding_models').get(group_id)
    if retained is None:
        raise ValueError('unrecognized scene grounding model group')
    workspace = agent.interpretations
    comparison = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    if (group.source_id != retained.source.id or group.provenance != retained.provenance
            or not _same(workspace.get_source(group.source_id), retained.source)
            or tuple(c.id for c in group.candidates) != tuple(row[0] for row in retained.versions)):
        raise ValueError('scene grounding model group content changed')
    for candidate, (ident, version, source, model) in zip(group.candidates, retained.versions):
        if (candidate.group_id != group_id or candidate.provenance != ('retained-scene-grounding-model',)
                or not _same(candidate.payload, version)
                or not _same(_model_snapshot(model), version.snapshot)
                or not _same(workspace.get_source(version.evidence_source_id), source)):
            raise ValueError('scene grounding model version content changed')
    if workspace.comparison_basis(group_id) != comparison:
        raise ValueError('model comparison changed during inspection')
    return retained, group, comparison


def fit_grounding_model(agent, training_records, validation_records, *, group_id=None,
                        max_atoms=3, max_patterns=512, max_matches=4096):
    """Retain a historical model fit; a refit withdraws previous admission.

    Incomplete fits remain available for diagnosis but cannot be admitted.
    Later teaching edits do not rewrite an existing historical model version.
    """
    published = None
    try:
        training, validation = tuple(training_records), tuple(validation_records)
        records = (*training, *validation)
        if not training or not validation or any(type(r) is not RetainedGroundingExample for r in records):
            raise ValueError('nonempty retained training and heldout records are required')
        if len({r.evidence_source_id for r in records}) != len(records):
            raise ValueError('training and heldout records must be unique and disjoint')
        previous = _model_state(agent, group_id) if group_id is not None else None
        _validate_examples(agent, records)
        model = fit_scene_grounding(tuple(r.example for r in training), tuple(r.example for r in validation),
                                    max_atoms=max_atoms, max_patterns=max_patterns, max_matches=max_matches)
        snapshot = _model_snapshot(model)
        evidence = agent.interpretations.add_source('Scene grounding fit from retained explicit supervision',
            modality='model-fit', provider='scene-grounding-learning', payload=deepcopy(snapshot),
            metadata={'training_evidence_ids': tuple(r.evidence_source_id for r in training),
                      'heldout_evidence_ids': tuple(r.evidence_source_id for r in validation),
                      'historical_supervision': True, 'max_atoms': max_atoms,
                      'max_patterns': max_patterns, 'max_matches': max_matches})
        version = _ModelVersion(snapshot, evidence.id)
        cached_version, cached_source, cached_model = deepcopy(version), deepcopy(evidence), deepcopy(model)
        _validate_examples(agent, records)
        workspace = agent.interpretations
        if previous is None:
            group = workspace.create_group(evidence.id, provenance=('scene grounding model versions',))
            retained = _ModelGroup(deepcopy(evidence), group.provenance, ())
        else:
            retained, group, comparison = _model_state(agent, group_id)
            if comparison != previous[2]:
                raise ValueError('model admission changed during refit')
        candidate = workspace.propose(group.id, version, provenance=('retained-scene-grounding-model',))
        _registry(agent, '_scene_grounding_models')[group.id] = replace(retained, versions=(
            *retained.versions, (candidate.id, cached_version, cached_source, cached_model)))
        published = group.id, candidate.id
        workspace.unset(group.id, reason='new grounding fit requires explicit admission')
        _validate_examples(agent, records)
        return SceneGroundingModelHandle(group.id, candidate.id, model.id, evidence.id)
    except Exception as error:
        if published is not None:
            agent.interpretations.reject(*published, reason='teaching changed during grounding model publication')
        return Unknown('scene_grounding_fit_unavailable', f'{type(error).__name__}: {error}')


def admit_grounding_model(agent, handle, *, reason):
    """Explicitly admit an authentic, complete model version without choosing meaning."""
    try:
        if type(handle) is not SceneGroundingModelHandle or type(reason) is not str or not reason.strip():
            raise ValueError('admission requires a retained model handle and explicit reason')
        _, group, comparison = _model_state(agent, handle.group_id)
        candidate = next(c for c in group.candidates if c.id == handle.candidate_id)
        if (candidate.rejected or candidate.payload.snapshot['_id'] != handle.model_id
                or candidate.payload.evidence_source_id != handle.evidence_source_id):
            raise ValueError('handle does not identify the retained model version')
        if not candidate.payload.snapshot['_complete']:
            raise ValueError('an incomplete grounding fit cannot be admitted')
        workspace = agent.interpretations
        if workspace.comparison_basis(group.id) != comparison:
            raise ValueError('model comparison changed before admission')
        expected = (comparison[0], group.revision + 1, candidate.id, comparison[3],
                    False, comparison[5], comparison[6])
        workspace.select(group.id, candidate.id, reason=reason, evidence_ids=(handle.evidence_source_id,))
        if workspace.comparison_basis(group.id) != expected:
            raise ValueError('model admission changed during selection')
        dependency = capture_dependency(workspace, group.id, basis=('explicit scene grounding model admission', reason),
                                        evidence_ids=(handle.evidence_source_id,))
        result = replace(handle, dependency=dependency)
        cached = deepcopy(result)
        _model_state(agent, group.id)
        if validate_dependencies(workspace, (dependency,)) is not True or workspace.comparison_basis(group.id) != expected:
            raise ValueError('model admission changed during dependency capture')
        _registry(agent, '_scene_grounding_admissions')[(group.id, candidate.id)] = cached
        return result
    except Exception as error:
        return Unknown('scene_grounding_admission_unavailable', f'{type(error).__name__}: {error}')


def get_grounding_model(agent, handle):
    """Return a detached authenticated model only while its admission is current."""
    try:
        if type(handle) is not SceneGroundingModelHandle or handle.dependency is None:
            raise ValueError('grounding model has not been explicitly admitted')
        cached = _registry(agent, '_scene_grounding_admissions').get((handle.group_id, handle.candidate_id))
        if cached is None or not _same(handle, cached):
            raise ValueError('unrecognized grounding admission')
        retained, group, comparison = _model_state(agent, handle.group_id)
        if group.selected_id != handle.candidate_id:
            raise ValueError('grounding model admission is no longer selected')
        model = deepcopy(next(row[3] for row in retained.versions if row[0] == handle.candidate_id))
        _model_state(agent, handle.group_id)
        if validate_dependencies(agent.interpretations, (handle.dependency,)) is not True:
            raise ValueError('grounding model admission changed')
        if agent.interpretations.comparison_basis(group.id) != comparison:
            raise ValueError('grounding model comparison changed during retrieval')
        return model
    except Exception as error:
        return Unknown('scene_grounding_model_unavailable', f'{type(error).__name__}: {error}')


def grounding_dependencies(agent, group_id, candidate_id):
    """Authenticate a learned child and return its exact model/scene commitments.

    Unregistered candidates receive no inferred dependencies. Registered but stale
    or modified children fail closed, so downstream execution cannot shed their
    justification merely by retaining a bound Ref.
    """
    return _grounding_dependencies(agent, group_id, candidate_id, frozenset())


def _grounding_dependencies(agent, group_id, candidate_id, visited, *, allow_unresolved=False):
    try:
        identity = (group_id, candidate_id)
        if identity in visited:
            raise ValueError('grounding derivation contains a cycle')
        visited = visited | {identity}
        workspace = agent.interpretations
        retained = _registry(agent, '_scene_grounding_children').get(identity)
        if retained is None:
            lineage = getattr(workspace, '_grounding_derivations', {}).get(candidate_id)
            if lineage is None:
                comparison = workspace.comparison_basis(group_id)
                group = workspace.get(group_id)
                candidate = next(c for c in group.candidates if c.id == candidate_id)
                unresolved = type(candidate.payload) is UnresolvedGrounding
                if workspace.comparison_basis(group_id) != comparison:
                    raise ValueError('unregistered grounding comparison changed during inspection')
                if unresolved:
                    return Unknown('grounding_unresolved', 'Unresolved grounding alternatives cannot authorize execution.')
                return ()
            lineage_group, parent_id, snapshot = lineage
            comparison = workspace.comparison_basis(group_id)
            group = workspace.get(group_id)
            candidate = next(c for c in group.candidates if c.id == candidate_id)
            parent = next(c for c in group.candidates if c.id == parent_id)
            if (lineage_group != group_id or candidate.group_id != group_id
                    or parent.group_id != group_id or not _same(candidate, snapshot)):
                raise ValueError('grounding derivation content changed')
            dependencies = _grounding_dependencies(agent, group_id, parent_id, visited)
            if workspace.comparison_basis(group_id) != comparison:
                raise ValueError('grounding descendant comparison changed during validation')
            return dependencies
        comparison = workspace.comparison_basis(group_id)
        group = workspace.get(group_id)
        candidate = next(c for c in group.candidates if c.id == candidate_id)
        parent = next(c for c in group.candidates if c.id == retained.parent.id)
        if (group.provenance != retained.group_provenance or group.source_id != retained.source.id
                or not _same(workspace.get_source(group.source_id), retained.source)
                or not _same(candidate, retained.candidate) or not _same(parent, retained.parent)
                or not _same(workspace.get_source(retained.report_source.id), retained.report_source)):
            raise ValueError('learned grounding candidate or its evidence changed')
        model = get_grounding_model(agent, retained.model)
        if isinstance(model, Unknown):
            raise ValueError(model.detail)
        _validate_snapshot(workspace, retained.scene, GRAPH_PROPOSALS)
        inherited = _grounding_dependencies(agent, group_id, retained.parent.id, visited)
        if isinstance(inherited, Unknown):
            raise ValueError(inherited.detail)
        if any(dep not in retained.dependencies for dep in inherited):
            raise ValueError('grounded parent dependencies changed')
        result = deepcopy(retained.dependencies)
        valid = validate_dependencies(workspace, retained.dependencies)
        if valid is not True:
            raise ValueError(valid.reason)
        _final_comparisons(workspace, (retained.scene,))
        if workspace.comparison_basis(group_id) != comparison:
            raise ValueError('learned grounding comparison changed during validation')
        if type(retained.candidate.payload) is UnresolvedGrounding and not allow_unresolved:
            return Unknown('grounding_unresolved', retained.candidate.payload.reason)
        return result
    except Exception as error:
        return Unknown('scene_grounding_dependency_changed', f'{type(error).__name__}: {error}')


def propose_scene_groundings(agent, admitted_handle, language_group_id, candidate_id, path,
                             scene_group_id, scene_candidate_id):
    """Publish all distinct inferred bindings under explicit model/scene commitments.

    Exhausted search is local to the model and graph, not proof of understanding.
    Incomplete searches or unresolved query rivals retain diagnostic evidence and
    publish no bindings; completeness alone does not settle conflicting models.
    Supported bindings coexist with nonexecutable known-unknown and unseen
    referent alternatives. Graph search exhaustion is not scene completeness.
    """
    published = []
    try:
        workspace = agent.interpretations
        model = get_grounding_model(agent, admitted_handle)
        if isinstance(model, Unknown):
            raise ValueError(model.detail)
        language = _capture(workspace, language_group_id, candidate_id, SentenceAlternative)
        scene = _capture(workspace, scene_group_id, scene_candidate_id, GRAPH_PROPOSALS)
        if scene.group.selected_id != scene_candidate_id:
            raise ValueError('grounding publication requires this explicitly selected scene')
        scene_dependency = capture_dependency(workspace, scene_group_id,
            basis=('explicit selected scene for learned grounding',), evidence_ids=(scene.source.id,))
        inherited = grounding_dependencies(agent, language_group_id, candidate_id)
        if isinstance(inherited, Unknown):
            raise ValueError(inherited.detail)
        dependencies = tuple(dict.fromkeys((*inherited, admitted_handle.dependency, scene_dependency)))
        description = _description(_candidate(language).payload, path)
        prediction = model.propose(description, _candidate(scene).payload.graph)
        evidence = workspace.add_source('Learned scene grounding search; alternatives remain unselected',
            modality='grounding-search', provider='scene-grounding-learning', payload=deepcopy(prediction),
            metadata={'model': admitted_handle, 'language': language, 'scene': scene, 'path': path,
                      'queries': model.queries, 'training_examples': model.training_examples,
                      'validation_examples': model.validation_examples, 'dependencies': dependencies})
        cached_evidence = deepcopy(evidence)
        _validate_snapshot(workspace, language, SentenceAlternative)
        _validate_snapshot(workspace, scene, GRAPH_PROPOSALS)
        if isinstance(get_grounding_model(agent, admitted_handle), Unknown):
            raise ValueError('model changed during grounding inference')
        if validate_dependencies(workspace, dependencies) is not True:
            raise ValueError('grounding support changed during inference')
        _final_comparisons(workspace, (language, scene))
        validated_queries = {query.id for query in model.queries
                             if query.validation_example_ids and not query.conflicting_validation_example_ids}
        usable_evidence = bool(prediction.query_evidence) and all(
            query_id in validated_queries and assessment.complete and not assessment.unresolved
            for query_id, assessment in prediction.query_evidence)
        if not prediction.complete or model.unresolved or not usable_evidence:
            return SceneGroundingReport((), prediction.complete, evidence.id, prediction.unresolved)
        # A complete query with no observed witness still has unknown roots and
        # may have unseen witnesses. Keep these alternatives even when a rival
        # query's empty answer prevents publishing any executable binding.
        references = (() if prediction.unresolved else
                      tuple(dict.fromkeys(match.reference for match in prediction.matches)))
        uncertain = {}
        unseen_queries = []
        for query_id, query_evidence in prediction.query_evidence:
            for root in query_evidence.roots:
                if root.status == 'unknown':
                    uncertain.setdefault(root.reference, []).append(query_id)
            if query_evidence.unseen_referents_possible:
                unseen_queries.append(query_id)
        # Keep every distinct reference. A display order is never a decision.
        for reference in references:
            graph = _candidate(scene).payload.graph
            if reference not in (graph_root(graph), *graph.nodes):
                raise ValueError('inferred reference is not declared in the scene')
            binding = MentionBinding(path, reference, (evidence.id, admitted_handle.evidence_source_id, scene.source.id),
                                     'admitted learned relational query applied to explicitly selected scene')
            child = propose_grounding(workspace, language_group_id, candidate_id, (binding,))
            registry = _registry(agent, '_scene_grounding_children')
            registry[(language_group_id, child.id)] = _GroundedCandidate(
                deepcopy(language.source), language.group.provenance, deepcopy(child),
                deepcopy(_candidate(language)), deepcopy(admitted_handle), deepcopy(scene),
                deepcopy(dependencies), cached_evidence)
            published.append(child.id)
            valid = grounding_dependencies(agent, language_group_id, child.id)
            if isinstance(valid, Unknown):
                raise ValueError(valid.detail)
        bound_ids = tuple(published)
        unresolved_ids = []
        unresolved_payloads = [UnresolvedGrounding(
            reference, tuple(dict.fromkeys(query_ids)),
            'Retained scene evidence does not resolve this known referent for every query.', evidence.id)
            for reference, query_ids in uncertain.items()]
        if unseen_queries:
            unresolved_payloads.append(UnresolvedGrounding(
                None, tuple(dict.fromkeys(unseen_queries)),
                'The retained scene may omit other referents; matching known roots is not exhaustive perception.',
                evidence.id))
        for unresolved in unresolved_payloads:
            child = workspace.propose(language_group_id, unresolved,
                provenance=('unresolved-scene-grounding', f'grounding-parent:{candidate_id}',
                            f'grounding-evidence:{evidence.id}'))
            _registry(agent, '_scene_grounding_children')[(language_group_id, child.id)] = _GroundedCandidate(
                deepcopy(language.source), language.group.provenance, deepcopy(child),
                deepcopy(_candidate(language)), deepcopy(admitted_handle), deepcopy(scene),
                deepcopy(dependencies), cached_evidence)
            published.append(child.id)
            unresolved_ids.append(child.id)
        for ident in published:
            valid = _grounding_dependencies(agent, language_group_id, ident, frozenset(), allow_unresolved=True)
            if isinstance(valid, Unknown):
                raise ValueError('grounding changed during batch publication: ' + valid.detail)
        expected_comparison = (*language.comparison[:3],
                               (*language.comparison[3], *published), *language.comparison[4:])
        if workspace.comparison_basis(language_group_id) != expected_comparison:
            raise ValueError('language comparison changed during grounding publication')
        return SceneGroundingReport(bound_ids, True, evidence.id, prediction.unresolved, tuple(unresolved_ids))
    except Exception as error:
        for ident in published:
            agent.interpretations.reject(language_group_id, ident, reason='grounding support changed during publication')
        return Unknown('scene_grounding_unavailable', f'{type(error).__name__}: {error}')


def propose_groundings(agent, admitted_handle, language_group_id, candidate_id, path,
                       graph_group_id, graph_candidate_id):
    """Publish grounded readings from authenticated visual or generic evidence graphs."""
    return propose_scene_groundings(agent, admitted_handle, language_group_id, candidate_id,
                                    path, graph_group_id, graph_candidate_id)
