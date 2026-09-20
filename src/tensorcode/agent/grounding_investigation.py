"""Retain grounding predictions before explicit structured teacher feedback.

Offered scene graphs and label answers are supplied, not discovered or understood
from a conversation. A feedback-driven refit appends training supervision while
preserving heldout evidence; the new version requires separate explicit admission.
"""
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from ..learning.experience import _same
from ..learning.grounding_investigation import investigate_grounding
from ..outcomes import Unknown
from ..records import Ref
from .evidence_graph import graph_root
from .scene_grounding import GRAPH_PROPOSALS
from .scene_grounding import (
    RetainedGroundingExample, SceneGroundingModelHandle, _capture, _candidate,
    _description, _final_comparisons, _model_state, _validate_example,
    _validate_snapshot, fit_grounding_model, get_grounding_model,
    retain_grounding_example,
)
from .task_dependencies import validate_dependencies
from .understand import SentenceAlternative


@dataclass(frozen=True)
class GroundingInvestigationProposal:
    id: str
    evidence_source_id: str
    investigation: object


@dataclass(frozen=True)
class GroundingFeedback:
    id: str
    evidence_source_id: str
    teaching_record: RetainedGroundingExample
    confirmed_query_ids: tuple[str, ...]
    contradicted_query_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Prepared:
    proposal: GroundingInvestigationProposal
    evidence: object
    model: SceneGroundingModelHandle
    language: object
    scenes: tuple
    path: tuple


@dataclass(frozen=True)
class _Feedback:
    feedback: GroundingFeedback
    evidence: object
    proposal_id: str


def _registry(agent, name):
    if not hasattr(agent, name):
        setattr(agent, name, {})
    return getattr(agent, name)


def _proposals(agent):
    return _registry(agent, '_grounding_investigation_proposals')


def _feedback(agent):
    return _registry(agent, '_grounding_investigation_feedback')


def _states(agent):
    return _registry(agent, '_grounding_investigation_states')


def _validate_prepared(agent, retained):
    workspace = agent.interpretations
    if not _same(workspace.get_source(retained.evidence.id), retained.evidence):
        raise ValueError('retained pre-feedback predictions changed')
    model = get_grounding_model(agent, retained.model)
    if isinstance(model, Unknown):
        raise ValueError(model.detail)
    _validate_snapshot(workspace, retained.language, SentenceAlternative)
    for snapshot in retained.scenes:
        _validate_snapshot(workspace, snapshot, GRAPH_PROPOSALS)
    valid = validate_dependencies(workspace, (retained.model.dependency,))
    if valid is not True:
        raise ValueError(valid.reason)
    _final_comparisons(workspace, (retained.language, *retained.scenes))
    return model


def _read_proposal(agent, proposal):
    if type(proposal) is not GroundingInvestigationProposal:
        raise ValueError('a retained investigation proposal is required')
    retained = _proposals(agent).get(proposal.id)
    if retained is None or not _same(proposal, retained.proposal):
        raise ValueError('unrecognized or modified grounding investigation')
    _validate_prepared(agent, retained)
    return retained


def prepare_grounding_investigation(agent, admitted_handle, language_group_id,
                                   candidate_id, path, scene_candidates):
    """Retain all query predictions before asking for explicit alignments.

    Only explicitly offered scenes disjoint from the original fit's scene/entity
    identities are ranked. Excluded offers remain documented in source metadata.
    Nothing is selected, bound, asserted, or executed.
    """
    try:
        model = get_grounding_model(agent, admitted_handle)
        if isinstance(model, Unknown):
            raise ValueError(model.detail)
        if (type(scene_candidates) is not tuple or not scene_candidates
                or any(type(pair) is not tuple or len(pair) != 2
                       or any(type(value) is not str for value in pair) for pair in scene_candidates)
                or len(set(scene_candidates)) != len(scene_candidates)):
            raise ValueError('explicit unique (scene group, candidate) tuples are required')
        workspace = agent.interpretations
        language = _capture(workspace, language_group_id, candidate_id, SentenceAlternative)
        scenes = tuple(_capture(workspace, gid, cid, GRAPH_PROPOSALS) for gid, cid in scene_candidates)
        description = _description(_candidate(language).payload, path)
        learned_ids = {reference for example in (*model.training_examples, *model.validation_examples)
                       for reference in (graph_root(example.scene), *example.scene.nodes)}
        eligible, exclusions = [], []
        for snapshot in scenes:
            graph = _candidate(snapshot).payload.graph
            if learned_ids.intersection((graph_root(graph), *graph.nodes)):
                exclusions.append((snapshot.group_id, snapshot.candidate_id, 'scene or entity identity overlaps original fit'))
            else:
                eligible.append(graph)
        if not eligible:
            raise ValueError('no offered scene is disjoint from original training and heldout evidence')
        investigation = investigate_grounding(model, description, tuple(eligible))
        evidence = workspace.add_source('Grounding query predictions retained before teacher feedback',
            modality='grounding-investigation', provider='grounding-investigation', payload=deepcopy(investigation),
            metadata={'model': admitted_handle, 'language': language, 'scenes': scenes,
                      'path': path, 'queries': model.queries,
                      'excluded_scenes': tuple(exclusions), 'feedback_observed': False})
        proposal = GroundingInvestigationProposal('grounding-investigation:' + uuid4().hex, evidence.id, investigation)
        retained = deepcopy(_Prepared(proposal, evidence, admitted_handle, language, scenes, path))
        result = deepcopy(proposal)
        _validate_prepared(agent, retained)
        _proposals(agent)[proposal.id] = retained
        _states(agent)[proposal.id] = 'prepared'
        return result
    except Exception as error:
        return Unknown('grounding_investigation_unavailable', f'{type(error).__name__}: {error}')


def record_grounding_feedback(agent, proposal, scene_id, positive_refs, negative_refs=(), *, basis):
    """Record a teacher's explicit answer for one offered eligible scene.

    Ranking is a recommendation; the caller explicitly chooses the scene. Unlabeled references never become
    negatives. Query agreement records consistency with labels, not visual truth.
    """
    reserved = None
    try:
        if type(proposal) is not GroundingInvestigationProposal:
            raise ValueError('a retained investigation proposal is required')
        authentic = _proposals(agent).get(proposal.id)
        if authentic is None or not _same(proposal, authentic.proposal):
            raise ValueError('unrecognized or modified grounding investigation')
        if _states(agent).get(proposal.id) != 'prepared':
            raise ValueError('investigation feedback has already been reserved or recorded')
        _states(agent)[proposal.id] = 'recording-feedback'
        reserved = proposal.id
        retained = _read_proposal(agent, proposal)
        investigation = retained.proposal.investigation
        if (not investigation.complete or type(scene_id) is not Ref
                or scene_id not in tuple(graph_root(graph) for graph in investigation.scenes)):
            raise ValueError('feedback must choose an offered novel scene with complete predictions')
        if type(basis) is not tuple or not basis or any(type(item) is not str or not item.strip() for item in basis):
            raise ValueError('feedback requires an explicit nonempty basis tuple')
        offered = [snapshot for snapshot in retained.scenes if graph_root(_candidate(snapshot).payload.graph) == scene_id]
        if len(offered) != 1:
            raise ValueError('feedback scene identity does not identify one retained graph candidate')
        snapshot = offered[0]
        teaching = retain_grounding_example(agent, retained.language.group_id,
            retained.language.candidate_id, retained.path, snapshot.group_id, snapshot.candidate_id,
            positive_refs, negative_refs, basis=basis)
        if isinstance(teaching, Unknown):
            raise ValueError(teaching.detail)
        if not _same(teaching.language, retained.language) or not _same(teaching.scene, snapshot):
            raise ValueError('feedback teaching differs from the pre-feedback interpretation comparison')
        _validate_example(agent, teaching)
        predictions = next(row for row in investigation.predictions if row.scene_id == scene_id)
        confirmed, contradicted = [], []
        for prediction in predictions.predictions:
            if not prediction.complete:
                raise ValueError('feedback cannot assess an incomplete query prediction')
            targets = set(prediction.references)
            supported = set(teaching.example.positive_refs) <= targets and not set(teaching.example.negative_refs) & targets
            (confirmed if supported else contradicted).append(prediction.query_id)
        evidence = agent.interpretations.add_source('Explicit grounding teacher feedback against prior predictions',
            modality='grounding-feedback', provider='explicit-grounding-teacher', payload=deepcopy(teaching),
            metadata={'proposal_id': proposal.id, 'prediction_source_id': retained.evidence.id,
                      'scene_id': scene_id, 'predictions': deepcopy(predictions), 'basis': basis,
                      'query_patterns': tuple((query.id, query.query) for query in retained.evidence.metadata['queries']
                                              if query.id in (*confirmed, *contradicted)),
                      'positive_refs': teaching.example.positive_refs, 'negative_refs': teaching.example.negative_refs,
                      'teaching_evidence_source_id': teaching.evidence_source_id,
                      'confirmed_query_ids': tuple(confirmed), 'contradicted_query_ids': tuple(contradicted)})
        feedback = GroundingFeedback('grounding-feedback:' + uuid4().hex, evidence.id, teaching,
                                      tuple(confirmed), tuple(contradicted))
        cached = deepcopy(_Feedback(feedback, evidence, proposal.id))
        result = deepcopy(feedback)
        _validate_example(agent, teaching)
        _validate_prepared(agent, retained)
        _feedback(agent)[feedback.id] = cached
        _states(agent)[feedback.id] = 'available'
        _states(agent)[proposal.id] = 'feedback-recorded'
        return result
    except Exception as error:
        if reserved is not None:
            _states(agent)[reserved] = 'feedback-failed'
        return Unknown('grounding_feedback_unavailable', f'{type(error).__name__}: {error}')


def _read_feedback(agent, feedback):
    if type(feedback) is not GroundingFeedback:
        raise ValueError('retained grounding feedback is required')
    cached = _feedback(agent).get(feedback.id)
    if cached is None or not _same(feedback, cached.feedback):
        raise ValueError('unrecognized or modified grounding feedback')
    if not _same(agent.interpretations.get_source(cached.evidence.id), cached.evidence):
        raise ValueError('retained feedback source changed')
    retained = _proposals(agent)[cached.proposal_id]
    _validate_example(agent, cached.feedback.teaching_record)
    _validate_prepared(agent, retained)
    return cached, retained


def refit_grounding_from_feedback(agent, feedback):
    """Consume one authenticated answer into training and withdraw old admission.

    Heldout examples and search bounds are preserved. The returned version is
    unadmitted. Failed attempts are retained as failed, never implicitly replayed.
    """
    reserved = None
    published = None
    try:
        if type(feedback) is not GroundingFeedback:
            raise ValueError('retained grounding feedback is required')
        authentic = _feedback(agent).get(feedback.id)
        if authentic is None or not _same(feedback, authentic.feedback):
            raise ValueError('unrecognized or modified grounding feedback')
        if _states(agent).get(feedback.id) != 'available':
            raise ValueError('grounding feedback is unavailable or already consumed')
        _states(agent)[feedback.id] = 'refitting'
        reserved = feedback.id
        cached, retained = _read_feedback(agent, feedback)
        model = get_grounding_model(agent, retained.model)
        if isinstance(model, Unknown):
            raise ValueError(model.detail)
        fit_source = agent.interpretations.get_source(retained.model.evidence_source_id)
        train_ids = fit_source.metadata['training_evidence_ids']
        heldout_ids = fit_source.metadata['heldout_evidence_ids']
        registry = getattr(agent, '_scene_grounding_examples', {})
        training = tuple(deepcopy(registry[identity][0]) for identity in train_ids)
        heldout = tuple(deepcopy(registry[identity][0]) for identity in heldout_ids)
        if (not _same(tuple(record.example for record in training), model.training_examples)
                or not _same(tuple(record.example for record in heldout), model.validation_examples)):
            raise ValueError('refit records do not reproduce the admitted historical training split')
        for record in (*training, *heldout):
            _validate_example(agent, record)
        _read_feedback(agent, feedback)
        appended = cached.feedback.teaching_record
        updated = fit_grounding_model(agent, (*training, appended), heldout,
            group_id=retained.model.group_id, max_atoms=fit_source.metadata['max_atoms'],
            max_patterns=fit_source.metadata['max_patterns'], max_matches=fit_source.metadata['max_matches'])
        if isinstance(updated, Unknown):
            raise ValueError(updated.detail)
        published = (updated.group_id, updated.candidate_id)
        # The old model dependency is intentionally stale now. Authenticate the
        # retained prediction/answer content and teaching comparisons separately.
        if (not _same(agent.interpretations.get_source(cached.evidence.id), cached.evidence)
                or not _same(agent.interpretations.get_source(retained.evidence.id), retained.evidence)):
            raise ValueError('feedback or original predictions changed during refitting')
        _validate_example(agent, appended)
        _validate_snapshot(agent.interpretations, retained.language, SentenceAlternative)
        for snapshot in retained.scenes:
            _validate_snapshot(agent.interpretations, snapshot, GRAPH_PROPOSALS)
        _, group, comparison = _model_state(agent, updated.group_id)
        version_source = agent.interpretations.get_source(updated.evidence_source_id)
        if (updated.group_id != retained.model.group_id or group.selected_id is not None
                or version_source.metadata['training_evidence_ids'] != (*train_ids, appended.evidence_source_id)
                or version_source.metadata['heldout_evidence_ids'] != heldout_ids):
            raise ValueError('refit did not preserve the historical split or withdraw admission')
        _final_comparisons(agent.interpretations, (retained.language, *retained.scenes))
        if agent.interpretations.comparison_basis(group.id) != comparison:
            raise ValueError('new model comparison changed after refitting')
        _states(agent)[feedback.id] = 'consumed'
        return updated
    except Exception as error:
        if published is not None:
            agent.interpretations.reject(*published, reason='feedback evidence or comparison changed during refitting')
        if reserved is not None:
            _states(agent)[reserved] = 'refit-failed'
        return Unknown('grounding_feedback_refit_unavailable', f'{type(error).__name__}: {error}')
