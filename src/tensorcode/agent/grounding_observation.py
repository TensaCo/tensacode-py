"""Targeted read-only observation of retained scene-grounding uncertainty.

The caller explicitly chooses a generated probe and a provider. Predictions are
retained before the call. Answers yield scene alternatives, never assertions,
reference bindings, scene selection, or actions.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from uuid import uuid4

from ..learning.experience import _same
from ..learning.grounding_probes import propose_grounding_probes
from ..outcomes import Unknown
from .plugin import Plugin
from .scene import SceneProposal
from .scene_grounding import (
    GRAPH_PROPOSALS, _capture, _candidate, _description, _final_comparisons, _validate_snapshot,
    get_grounding_model, grounding_dependencies,
)
from .task_dependencies import _expected_basis, capture_dependency, validate_dependencies
from .understand import SentenceAlternative


@dataclass(frozen=True)
class GroundingObservationProposal:
    id: str
    evidence_source_id: str
    plan: object


@dataclass(frozen=True)
class GroundingObservationResult:
    id: str
    evidence_source_id: str
    scene_candidate_id: str | None
    observation: bool | Unknown


@dataclass(frozen=True)
class _Prepared:
    proposal: GroundingObservationProposal
    evidence: object
    model: object
    language: object
    scene: object
    path: tuple
    dependencies: tuple
    supporting_snapshots: tuple
    lineage: tuple


def _registry(agent, name):
    if not hasattr(agent, name):
        setattr(agent, name, {})
    return getattr(agent, name)


def _lineage(agent, group_id, candidate_id):
    """Capture authenticated ancestry and every retained grounding-search source."""
    records, visited = [], set()
    while True:
        if candidate_id in visited:
            raise ValueError('grounding lineage cycle')
        visited.add(candidate_id)
        child = getattr(agent, '_scene_grounding_children', {}).get((group_id, candidate_id))
        if child is not None:
            records.append(('learned', candidate_id, deepcopy(child),
                            agent.interpretations.get_source(child.report_source.id)))
            candidate_id = child.parent.id
            continue
        derivation = getattr(agent.interpretations, '_grounding_derivations', {}).get(candidate_id)
        if derivation is None:
            return tuple(records)
        if derivation[0] != group_id:
            raise ValueError('grounding lineage group mismatch')
        records.append(('derived', candidate_id, deepcopy(derivation)))
        candidate_id = derivation[1]


def _validate(agent, retained, *, published=None):
    workspace = agent.interpretations
    if not _same(_lineage(agent, retained.language.group_id, retained.language.candidate_id), retained.lineage):
        raise ValueError('grounding lineage or search evidence changed')
    if not _same(workspace.get_source(retained.evidence.id), retained.evidence):
        raise ValueError('retained observation plan changed')
    model = get_grounding_model(agent, retained.model)
    if isinstance(model, Unknown):
        raise ValueError(model.detail)
    _validate_snapshot(workspace, retained.language, SentenceAlternative)
    supporting = tuple(snapshot for snapshot in retained.supporting_snapshots
                       if published is None or snapshot.group_id != retained.scene.group_id)
    for snapshot in supporting:
        _validate_snapshot(workspace, snapshot, type(_candidate(snapshot).payload))
    for dependency in retained.dependencies:
        if dependency.group_id in getattr(agent, '_scene_grounding_models', {}):
            admission = getattr(agent, '_scene_grounding_admissions', {}).get((dependency.group_id, dependency.candidate_id))
            if admission is None or admission.dependency != dependency or isinstance(get_grounding_model(agent, admission), Unknown):
                raise ValueError('supporting grounding model changed')
    inherited = grounding_dependencies(agent, retained.language.group_id, retained.language.candidate_id)
    if isinstance(inherited, Unknown):
        # A deliberate scene alternative invalidates old scene-dependent readings.
        # Their exact content is checked below; do not excuse any other change.
        if published is None:
            raise ValueError(inherited.detail)
    elif any(dependency not in retained.dependencies for dependency in inherited):
        raise ValueError('language grounding dependencies changed')
    if published is None:
        _validate_snapshot(workspace, retained.scene, GRAPH_PROPOSALS)
        valid_dependencies = retained.dependencies
    else:
        current = _capture(workspace, retained.scene.group_id, retained.scene.candidate_id, GRAPH_PROPOSALS)
        expected = (*retained.scene.comparison[:3], (*retained.scene.comparison[3], published.id),
                    *retained.scene.comparison[4:])
        if (current.comparison != expected or not _same(current.source, retained.scene.source)
                or current.frontier != retained.scene.frontier
                or not current.group.candidates or not _same(current.group.candidates[-1], published)
                or not _same(replace(current.group, candidates=current.group.candidates[:-1]), retained.scene.group)):
            raise ValueError('scene comparison changed beyond the proposed observation alternative')
        valid_dependencies = tuple(d for d in retained.dependencies if d.group_id != retained.scene.group_id)
    valid = validate_dependencies(workspace, valid_dependencies)
    if valid is not True:
        raise ValueError(valid.reason)
    _final_comparisons(workspace, (retained.language, *supporting))
    if published is None:
        _final_comparisons(workspace, (retained.scene,))
    elif workspace.comparison_basis(retained.scene.group_id) != expected:
        raise ValueError('scene comparison changed during post-publication validation')



def _final_epoch(agent, retained, published=None):
    """Callback-free guard after the last source/payload copy."""
    workspace = agent.interpretations
    snapshots = (retained.language, *(s for s in retained.supporting_snapshots
                 if published is None or s.group_id != retained.scene.group_id))
    _final_comparisons(workspace, snapshots)
    for dependency in retained.dependencies:
        if published is not None and dependency.group_id == retained.scene.group_id:
            continue
        if workspace.comparison_basis(dependency.group_id) != _expected_basis(dependency):
            raise ValueError('observation dependency changed during final evidence read')
    expected = retained.scene.comparison if published is None else (
        *retained.scene.comparison[:3], (*retained.scene.comparison[3], published.id),
        *retained.scene.comparison[4:])
    if workspace.comparison_basis(retained.scene.group_id) != expected:
        raise ValueError('scene comparison changed during final evidence read')


def prepare_grounding_observation(agent, admitted_handle, language_group_id,
                                  candidate_id, path, scene_group_id,
                                  scene_candidate_id, root, *, max_probes=64, max_states=65536):
    """Retain model-derived full-proposition probes and both outcome predictions."""
    try:
        workspace = agent.interpretations
        model = get_grounding_model(agent, admitted_handle)
        if isinstance(model, Unknown):
            raise ValueError(model.detail)
        language = _capture(workspace, language_group_id, candidate_id, SentenceAlternative)
        scene = _capture(workspace, scene_group_id, scene_candidate_id, GRAPH_PROPOSALS)
        if scene.group.selected_id != scene_candidate_id:
            raise ValueError('targeted observation requires the explicitly selected scene')
        scene_dependency = capture_dependency(workspace, scene_group_id,
            basis=('selected scene for targeted proposition observation',), evidence_ids=(scene.source.id,))
        inherited = grounding_dependencies(agent, language_group_id, candidate_id)
        if isinstance(inherited, Unknown):
            raise ValueError(inherited.detail)
        dependencies = tuple(dict.fromkeys((*inherited, admitted_handle.dependency, scene_dependency)))
        supporting = tuple(_capture(workspace, dependency.group_id, dependency.candidate_id,
            type(workspace.get(dependency.group_id).selected.payload)) for dependency in dependencies)
        description = _description(_candidate(language).payload, path)
        plan = propose_grounding_probes(model, description, _candidate(scene).payload.graph, root,
                                        max_probes=max_probes, max_states=max_states)
        evidence = workspace.add_source('Targeted grounding probes predicted before observation',
            modality='grounding-observation-plan', provider='grounding-probe-planner', payload=deepcopy(plan),
            metadata={'model': admitted_handle, 'language': language, 'scene': scene, 'path': path,
                      'evidence_source': scene.source, 'dependencies': dependencies,
                      'max_probes': max_probes, 'max_states': max_states})
        proposal = GroundingObservationProposal('grounding-observation:' + uuid4().hex, evidence.id, plan)
        retained = deepcopy(_Prepared(proposal, evidence, admitted_handle, language, scene, path, dependencies, supporting,
            _lineage(agent, language_group_id, candidate_id)))
        result = deepcopy(proposal)
        _validate(agent, retained)
        _final_epoch(agent, retained)
        _registry(agent, '_grounding_observation_proposals')[proposal.id] = retained
        _registry(agent, '_grounding_observation_states')[proposal.id] = 'prepared'
        return result
    except Exception as error:
        return Unknown('grounding_observation_unavailable', f'{type(error).__name__}: {error}')


def observe_grounding_proposal(agent, proposal, probe_id, observer):
    """Consume one explicit read-only observation, retaining failure and uncertainty.

    True supports the asked proposition. False supports precisely its opposite
    polarity with all other metadata intact. Neither answer selects a scene.
    """
    reserved = None
    request = observed = published = None
    provider_name = ''
    try:
        if type(proposal) is not GroundingObservationProposal:
            raise ValueError('an authenticated observation proposal is required')
        retained = _registry(agent, '_grounding_observation_proposals').get(proposal.id)
        if retained is None or not _same(proposal, retained.proposal):
            raise ValueError('unrecognized or modified observation proposal')
        if not isinstance(observer, Plugin) or type(observer.name) is not str or not observer.name.strip():
            raise ValueError('observation requires an explicit named Plugin provider')
        provider_name = observer.name
        plan = retained.proposal.plan
        probe = next((probe for probe in plan.probes if probe.id == probe_id), None)
        if not plan.complete or probe is None:
            raise ValueError('choose an explicitly retained probe from a complete plan')
        states = _registry(agent, '_grounding_observation_states')
        if states.get(proposal.id) != 'prepared':
            raise ValueError('observation proposal is already consumed or in progress')
        states[proposal.id] = 'observing'
        reserved = proposal.id
        _validate(agent, retained)
        workspace = agent.interpretations
        request = workspace.add_source('Explicit read-only scene proposition observation request',
            modality='grounding-observation-request', provider=provider_name, payload=deepcopy(probe),
            metadata={'proposal_id': proposal.id, 'prediction_source_id': proposal.evidence_source_id,
                      'evidence_source': retained.scene.source, 'scene': retained.scene,
                      'dependencies': retained.dependencies})
        expected_request = deepcopy(request)
        _validate(agent, retained)
        if not _same(workspace.get_source(request.id), expected_request):
            raise ValueError('observation request changed before provider invocation')
        proposition_argument = deepcopy(probe.proposition)
        scene_argument = deepcopy(_candidate(retained.scene).payload.graph)
        source_argument = deepcopy(retained.scene.source)
        observe = (observer.observe_scene_proposition if type(_candidate(retained.scene).payload) is SceneProposal
                   else observer.observe_graph_proposition)
        _validate(agent, retained)
        if observer.name != provider_name or not _same(workspace.get_source(request.id), expected_request):
            raise ValueError('provider or request changed during argument preparation')
        _final_epoch(agent, retained)
        error = None
        try:
            raw = observe(proposition_argument, scene_argument, source_argument)
        except Exception as exception:
            error = {'type': type(exception).__name__, 'message': str(exception)}
            raw = None
            observation = Unknown('scene_observation_failed', f"{error['type']}: {error['message']}")
        else:
            observation = raw if type(raw) is bool or isinstance(raw, Unknown) else Unknown(
                'invalid_scene_observation', 'provider must return bool or Unknown')
        authentication_error = None
        try:
            _validate(agent, retained)
            if observer.name != provider_name:
                raise ValueError('observation provider identity changed during invocation')
            if not _same(workspace.get_source(request.id), expected_request):
                raise ValueError('observation request changed during invocation')
        except Exception as exception:
            authentication_error = f'{type(exception).__name__}: {exception}'
        observed = workspace.add_source('Read-only scene proposition observation result',
            modality='grounding-observation-result', provider=provider_name,
            payload={'raw_result': raw, 'observation': observation, 'error': error},
            metadata={'proposal_id': proposal.id, 'request_source_id': request.id,
                      'prediction_source_id': proposal.evidence_source_id, 'probe_id': probe.id,
                      'authenticated': authentication_error is None, 'authentication_error': authentication_error})
        expected_observed = deepcopy(observed)
        if authentication_error:
            raise ValueError(authentication_error)
        _validate(agent, retained)
        if (not _same(workspace.get_source(request.id), expected_request)
                or not _same(workspace.get_source(observed.id), expected_observed)):
            raise ValueError('request or observation evidence changed during retention')
        if type(observation) is bool:
            fact = deepcopy(probe.proposition) if observation else replace(deepcopy(probe.proposition), polarity=not probe.proposition.polarity)
            original = _candidate(retained.scene).payload
            graph = replace(original.graph, propositions=(*original.graph.propositions, fact))
            graph.validate()
            proposed = replace(original, graph=graph, provenance=(*original.provenance,
                f'observation:{observed.id}', f'observer:{provider_name}'))
            expected_payload = deepcopy(proposed)
            provenance = ('targeted-scene-observation', f'parent:{retained.scene.candidate_id}',
                          f'observation-source:{observed.id}')
            candidate = workspace.propose(retained.scene.group_id, proposed, provenance=provenance)
            published = replace(candidate, group_id=retained.scene.group_id, payload=expected_payload, provenance=provenance)
            _validate(agent, retained, published=published)
        if (not _same(workspace.get_source(request.id), expected_request)
                or not _same(workspace.get_source(observed.id), expected_observed)):
            raise ValueError('observation evidence changed during scene publication')
        result = GroundingObservationResult('grounding-observation-result:' + uuid4().hex,
            observed.id, published.id if published is not None else None, deepcopy(observation))
        cached = deepcopy((result, observed))
        returned = deepcopy(result)
        _validate(agent, retained, published=published)
        if (not _same(workspace.get_source(request.id), expected_request)
                or not _same(workspace.get_source(observed.id), expected_observed)):
            raise ValueError('observation evidence changed during final retention')
        _final_epoch(agent, retained, published)
        _registry(agent, '_grounding_observation_results')[result.id] = cached
        states[proposal.id] = 'consumed'
        return returned
    except Exception as error:
        if published is not None:
            agent.interpretations.reject(retained.scene.group_id, published.id,
                                         reason='observation support changed during scene publication')
        if reserved is not None:
            _registry(agent, '_grounding_observation_states')[reserved] = 'failed'
        if request is not None:
            # Preserve a terminal authentication record even when retention or
            # scene publication fails after the provider has already answered.
            agent.interpretations.add_source('Grounding observation attempt aborted',
                modality='grounding-observation-aborted', provider=provider_name,
                payload=Unknown('grounding_observation_changed', f'{type(error).__name__}: {error}'),
                metadata={'proposal_id': proposal.id, 'request_source_id': request.id,
                          'observation_source_id': observed.id if observed is not None else None,
                          'authenticated': False})
        return Unknown('grounding_observation_changed', f'{type(error).__name__}: {error}')
