"""Explicit activation of authenticated, selected document references.

This bridge supplies transport identity, not an affordance or an effect model.
A caller chooses activation; no labels, selectors, ancestor guesses, or natural
language action routing are synthesized here.
"""
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from ..learning.experience import _same
from ..outcomes import Unknown
from ..records import Ref
from ..runtime import use
from .document_evidence import RetainedDocument, retain_document_snapshot
from .evidence_graph import GraphProposal
from .scene_grounding import _capture, _candidate, _description, _validate_snapshot, _final_comparisons, grounding_dependencies
from .task_dependencies import _expected_basis, capture_dependency, validate_dependencies
from .understand import SentenceAlternative
from .plugin import Call


@dataclass(frozen=True)
class DocumentActionProposal:
    id: str
    evidence_source_id: str
    target: Ref


@dataclass(frozen=True)
class DocumentActionResult:
    id: str
    evidence_source_id: str
    receipt: object
    events: tuple


@dataclass(frozen=True)
class DocumentActionContext:
    """Detached execution identity and selected interpretation dependencies."""

    action: Call
    dependencies: tuple
    target: Ref


def _registry(agent, name):
    if not hasattr(agent, name):
        setattr(agent, name, {})
    return getattr(agent, name)


def _required(result):
    if result is not True:
        raise ValueError(str(result))


def capture_browser_document(agent, provider):
    """Retain a provider-owned capture without selecting its graph interpretation."""
    try:
        capture = provider.capture_document()
        if isinstance(capture, Unknown):
            return capture
        _required(provider.authenticate_document_capture(capture))
        expected = deepcopy(capture)
        document = retain_document_snapshot(agent, expected.snapshot, provider.name)
        if isinstance(document, Unknown):
            return document
        snapshot = _capture(agent.interpretations, document.group_id, document.candidate_id, GraphProposal)
        _required(provider.authenticate_document_capture(capture))
        if not _same(capture, expected) or not _same(snapshot.source.payload, expected.snapshot):
            raise ValueError('provider capture changed during retention')
        cached = (provider, deepcopy(expected), deepcopy(document), deepcopy(snapshot.source), deepcopy(_candidate(snapshot)))
        result = deepcopy(document)
        _validate_snapshot(agent.interpretations, snapshot, GraphProposal)
        _final_comparisons(agent.interpretations, (snapshot,))
        _registry(agent, '_document_captures')[(document.group_id, document.candidate_id)] = cached
        return result
    except Exception as error:
        return Unknown('document_capture_unavailable', f'{type(error).__name__}: {error}')


def _association(agent, provider, document):
    associated = _registry(agent, '_document_captures').get((document.group_id, document.candidate_id))
    if associated is None or associated[0] is not provider:
        raise ValueError('document is not associated with this provider instance')
    _, capture, identity, source, candidate = associated
    if provider.name != source.provider:
        raise ValueError('document provider identity changed')
    _required(provider.authenticate_document_capture(deepcopy(capture)))
    if (identity.source_id != document.source.id or not _same(source, document.source)
            or not _same(candidate, _candidate(document))):
        raise ValueError('document source or canonical graph changed')
    return associated


def _validate(agent, provider, retained, *, request=None):
    proposal, evidence, language, document, dependencies, token, capability = retained
    _association(agent, provider, document)
    _required(provider.validate_document_target(token))
    if not any(_same(cap, capability) for cap in provider.capabilities()):
        raise ValueError('activation capability changed')
    workspace = agent.interpretations
    if not _same(workspace.get_source(evidence.id), evidence):
        raise ValueError('document action evidence changed')
    if request is not None and not _same(workspace.get_source(request.id), request):
        raise ValueError('document action request changed')
    _validate_snapshot(workspace, language, SentenceAlternative)
    _validate_snapshot(workspace, document, GraphProposal)
    inherited = grounding_dependencies(agent, language.group_id, language.candidate_id)
    if isinstance(inherited, Unknown) or any(dep not in dependencies for dep in inherited):
        raise ValueError('grounded language support changed')
    _required(validate_dependencies(workspace, dependencies))
    _final_comparisons(workspace, (language, document))
    for dependency in dependencies:
        if workspace.comparison_basis(dependency.group_id) != _expected_basis(dependency):
            raise ValueError('document action dependency changed during final validation')


def prepare_document_action(agent, provider, language_group_id, candidate_id, path,
                            document_group_id, document_candidate_id):
    """Prepare explicit activation of exactly one selected, bound graph node."""
    try:
        workspace = agent.interpretations
        language = _capture(workspace, language_group_id, candidate_id, SentenceAlternative)
        document = _capture(workspace, document_group_id, document_candidate_id, GraphProposal)
        if language.group.selected_id != candidate_id or document.group.selected_id != document_candidate_id:
            raise ValueError('explicit language and document selections are required')
        associated = _association(agent, provider, document)
        entity = _description(_candidate(language).payload, path)
        target = entity.ref
        graph = _candidate(document).payload.graph
        if type(target) is not Ref or target not in graph.nodes:
            raise ValueError('chosen Entity must bind an exact retained document node')
        # The projection emits nodes in document/array order. No Ref string is parsed.
        position = graph.nodes.index(target)
        location = None
        for document_index, raw_document in enumerate(associated[1].snapshot['documents']):
            count = len(raw_document['nodes']['nodeType'])
            if position < count:
                location = (document_index, position)
                break
            position -= count
        if location is None:
            raise ValueError('reference has no canonical capture position')
        inherited = grounding_dependencies(agent, language_group_id, candidate_id)
        if isinstance(inherited, Unknown):
            raise ValueError(str(inherited))
        dependencies = tuple(dict.fromkeys((*inherited,
            capture_dependency(workspace, language_group_id, basis=('explicit activation reading',)),
            capture_dependency(workspace, document_group_id, basis=('explicit activation document',)))))
        capabilities = [cap for cap in provider.capabilities() if cap.name == 'activate_node']
        if len(capabilities) != 1 or tuple(p.name for p in capabilities[0].params) != ('target',):
            raise ValueError('provider must declare one activate_node(target) capability')
        token = provider.prepare_document_target(deepcopy(associated[1]), *location)
        if isinstance(token, Unknown):
            return token
        if type(token) is not str or not token.strip():
            raise ValueError('provider target must be an opaque nonempty string')
        evidence = workspace.add_source('Explicit document node activation proposal', modality='document-action-plan',
            provider=provider.name, payload={'target': target, 'location': location, 'operation': 'activate_node'},
            metadata={'language': language, 'document': document, 'dependencies': dependencies,
                      'capture_id': associated[1].id})
        proposal = DocumentActionProposal('document-action:' + uuid4().hex, evidence.id, target)
        retained = deepcopy((proposal, evidence, language, document, dependencies, token, capabilities[0]))
        result = deepcopy(proposal)
        _validate(agent, provider, retained)
        _registry(agent, '_document_actions')[proposal.id] = (provider, retained)
        _registry(agent, '_document_action_states')[proposal.id] = 'prepared'
        return result
    except Exception as error:
        return Unknown('document_action_unavailable', f'{type(error).__name__}: {error}')


def document_action_context(agent, provider, proposal):
    """Authenticate a prepared proposal without consuming or executing it."""
    try:
        record = _registry(agent, '_document_actions').get(proposal.id)
        if record is None or record[0] is not provider or not _same(record[1][0], proposal):
            raise ValueError('unrecognized document action or provider')
        states = _registry(agent, '_document_action_states')
        if states.get(proposal.id) != 'prepared':
            raise ValueError('document action is not prepared')
        retained = record[1]
        result = deepcopy(DocumentActionContext(
            Call(provider.name, retained[6].name, (('target', retained[5]),)),
            retained[4], retained[0].target))
        _validate(agent, provider, retained)
        if states.get(proposal.id) != 'prepared':
            raise ValueError('document action changed during context validation')
        return result
    except Exception as error:
        return Unknown('document_action_context_unavailable', f'{type(error).__name__}: {error}')


def execute_document_action(agent, provider, proposal, *, before_dispatch=None, task_revision=None,
                            final_dispatch_check=None):
    """Consume activation, optionally guarded by a caller and task revision.

    The caller receives retained before-observation source IDs and must return
    exactly True. Its callbacks cannot bypass the subsequent action dependency
    checks. The optional zero-argument final_dispatch_check runs after provider
    validation, allowing model authority checks without another provider call.
    Dependency bases and a supplied (task_id, revision) are checked callback-free
    after that terminal check.
    """
    request = None
    receipt = None
    events = []
    reserved = False
    try:
        if before_dispatch is not None and not callable(before_dispatch):
            raise TypeError('before_dispatch must be callable')
        if final_dispatch_check is not None and not callable(final_dispatch_check):
            raise TypeError('final_dispatch_check must be callable')
        if task_revision is not None and (type(task_revision) is not tuple or len(task_revision) != 2
                or type(task_revision[0]) is not str or type(task_revision[1]) is not int):
            raise TypeError('task_revision requires an exact (task_id, revision) tuple')
        record = _registry(agent, '_document_actions').get(proposal.id)
        if record is None or record[0] is not provider or not _same(record[1][0], proposal):
            raise ValueError('unrecognized document action or provider')
        retained = record[1]
        states = _registry(agent, '_document_action_states')
        if states.get(proposal.id) != 'prepared':
            raise ValueError('document action already consumed or in progress')
        states[proposal.id] = 'executing'
        reserved = True
        _validate(agent, provider, retained)
        request = agent.interpretations.add_source('Explicit document activation request',
            modality='document-action-request', provider=provider.name, payload=deepcopy(proposal),
            metadata={'proposal_source_id': proposal.evidence_source_id})
        expected = deepcopy(request)
        events = []
        def guard(_sources):
            try:
                _validate(agent, provider, retained, request=expected)
                if before_dispatch is not None:
                    result = before_dispatch(tuple(_sources))
                    if result is not True:
                        return result if isinstance(result, Unknown) else Unknown('document_action_guard_declined')
                    _validate(agent, provider, retained, request=expected)
                if final_dispatch_check is not None:
                    result = final_dispatch_check()
                    if result is not True:
                        return result if isinstance(result, Unknown) else Unknown('document_action_final_check_declined')
                # No provider, source-copy, or arbitrary payload callbacks after
                # the terminal model check: only retained scalar authority bases.
                for dependency in retained[4]:
                    if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
                        return Unknown('document_action_dependency_changed')
                if task_revision is not None and agent.tasks.current_revision(task_revision[0]) != task_revision[1]:
                    return Unknown('task_revision_changed')
                return True
            except Exception as error:
                return Unknown('document_action_changed', f'{type(error).__name__}: {error}')
        with use(agent.runtime):
            receipt = agent._invoke(provider, retained[6], {'target': retained[5]}, events,
                                    before_dispatch=guard)
        evidence = agent.interpretations.add_source('Document activation receipt and observations',
            modality='document-action-result', provider=provider.name,
            payload={'receipt': receipt, 'events': tuple(events)},
            metadata={'request_source_id': request.id, 'proposal_source_id': proposal.evidence_source_id})
        states[proposal.id] = 'consumed'
        return DocumentActionResult('document-action-result:' + uuid4().hex, evidence.id, receipt, tuple(events))
    except Exception as error:
        if reserved:
            _registry(agent, '_document_action_states')[proposal.id] = 'failed'
        outcome = Unknown('document_action_changed', f'{type(error).__name__}: {error}')
        agent.interpretations.add_source('Document activation declined', modality='document-action-aborted',
            provider=provider.name, payload={'outcome': outcome, 'receipt': receipt, 'events': tuple(events)},
            metadata={'request_source_id': request.id if request is not None else None})
        return outcome
