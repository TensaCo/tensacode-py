"""Explicit transport fixtures isolate document action authority and replay guards."""
from copy import deepcopy
from dataclasses import dataclass, replace
import pytest

from tensorcode.agent import Agent, Plugin
from tensorcode.agent.plugin import Capability, Param
from tensorcode.agent.document_actions import capture_browser_document, prepare_document_action, execute_document_action
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.language import Entity, Frame, Request
from tensorcode.outcomes import Unknown, Receipt

PATH = ('acts', 0, 'frame', 'roles', 'object')


@dataclass(frozen=True)
class Capture:
    id: str
    snapshot: dict


class Provider(Plugin):
    def __init__(self):
        super().__init__('explicit-document-provider')
        self.capture = Capture('capture:test', {'strings': ['#document', 'BUTTON', ''], 'documents': [
            {'nodes': {'parentIndex': [-1, 0], 'nodeType': [9, 1], 'nodeName': [0, 1],
                       'nodeValue': [2, 2], 'attributes': [[], []]}}]})
        self.calls = []
        self.token = None
        self.before_observation = lambda: None
        self.before_execution = lambda: None
        self.abstain = False
    def capture_document(self):
        return deepcopy(self.capture)
    def authenticate_document_capture(self, capture):
        return True if capture == self.capture else Unknown('forged_capture')
    def prepare_document_target(self, capture, document_index, node_index):
        if self.abstain:
            return Unknown('unavailable_target')
        assert (document_index, node_index) == (0, 1)
        self.token = 'opaque:test:token'
        return self.token
    def validate_document_target(self, token):
        return True if token == self.token and self.token is not None else Unknown('invalid_target')
    def capabilities(self):
        return (Capability('activate_node', (Param('target', 'opaque'),)),)
    def observe_evidence(self):
        self.before_observation()
        return {'document_snapshot': deepcopy(self.capture.snapshot)}
    def execute(self, act, *, key=None):
        self.before_execution()
        assert self.validate_document_target(act.arg('target')) is True
        self.calls.append(act)
        self.token = None
        return Receipt(act, 'applied')


def prepared(*, select=True):
    provider = Provider()
    agent = Agent([provider])
    document = capture_browser_document(agent, provider)
    assert not isinstance(document, Unknown), document
    group = agent.interpretations.get(document.group_id)
    target = group.candidates[0].payload.graph.nodes[1]
    source = agent.interpretations.add_source('supplied bound reading', provider='authored fixture')
    language = agent.interpretations.create_group(source.id)
    frame = Frame('explicit-activation', {'object': Entity('description', 'authored target', ref=target)})
    candidate = agent.interpretations.propose(language.id, SentenceAlternative(None, (Act('request', Request(frame), frame),)))
    if select:
        agent.interpretations.select(language.id, candidate.id, reason='explicit supplied reading')
        agent.interpretations.select(document.group_id, document.candidate_id, reason='explicit supplied graph')
    return agent, provider, document, language.id, candidate.id


def plan(context):
    agent, provider, document, gid, cid = context
    return prepare_document_action(agent, provider, gid, cid, PATH, document.group_id, document.candidate_id)


def test_selected_exact_node_activates_once_and_retains_receipt():
    context = prepared()
    agent, provider, document, _, _ = context
    proposal = plan(context)
    assert not isinstance(proposal, Unknown), proposal
    result = execute_document_action(agent, provider, proposal)
    assert not isinstance(result, Unknown), result
    assert result.receipt.status == 'applied' and len(provider.calls) == 1
    assert any(event['type'] == 'dispatch_guard' and event['allowed'] for event in result.events)
    assert agent.interpretations.get_source(result.evidence_source_id).payload['receipt'] == result.receipt
    assert isinstance(execute_document_action(agent, provider, proposal), Unknown)
    assert len(provider.calls) == 1 and not agent.store.propositions()


@pytest.mark.parametrize('failure', ['unselected', 'source', 'provider', 'abstention'])
def test_prepare_refuses_missing_authority(failure):
    context = prepared(select=failure != 'unselected')
    agent, provider, document, gid, cid = context
    if failure == 'source':
        agent.interpretations._sources[document.source_id].payload['strings'][1] = 'CHANGED'
    if failure == 'provider':
        context = agent, Provider(), document, gid, cid
    if failure == 'abstention':
        provider.abstain = True
    assert isinstance(plan(context), Unknown)
    assert not provider.calls


@pytest.mark.parametrize('failure', ['reading', 'document', 'source'])
def test_preaction_observation_changes_cannot_authorize_dispatch(failure):
    context = prepared()
    agent, provider, document, gid, cid = context
    proposal = plan(context)
    def mutation():
        if failure == 'reading':
            agent.interpretations.unset(gid, reason='withdrawn reading')
        elif failure == 'document':
            agent.interpretations.unset(document.group_id, reason='withdrawn document')
        else:
            agent.interpretations._sources[document.source_id].payload['strings'][1] = 'CHANGED'
    provider.before_observation = mutation
    result = execute_document_action(agent, provider, proposal)
    assert result.receipt.status == 'rejected' and not provider.calls
    assert any(event['type'] == 'dispatch_guard' and not event['allowed'] for event in result.events)


def test_forged_and_reentrant_attempts_do_not_dispatch_twice():
    context = prepared()
    agent, provider, _, _, _ = context
    proposal = plan(context)
    forged = replace(proposal, evidence_source_id='invented')
    assert isinstance(execute_document_action(agent, provider, forged), Unknown)
    assert isinstance(execute_document_action(agent, Provider(), proposal), Unknown)
    def recurse():
        assert isinstance(execute_document_action(agent, provider, proposal), Unknown)
    provider.before_execution = recurse
    result = execute_document_action(agent, provider, proposal)
    assert result.receipt.status == 'applied' and len(provider.calls) == 1


def test_capability_callback_cannot_withdraw_supporting_model_dependency(monkeypatch):
    import tensorcode.agent.document_actions as bridge
    from tensorcode.agent.task_dependencies import capture_dependency
    context = prepared()
    agent, provider, _, _, _ = context
    workspace = agent.interpretations
    source = workspace.add_source('authored model dependency fixture')
    group = workspace.create_group(source.id)
    candidate = workspace.propose(group.id, {'explicit': 'supporting model fixture'})
    workspace.select(group.id, candidate.id, reason='explicit fixture model admission')
    dependency = capture_dependency(workspace, group.id, basis=('authored dependency isolation',))
    monkeypatch.setattr(bridge, 'grounding_dependencies', lambda *args: (dependency,))
    proposal = plan(context)
    assert not isinstance(proposal, Unknown), proposal
    capabilities = provider.capabilities
    def changed_capabilities():
        workspace.unset(group.id, reason='withdrawn during capabilities callback')
        return capabilities()
    monkeypatch.setattr(provider, 'capabilities', changed_capabilities)
    assert isinstance(execute_document_action(agent, provider, proposal), Unknown)
    assert not provider.calls


def test_target_token_must_not_execute_user_stringification_hooks(monkeypatch):
    context = prepared()
    agent, provider, _, _, _ = context
    class MalformedToken:
        def __str__(self):
            pytest.fail('token stringification must not run')
    monkeypatch.setattr(provider, 'prepare_document_target', lambda *args: MalformedToken())
    assert isinstance(plan(context), Unknown)
    assert not provider.calls


def test_postaction_authority_change_does_not_erase_performed_receipt():
    context = prepared()
    agent, provider, document, _, _ = context
    proposal = plan(context)
    observations = 0
    def observe():
        nonlocal observations
        observations += 1
        if observations > 1:
            agent.interpretations.unset(document.group_id, reason='post-action observation changed interpretation')
    provider.before_observation = observe
    result = execute_document_action(agent, provider, proposal)
    assert result.receipt.status == 'applied' and len(provider.calls) == 1
    evidence = agent.interpretations.get_source(result.evidence_source_id)
    assert evidence.payload['receipt'].status == 'applied'
    assert any(event['type'] == 'receipt' for event in evidence.payload['events'])
