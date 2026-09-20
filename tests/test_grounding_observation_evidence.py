"""Authentic retained predictions constrain explicitly chosen read-only observers."""
from dataclasses import replace
import pytest

import tensorcode.agent.grounding_observation as controller
from tensorcode.agent.plugin import Plugin
from tensorcode.outcomes import Unknown
from tensorcode.records import Proposition
from test_scene_grounding_uncertainty import setup, language, scene, PATH


class Observer(Plugin):
    def __init__(self, callback=lambda *args: True):
        super().__init__('explicit-test-observer')
        self.callback, self.calls = callback, []

    def observe_scene_proposition(self, proposition, graph, source):
        self.calls.append((proposition, graph, source))
        return self.callback(proposition, graph, source)


def prepared():
    owner, model = setup()
    lg, lc = language(owner)
    sg, sc, graph = scene(owner, 'observation')
    owner.interpretations.select(sg, sc, reason='explicit fixture selection')
    proposal = controller.prepare_grounding_observation(owner, model, lg, lc, PATH, sg, sc, graph.nodes[1])
    assert not isinstance(proposal, Unknown), proposal
    assert len(proposal.plan.probes) == 1
    return owner, model, lg, lc, sg, sc, graph, proposal


def run(owner, proposal, observer):
    return controller.observe_grounding_proposal(owner, proposal, proposal.plan.probes[0].id, observer)


def sources(owner, modality):
    return [s for s in owner.interpretations.sources() if s.modality == modality]


@pytest.mark.parametrize('answer', [True, False])
def test_request_and_both_predictions_precede_detached_provider_call(answer):
    owner, _, _, _, sg, sc, graph, proposal = prepared()
    asked = proposal.plan.probes[0]
    def callback(proposition, received, source):
        request, = sources(owner, 'grounding-observation-request')
        assert request.payload == asked
        assert request.payload.positive_evidence and request.payload.negative_evidence
        assert owner.interpretations.get_source(proposal.evidence_source_id).payload == proposal.plan
        assert proposition == asked.proposition and received == graph
        assert source.payload == b'authored scene fixture, not learned pixels'
        assert source.metadata['image_ref'] == graph.image.id
        proposition.roles['extra'] = 'mutating detached argument'
        source.metadata['image_ref'] = 'tampered detached argument'
        return answer
    observer = Observer(callback)
    result = run(owner, proposal, observer)
    assert not isinstance(result, Unknown), result
    group = owner.interpretations.get(sg)
    assert group.selected_id == sc and len(group.candidates) == 2
    added = next(c for c in group.candidates if c.id == result.scene_candidate_id)
    assert added.payload.graph.propositions == (*graph.propositions, replace(asked.proposition, polarity=answer))
    assert group.candidates[0].payload.graph == graph
    assert len(observer.calls) == 1
    assert isinstance(run(owner, proposal, observer), Unknown)
    assert len(observer.calls) == 1


@pytest.mark.parametrize('answer', [Unknown('unobserved'), 1, None, 'yes'])
def test_abstention_and_malformed_answers_never_create_scene_facts(answer):
    owner, _, _, _, sg, _, _, proposal = prepared()
    result = run(owner, proposal, Observer(lambda *args: answer))
    assert not isinstance(result, Unknown), result
    assert isinstance(result.observation, Unknown) and result.scene_candidate_id is None
    assert len(owner.interpretations.get(sg).candidates) == 1
    retained = owner.interpretations.get_source(result.evidence_source_id)
    assert retained.payload['raw_result'] == answer and retained.metadata['authenticated']


def test_exception_and_default_plugin_preserve_unknown():
    owner, _, _, _, sg, _, _, proposal = prepared()
    def raises(*args):
        raise RuntimeError('test observer failure')
    result = run(owner, proposal, Observer(raises))
    assert result.observation.reason == 'scene_observation_failed'
    assert owner.interpretations.get_source(result.evidence_source_id).payload['error']['type'] == 'RuntimeError'
    owner, _, _, _, sg, _, _, proposal = prepared()
    result = run(owner, proposal, Plugin('default'))
    assert result.observation.reason == 'unobserved_scene_proposition'
    assert len(owner.interpretations.get(sg).candidates) == 1


@pytest.mark.parametrize('target', ['scene', 'language', 'model'])
def test_stale_comparison_never_invokes_provider(target):
    owner, model, lg, lc, sg, sc, _, proposal = prepared()
    gid, cid = {'scene': (sg, sc), 'language': (lg, lc), 'model': (model.group_id, model.candidate_id)}[target]
    owner.interpretations.select(gid, cid, reason='new explicit comparison')
    observer = Observer()
    assert isinstance(run(owner, proposal, observer), Unknown)
    assert not observer.calls


@pytest.mark.parametrize('target', ['scene', 'request'])
def test_callback_mutation_retains_failed_observation_without_publication(target):
    owner, _, _, _, sg, sc, _, proposal = prepared()
    def mutate(*args):
        if target == 'scene':
            owner.interpretations.select(sg, sc, reason='changed during callback')
        else:
            request, = sources(owner, 'grounding-observation-request')
            owner.interpretations._sources[request.id].metadata['corruption'] = True
        return True
    assert isinstance(run(owner, proposal, Observer(mutate)), Unknown)
    assert len(owner.interpretations.get(sg).candidates) == 1
    observed, = sources(owner, 'grounding-observation-result')
    assert observed.payload['raw_result'] is True and not observed.metadata['authenticated']
    assert sources(owner, 'grounding-observation-aborted')


def test_reentrant_attempt_and_forged_proposal_cannot_consume_canonical_plan():
    owner, _, _, _, _, _, _, proposal = prepared()
    observer = Observer()
    assert isinstance(run(owner, replace(proposal, evidence_source_id='forged'), observer), Unknown)
    assert not observer.calls
    def recurse(*args):
        assert isinstance(run(owner, proposal, observer), Unknown)
        return True
    observer.callback = recurse
    result = run(owner, proposal, observer)
    assert not isinstance(result, Unknown), result
    assert len(observer.calls) == 1


@pytest.mark.parametrize('phase', ['arguments', 'final_cache'])
def test_copy_side_effects_are_checked_before_call_or_return(monkeypatch, phase):
    owner, _, _, _, sg, sc, _, proposal = prepared()
    original = controller.deepcopy
    changed = False
    def copying(value):
        nonlocal changed
        copied = original(value)
        target = (isinstance(value, Proposition) if phase == 'arguments' else
                  isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], controller.GroundingObservationResult))
        if target and not changed:
            changed = True
            owner.interpretations.select(sg, sc, reason='mutation from copy hook')
        return copied
    monkeypatch.setattr(controller, 'deepcopy', copying)
    observer = Observer()
    result = run(owner, proposal, observer)
    assert isinstance(result, Unknown) and changed
    assert len(observer.calls) == (0 if phase == 'arguments' else 1)
    group = owner.interpretations.get(sg)
    if phase == 'final_cache':
        assert len(group.candidates) == 2 and group.candidates[-1].rejected
    assert sources(owner, 'grounding-observation-aborted')


def test_postpublication_cannot_excuse_changed_ancestor_search_evidence(monkeypatch):
    from tensorcode.agent.scene_grounding import propose_scene_groundings
    from tensorcode.agent.understand import Act, SentenceAlternative
    from tensorcode.language import Entity, Frame, Request
    owner, model = setup()
    workspace = owner.interpretations
    source = workspace.add_source('inspect two mentions', provider='authored test')
    group = workspace.create_group(source.id)
    frame = Frame('inspect', {'object': Entity('description', 'the marked thing'),
                              'other': Entity('description', 'the marked thing')})
    language_candidate = workspace.propose(group.id, SentenceAlternative(None, (Act('request', Request(frame), frame),)))
    sg, sc, graph = scene(owner, 'ancestor-observation')
    workspace.select(sg, sc, reason='explicit fixture scene')
    report = propose_scene_groundings(owner, model, group.id, language_candidate.id, PATH, sg, sc)
    child_id, = report.candidate_ids
    proposal = controller.prepare_grounding_observation(owner, model, group.id, child_id,
        ('acts', 0, 'frame', 'roles', 'other'), sg, sc, graph.nodes[1])
    assert not isinstance(proposal, Unknown), proposal
    original = workspace.propose
    def publication(*args, **kwargs):
        candidate = original(*args, **kwargs)
        workspace._sources[report.source_id].metadata['changed_after_scene_publication'] = True
        return candidate
    monkeypatch.setattr(workspace, 'propose', publication)
    observer = Observer()
    result = run(owner, proposal, observer)
    assert isinstance(result, Unknown), result
    assert len(observer.calls) == 1
    assert workspace.get(sg).candidates[-1].rejected
    assert sources(owner, 'grounding-observation-result')
    assert sources(owner, 'grounding-observation-aborted')
