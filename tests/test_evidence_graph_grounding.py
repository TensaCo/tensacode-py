"""Generic retained evidence supports learned grounding without pretending it is an image."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

from tensorcode.agent.evidence_graph import EvidenceGraph, GraphProposal
from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.scene_grounding import retain_grounding_example, fit_grounding_model, admit_grounding_model, propose_groundings, grounding_dependencies
from tensorcode.agent.grounding_observation import prepare_grounding_observation, observe_grounding_proposal
from tensorcode.agent.plugin import Plugin
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref, Proposition
from test_scene_grounding_uncertainty import language, PATH


def graph_candidate(owner, name, *, training=False, modality='document'):
    root, a, b = (Ref(name + ':' + suffix) for suffix in ('root', 'a', 'b'))
    facts = [Proposition('opaque-mark', {'entity': a})]
    if training:
        facts.append(Proposition('opaque-mark', {'entity': b}, polarity=False))
    graph = EvidenceGraph(root, (a, b), tuple(facts), ('supplied structured evidence',))
    source = owner.interpretations.add_source('supplied document graph', modality=modality,
        provider='authored fixture', metadata={'root_ref': root.id}, payload=b'original evidence bytes')
    group = owner.interpretations.create_group(source.id)
    candidate = owner.interpretations.propose(group.id, GraphProposal(graph, source.id))
    return group.id, candidate.id, graph, source


def model_fixture():
    owner = SimpleNamespace(interpretations=InterpretationWorkspace())
    examples = []
    for name in ('training-one', 'training-two', 'heldout'):
        lg, lc = language(owner)
        sg, sc, graph, _ = graph_candidate(owner, name, training=True)
        example = retain_grounding_example(owner, lg, lc, PATH, sg, sc,
            (graph.nodes[0],), (graph.nodes[1],), basis=('explicit authored alignments',))
        assert not isinstance(example, Unknown), example
        examples.append(example)
    fitted = fit_grounding_model(owner, examples[:2], examples[2:], max_atoms=1)
    assert not isinstance(fitted, Unknown), fitted
    model = admit_grounding_model(owner, fitted, reason='explicit test admission')
    assert not isinstance(model, Unknown), model
    return owner, model


@pytest.mark.parametrize('modality', ['document', 'environment', 'audio'])
def test_generic_graph_grounding_preserves_source_and_unknowns(modality):
    owner, model = model_fixture()
    lg, lc = language(owner)
    sg, sc, graph, source = graph_candidate(owner, 'novel', modality=modality)
    owner.interpretations.select(sg, sc, reason='explicit graph interpretation')
    report = propose_groundings(owner, model, lg, lc, PATH, sg, sc)
    assert report.complete and len(report.candidate_ids) == 1
    child = next(c for c in owner.interpretations.get(lg).candidates if c.id in report.candidate_ids)
    assert child.payload.acts[0].frame.roles['object'].ref == graph.nodes[0]
    assert len(grounding_dependencies(owner, lg, child.id)) == 2
    assert report.unresolved_candidate_ids
    assert owner.interpretations.get_source(source.id).modality == modality
    assert 'image_ref' not in source.metadata


@pytest.mark.parametrize('answer', [True, False])
def test_generic_observer_has_distinct_hook_and_retains_original_source(answer):
    owner, model = model_fixture()
    lg, lc = language(owner)
    sg, sc, graph, source = graph_candidate(owner, 'novel', modality='environment')
    owner.interpretations.select(sg, sc, reason='explicit graph interpretation')
    proposal = prepare_grounding_observation(owner, model, lg, lc, PATH, sg, sc, graph.nodes[1])
    assert not isinstance(proposal, Unknown), proposal
    class Observer(Plugin):
        def observe_scene_proposition(self, *args):
            pytest.fail('generic evidence must not invoke visual hook')
        def observe_graph_proposition(self, proposition, supplied_graph, supplied_source):
            assert supplied_graph == graph and supplied_source == source
            assert supplied_source.modality == 'environment'
            request = [s for s in owner.interpretations.sources() if s.modality == 'grounding-observation-request'][-1]
            assert request.metadata['evidence_source'] == source
            assert request.payload.positive_evidence and request.payload.negative_evidence
            return answer
    result = observe_grounding_proposal(owner, proposal, proposal.plan.probes[0].id, Observer('generic'))
    assert not isinstance(result, Unknown), result
    group = owner.interpretations.get(sg)
    updated = next(c for c in group.candidates if c.id == result.scene_candidate_id)
    assert type(updated.payload) is GraphProposal and updated.payload.source_id == source.id
    assert updated.payload.graph.propositions == (*graph.propositions, replace(proposal.plan.probes[0].proposition, polarity=answer))
    assert group.selected_id == sc


@pytest.mark.parametrize('mismatch', ['source', 'root'])
def test_generic_teaching_rejects_source_mismatch(mismatch):
    owner = SimpleNamespace(interpretations=InterpretationWorkspace())
    lg, lc = language(owner)
    sg, sc, graph, source = graph_candidate(owner, 'mismatched')
    workspace = owner.interpretations
    if mismatch == 'source':
        bad = workspace.propose(sg, GraphProposal(graph, 'another-retained-source'))
        sc = bad.id
    else:
        workspace._sources[source.id].metadata['root_ref'] = 'wrong-root'
    record = retain_grounding_example(owner, lg, lc, PATH, sg, sc,
        (graph.nodes[0],), basis=('explicit fixture',))
    assert isinstance(record, Unknown)


def test_generic_default_observer_abstains_without_visual_fallback():
    owner, model = model_fixture()
    lg, lc = language(owner)
    sg, sc, graph, _ = graph_candidate(owner, 'novel')
    owner.interpretations.select(sg, sc, reason='explicit graph interpretation')
    proposal = prepare_grounding_observation(owner, model, lg, lc, PATH, sg, sc, graph.nodes[1])
    result = observe_grounding_proposal(owner, proposal, proposal.plan.probes[0].id, Plugin('default'))
    assert result.observation.reason == 'unobserved_graph_proposition'
    assert len(owner.interpretations.get(sg).candidates) == 1


def test_generic_investigation_retains_teacher_evidence_and_refits_without_readmission():
    from tensorcode.agent.grounding_investigation import prepare_grounding_investigation, record_grounding_feedback, refit_grounding_from_feedback
    from tensorcode.agent.scene_grounding import get_grounding_model
    owner, model = model_fixture()
    lg, lc = language(owner)
    sg, sc, graph, source = graph_candidate(owner, 'teacher-observation', training=True)
    proposal = prepare_grounding_investigation(owner, model, lg, lc, PATH, ((sg, sc),))
    assert not isinstance(proposal, Unknown), proposal
    assert proposal.investigation.scenes == (graph,)
    before = owner.interpretations.get_source(proposal.evidence_source_id)
    feedback = record_grounding_feedback(owner, proposal, graph.root, (graph.nodes[0],),
        (graph.nodes[1],), basis=('explicit generic evidence alignment',))
    assert not isinstance(feedback, Unknown), feedback
    assert feedback.teaching_record.scene.source == source
    assert owner.interpretations.get_source(proposal.evidence_source_id) == before
    fitted = refit_grounding_from_feedback(owner, feedback)
    assert not isinstance(fitted, Unknown), fitted
    assert fitted.group_id == model.group_id and fitted.candidate_id != model.candidate_id
    assert isinstance(get_grounding_model(owner, model), Unknown)
    assert isinstance(get_grounding_model(owner, fitted), Unknown)
    assert owner.interpretations.get(model.group_id).selected_id is None


def test_generic_source_change_during_observation_rejects_publication():
    owner, model = model_fixture()
    lg, lc = language(owner)
    sg, sc, graph, source = graph_candidate(owner, 'novel')
    owner.interpretations.select(sg, sc, reason='explicit graph interpretation')
    proposal = prepare_grounding_observation(owner, model, lg, lc, PATH, sg, sc, graph.nodes[1])
    class Observer(Plugin):
        def observe_graph_proposition(self, *args):
            retained = owner.interpretations._sources[source.id]
            owner.interpretations._sources[source.id] = replace(retained, payload=b'changed evidence')
            return True
    result = observe_grounding_proposal(owner, proposal, proposal.plan.probes[0].id, Observer('mutating-source'))
    assert isinstance(result, Unknown)
    assert len(owner.interpretations.get(sg).candidates) == 1
    observations = [s for s in owner.interpretations.sources() if s.modality == 'grounding-observation-result']
    assert len(observations) == 1 and observations[0].payload['raw_result'] is True
    assert observations[0].metadata['authenticated'] is False
