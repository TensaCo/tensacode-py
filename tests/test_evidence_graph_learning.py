"""Document graph fixtures exercise learned relations, not pixel understanding."""
from dataclasses import replace

import pytest

from tensorcode.agent.evidence_graph import EvidenceGraph, GraphProposal, graph_root
from tensorcode.agent.scene import SceneGraph
from tensorcode.learning.graph_evidence import assess_query
from tensorcode.learning.graph_partial import match_partial_query
from tensorcode.learning.graph_queries import query_from_facts
from tensorcode.learning.grounding_probes import propose_grounding_probes
from tensorcode.learning.scene_grounding import GroundingExample, fit_scene_grounding
from tensorcode.records import Proposition, Ref


def example(name):
    target, distractor, section, other = (Ref(name + ':' + key) for key in ('target', 'distractor', 'section', 'other'))
    graph = EvidenceGraph(Ref('document:' + name), (target, distractor, section, other), (
        Proposition('document-parent', {'child': target, 'parent': section}),
        Proposition('document-parent', {'child': distractor, 'parent': other}),
        Proposition('document-attribute', {'node': section, 'value': 'selected'}),
        Proposition('document-attribute', {'node': other, 'value': 'other'}),
    ), ('snapshot may omit other nodes',))
    return GroundingExample(name, {'mention': 'the relevant child'}, graph, (target,),
        (distractor, section, other), ('explicit document alignment',))


def test_document_relational_queries_are_learned_and_ground_fresh_identities():
    learned = fit_scene_grounding((example('train1'), example('train2')), (example('held'),))
    fresh = example('fresh')
    result = learned.propose(fresh.description, fresh.scene)
    assert learned.complete and result.complete
    assert {row.reference for row in result.matches} == set(fresh.positive_refs)
    assert all(len(row.matched_proposition_indices) >= 2 for row in result.matches)
    assert all(row.validation_example_ids == ('held',) for row in result.matches)
    assert graph_root(fresh.scene) == Ref('document:fresh')
    assert not hasattr(fresh.scene, 'image')


def test_document_partial_witness_probes_missing_parent_attribute():
    learned = fit_scene_grounding((example('train1'), example('train2')), (example('held'),))
    fresh = example('new')
    incomplete = replace(fresh.scene, propositions=tuple(p for i, p in enumerate(fresh.scene.propositions) if i != 2))
    result = learned.propose(fresh.description, incomplete)
    assert not result.matches
    plan = propose_grounding_probes(learned, fresh.description, incomplete, fresh.positive_refs[0], max_states=200000)
    assert plan.complete and plan.probes
    assert plan.scene_id == incomplete.root
    wanted = fresh.scene.propositions[2]
    probe = next(p for p in plan.probes if p.proposition == wanted)
    assert any(w.supporting and dict(w.bindings).get(0) == fresh.positive_refs[0] for w in probe.witnesses)
    assert any(any(row.reference == fresh.positive_refs[0] and row.status == 'supported' for row in evidence.roots)
        for _, evidence in probe.positive_evidence)
    assert all(evidence.unseen_referents_possible for _, evidence in probe.negative_evidence)
    assert incomplete.propositions != fresh.scene.propositions


def test_document_root_and_visual_root_share_algorithms_without_relabeling():
    root = Ref('document:root')
    fact = Proposition('contains-information', {'source': root})
    document = EvidenceGraph(root, propositions=(fact,))
    query = query_from_facts((fact,), root)
    assessment = assess_query(query, document)
    assert assessment.complete and assessment.roots[0].status == 'supported'
    assert assessment.roots[0].reference == root
    partial = match_partial_query(query, document, root)
    assert partial.complete and len(partial.matches) == 2
    visual = SceneGraph(Ref('image:visual'))
    assert graph_root(visual) == visual.image
    with pytest.raises(TypeError, match='expected'):
        graph_root(type('PretendGraph', (), {'root': root})())


def test_graph_proposal_retains_explicit_source_id_and_validates_shape():
    graph = example('proposal').scene
    proposal = GraphProposal(graph, 'source:retained', ('document extraction',))
    proposal.validate()
    assert proposal.source_id == 'source:retained'
    for source_id in ('', '   ', None, 1):
        with pytest.raises(ValueError):
            GraphProposal(graph, source_id)
    with pytest.raises(TypeError):
        GraphProposal(SceneGraph(Ref('image:wrong')), 'source:retained')


def test_neutral_graph_rejects_undeclared_nested_and_scope_references_and_bad_types():
    root, node, missing = Ref('document:root'), Ref('node:one'), Ref('node:absent')
    for fact in (Proposition('nested', {'node': node, 'payload': {'items': [missing]}}),
                 Proposition('scoped', {'node': node}, scope=missing)):
        with pytest.raises(ValueError, match='undeclared'):
            EvidenceGraph(root, (node,), (fact,))
    with pytest.raises(TypeError):
        EvidenceGraph('document:root')
    with pytest.raises(TypeError):
        EvidenceGraph(root, (node,), ('not a proposition',))
    with pytest.raises(ValueError):
        EvidenceGraph(root, (root,))
    with pytest.raises(TypeError):
        EvidenceGraph(root, limitations=(False,))


def test_document_training_and_validation_reference_leakage_is_rejected():
    first, second = example('first'), example('second')
    with pytest.raises(ValueError, match='disjoint'):
        fit_scene_grounding((first, second), (replace(first, id='held'),))


def test_document_investigation_compares_learned_query_rivals_and_retains_evidence():
    from tensorcode.learning.grounding_investigation import investigate_grounding

    def correlated(name, crossed=False):
        a, b = Ref(name + ':a'), Ref(name + ':b')
        graph = EvidenceGraph(Ref('document:' + name), (a, b), (
            Proposition('field-x', {'node': a, 'value': True}),
            Proposition('field-y', {'node': a, 'value': not crossed}),
            Proposition('field-x', {'node': b, 'value': False}),
            Proposition('field-y', {'node': b, 'value': crossed}),
        ))
        return GroundingExample(name, {'mention': 'target'}, graph, (a,), (b,))

    learned = fit_scene_grounding((correlated('train1'), correlated('train2')), (correlated('held'),), max_atoms=2)
    same, crossed = correlated('same'), correlated('crossed', True)
    report = investigate_grounding(learned, same.description, (same.scene, crossed.scene))
    assert report.complete and report.best_scene_ids == (crossed.scene.root,)
    assert all(type(graph) is EvidenceGraph for graph in report.scenes)
    assert all(row.evidence.unseen_referents_possible for prediction in report.predictions for row in prediction.predictions)
