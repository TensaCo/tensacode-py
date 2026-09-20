"""Authored graphs/alignments isolate induction; no image understanding claim."""
from dataclasses import replace
import pytest
from tensorcode.agent.scene import SceneGraph
from tensorcode.records import Proposition, Ref
from tensorcode.learning.scene_grounding import GroundingExample, fit_scene_grounding


def fixture(name, *, renamed=False, second=False, reverse=False, marker=True):
    nodes = tuple(Ref(name + ':' + n) for n in ('target', 'distractor', 'anchor', 'other'))
    t, d, a, other = nodes
    rel, attr = ('opaque_relation', 'opaque_attribute') if renamed else ('near', 'marked')
    subj, obj = ('q', 'z') if renamed else ('subject', 'object')
    facts = [Proposition(rel, {subj: a if reverse else t, obj: t if reverse else a}),
             Proposition(rel, {subj: d, obj: other}),
             Proposition(attr, {subj: a, 'value': marker}),
             Proposition(attr, {subj: other, 'value': False})]
    if second: facts.append(Proposition(rel, {subj: d, obj: a}))
    scene = SceneGraph(Ref('image:' + name), nodes, tuple(facts))
    return GroundingExample(name, {'utterance': 'the relational target'}, scene, (t,), (d, a, other), ('explicit alignment',))


def fitted(**kwargs):
    return fit_scene_grounding([fixture('train1', **kwargs), fixture('train2', **kwargs)], [fixture('held', **kwargs)])


def targets(result): return {match.reference for match in result.matches}


def test_induced_conjunction_identifies_new_scene_target_with_full_match_evidence():
    model = fitted()
    fresh = fixture('new')
    result = model.propose(fresh.description, fresh.scene)
    assert model.complete and result.complete
    assert targets(result) == set(fresh.positive_refs)
    assert result.matches and all(len(match.matched_proposition_indices) >= 2 for match in result.matches)
    assert all(match.assignments[0] == match.reference for match in result.matches)
    assert all(match.training_example_ids == ('train1', 'train2') and match.validation_example_ids == ('held',) for match in result.matches)


def test_relation_direction_change_refuses_old_binding_and_second_target_retained():
    model = fitted()
    reversed_scene = fixture('reverse', reverse=True)
    assert reversed_scene.positive_refs[0] not in targets(model.propose(reversed_scene.description, reversed_scene.scene))
    doubled = fixture('double', second=True)
    result = model.propose(doubled.description, doubled.scene)
    assert targets(result) == {doubled.scene.nodes[0], doubled.scene.nodes[1]}


def test_consistent_opaque_predicate_role_entity_renaming_still_learns():
    model = fitted(renamed=True)
    fresh = fixture('renamedfresh', renamed=True)
    assert targets(model.propose(fresh.description, fresh.scene)) == set(fresh.positive_refs)


def test_typed_values_do_not_match_numeric_alias_and_metadata_is_not_dropped():
    model = fitted()
    fresh = fixture('numeric', marker=1)
    assert not model.propose(fresh.description, fresh.scene).matches
    fresh = fixture('modal')
    scene = replace(fresh.scene, propositions=tuple(replace(p, modality='possible') for p in fresh.scene.propositions))
    assert not model.propose(fresh.description, scene).matches


def test_no_implicit_negatives_and_unknown_description_remains_unresolved():
    train = [replace(fixture(n), negative_refs=()) for n in ('a', 'b')]
    validation = [replace(fixture('c'), negative_refs=())]
    model = fit_scene_grounding(train, validation)
    fresh = fixture('d')
    # Broad single-relation patterns stay compatible without negative teaching.
    assert fresh.scene.nodes[1] in targets(model.propose(fresh.description, fresh.scene))
    result = model.propose({'utterance': 'unseen description'}, fresh.scene)
    assert not result.matches and result.unresolved


def test_budget_exhaustion_and_missing_validation_remain_visible():
    bounded = fit_scene_grounding([fixture('a'), fixture('b')], [fixture('c')], max_patterns=1)
    assert not bounded.complete and bounded.unresolved
    model = fit_scene_grounding([fixture('a'), fixture('b')], [])
    fresh = fixture('d')
    result = model.propose(fresh.description, fresh.scene)
    assert not result.matches and any(x.startswith('unvalidated_query:') for x in result.unresolved)


def test_heldout_scene_entity_leakage_and_explicit_label_conflict_rejected():
    with pytest.raises(ValueError, match='disjoint'):
        fit_scene_grounding([fixture('a'), fixture('b')], [replace(fixture('a'), id='renamed')])
    example = fixture('a')
    with pytest.raises(ValueError, match='conflicting_alignment'):
        fit_scene_grounding([replace(example, negative_refs=example.positive_refs)], [])


def test_changed_heldout_scene_preserves_unvalidated_query_instead_of_false_support():
    model = fit_scene_grounding([fixture('a'), fixture('b')], [fixture('held', reverse=True)])
    assert model.queries and all(not q.validation_example_ids for q in model.queries)
    fresh = fixture('new')
    result = model.propose(fresh.description, fresh.scene)
    assert not result.matches and any(x.startswith('unvalidated_query:') for x in result.unresolved)


def test_conflicting_validation_cannot_borrow_other_positive_validation_support():
    model = fit_scene_grounding([fixture('a'), fixture('b')], [fixture('good'), fixture('bad', reverse=True)])
    assert any(query.validation_example_ids and query.conflicting_validation_example_ids for query in model.queries)
    fresh = fixture('new')
    result = model.propose(fresh.description, fresh.scene)
    assert not result.matches and any(x.startswith('unvalidated_query:') for x in result.unresolved)


def test_model_evidence_snapshots_detached_and_unsupported_description_explicit():
    model = fitted()
    model.training_examples[0].scene.propositions[0].roles.clear()
    assert model.training_examples[0].scene.propositions[0].roles
    fresh = fixture('fresh')
    result = model.propose(object(), fresh.scene)
    assert not result.complete and not result.matches and result.unresolved


def whole_scene(name):
    image = Ref('image:' + name)
    group, distractor, other = (Ref(name + ':' + n) for n in ('group', 'distractor', 'other'))
    graph = SceneGraph(image, (group, distractor, other), (
        Proposition('organization', {'whole': image, 'component': group}),
        Proposition('organization', {'whole': distractor, 'component': other}),
        Proposition('coherence', {'group': group, 'value': True}),
        Proposition('coherence', {'group': other, 'value': False}),
    ))
    return GroundingExample(name, {'description': 'the coherent situation'}, graph,
                            (image,), (group, distractor, other))


def test_whole_scene_referent_learned_from_relational_organization():
    model = fit_scene_grounding([whole_scene('a'), whole_scene('b')], [whole_scene('held')])
    fresh = whole_scene('new')
    result = model.propose(fresh.description, fresh.scene)
    assert result.complete and targets(result) == {fresh.scene.image}
    assert all(len(match.matched_proposition_indices) >= 2 for match in result.matches)


def test_same_scene_identity_cannot_supply_inconsistent_graphs_or_independent_support():
    first = fixture('a')
    changed = replace(fixture('a', reverse=True), id='different-example')
    with pytest.raises(ValueError, match='inconsistent'):
        fit_scene_grounding([first, changed], [fixture('held')])
    repeated = replace(first, id='second-annotation')
    model = fit_scene_grounding([first, repeated], [fixture('held')])
    assert not model.queries


def test_negative_only_training_constrains_queries_without_positive_corroboration():
    from test_grounding_investigation import example as correlated, scene as crossed
    description = {'description': 'supplied target'}
    negative_graph = crossed('negative', crossed=True)
    # Reject the color-positive node. The size-positive rival stays consistent.
    constraint = GroundingExample('negative', description, negative_graph, (), (negative_graph.nodes[0],))
    base = fit_scene_grounding([correlated('a'), correlated('b')], [correlated('held')], max_atoms=2)
    constrained = fit_scene_grounding([correlated('a'), correlated('b'), constraint], [correlated('held')], max_atoms=2)
    assert len(constrained.queries) < len(base.queries)
    assert constrained.queries and all('negative' in query.training_example_ids for query in constrained.queries)
    assert not fit_scene_grounding([correlated('a'), constraint], [correlated('held')], max_atoms=2).queries


def test_negative_only_validation_is_constraint_not_positive_validation():
    held = fixture('held')
    negative_only = replace(held, positive_refs=())
    model = fit_scene_grounding([fixture('a'), fixture('b')], [negative_only])
    assert model.queries and all(not query.validation_example_ids for query in model.queries)
    fresh = fixture('fresh')
    assert not model.propose(fresh.description, fresh.scene).matches
    combined = fit_scene_grounding([fixture('a'), fixture('b')], [fixture('positive'), negative_only])
    assert combined.propose(fresh.description, fresh.scene).matches
    assert all(query.validation_example_ids == ('positive',) for query in combined.queries)
    contradicted = replace(fixture('negative'), positive_refs=(), negative_refs=fixture('negative').positive_refs)
    denied = fit_scene_grounding([fixture('a'), fixture('b')], [fixture('positive'), contradicted])
    assert not denied.propose(fresh.description, fresh.scene).matches


def test_examples_without_any_alignment_rejected():
    empty = replace(fixture('a'), positive_refs=(), negative_refs=())
    with pytest.raises(ValueError, match='at_least_one_alignment'):
        fit_scene_grounding([empty], [])


def test_complete_zero_match_rival_remains_unresolved_beside_nonempty_queries():
    from test_grounding_investigation import fit, scene
    model = fit()
    result = model.propose({'description': 'supplied target'}, scene('crossed', crossed=True))
    assert result.complete and result.matches
    assert any(reason.startswith('query_predicts_no_referent:') for reason in result.unresolved)


def contradicted(example):
    opposing = replace(example.scene.propositions[0], polarity=False)
    return replace(example, scene=replace(example.scene, propositions=(*example.scene.propositions, opposing)))


def test_contradictory_training_support_prevents_model_admission():
    model = fit_scene_grounding([contradicted(fixture('a')), fixture('b')], [fixture('held')])
    assert not model.complete
    assert any('contradictory_match_evidence' in reason for reason in model.unresolved)


def test_contradictory_novel_scene_retains_diagnostic_matches_and_opposing_indices():
    model = fitted()
    fresh = contradicted(fixture('fresh'))
    result = model.propose(fresh.description, fresh.scene)
    assert result.matches and not result.complete
    assert any('contradictory_match_evidence' in reason for reason in result.unresolved)
    assert any((0, (4,)) in match.conflicts for match in result.matches)


def test_contradictory_validation_cannot_certify_positive_support():
    model = fit_scene_grounding([fixture('a'), fixture('b')], [contradicted(fixture('held'))])
    assert not model.complete and model.unresolved
    assert all(not query.validation_example_ids for query in model.queries)


def unary_example(name):
    target, other = Ref(name + ':target'), Ref(name + ':other')
    graph = SceneGraph(Ref('image:' + name), (target, other), (
        Proposition('opaque', {'entity': target}),
        Proposition('opaque', {'entity': other}, polarity=False),
    ))
    return GroundingExample(name, 'supplied unary description', graph, (target,), (other,))


def test_per_root_missing_support_is_unknown_and_explicit_opposition_refutes():
    model = fit_scene_grounding([unary_example('a'), unary_example('b')], [unary_example('held')], max_atoms=1)
    fresh = unary_example('new')
    partial = replace(fresh.scene, propositions=fresh.scene.propositions[:1])
    result = model.propose(fresh.description, partial)
    assert result.complete and targets(result) == {fresh.scene.nodes[0]}
    assert result.query_evidence
    evidence = result.query_evidence[0][1]
    statuses = {row.reference: row.status for row in evidence.roots}
    assert statuses[fresh.scene.nodes[0]] == 'supported'
    assert statuses[fresh.scene.nodes[1]] == 'unknown'
    assert evidence.unseen_referents_possible
    explicit = model.propose(fresh.description, fresh.scene).query_evidence[0][1]
    rejected = next(row for row in explicit.roots if row.reference == fresh.scene.nodes[1])
    assert rejected.status == 'refuted' and rejected.refuting_atoms


def test_missing_relational_witness_retains_unknown_root_and_budget_incomplete():
    model = fitted()
    fresh = fixture('fresh')
    partial = replace(fresh.scene, propositions=fresh.scene.propositions[:2])
    result = model.propose(fresh.description, partial)
    assert result.complete and not result.matches
    assert all(next(row for row in evidence.roots if row.reference == fresh.scene.nodes[0]).status == 'unknown'
               for _, evidence in result.query_evidence)
    model._max_matches = 1
    bounded = model.propose(fresh.description, fresh.scene)
    assert not bounded.complete and any(not evidence.complete for _, evidence in bounded.query_evidence)
