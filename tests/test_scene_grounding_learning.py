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
