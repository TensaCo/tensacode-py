"""Executed pure induction on supplied graphs; no answers supplied to ranking."""
from dataclasses import replace
import pytest
from tensorcode.agent.scene import SceneGraph
from tensorcode.learning.scene_grounding import GroundingExample, fit_scene_grounding
from tensorcode.learning.grounding_investigation import investigate_grounding
from tensorcode.records import Proposition, Ref


def scene(name, *, crossed=False, rename=False, whole=False):
    image = Ref('image:' + name)
    target = image if whole else Ref(name + ':a')
    other = Ref(name + ':b')
    nodes = (other,) if whole else (target, other)
    color, size = ('opaque1', 'opaque2') if rename else ('color', 'size')
    role = 'r' if rename else 'entity'
    return SceneGraph(image, nodes, (
        Proposition(color, {role: target, 'value': True}),
        Proposition(size, {role: target, 'value': not crossed}),
        Proposition(color, {role: other, 'value': False}),
        Proposition(size, {role: other, 'value': crossed}),
    ))


def example(name, **kwargs):
    graph = scene(name, **kwargs)
    target = graph.image if kwargs.get('whole') else graph.nodes[0]
    return GroundingExample(name, {'description': 'supplied target'}, graph, (target,), (graph.nodes[-1],))


def fit(*, validation=True, **kwargs):
    return fit_scene_grounding([example('a', **kwargs), example('b', **kwargs)],
                               [example('held', **kwargs)] if validation else [], max_atoms=2)


def test_correlated_teaching_induces_rivals_and_crossed_scenes_discriminate():
    model = fit()
    correlated, crossed = scene('same'), scene('crossed', crossed=True)
    result = investigate_grounding(model, {'description': 'supplied target'}, [correlated, crossed])
    assert result.complete and result.best_scene_ids == (crossed.image,)
    same, cross = result.predictions
    assert not same.discriminating and cross.discriminating
    assert len({p.references for p in cross.predictions}) >= 2
    assert set(p.query_id for p in cross.predictions) == {q.id for q in model.queries}
    assert all(p.validated for p in cross.predictions)


def test_all_best_ties_retained_and_no_discriminating_scene_has_no_winner():
    model = fit()
    graphs = [scene('x', crossed=True), scene('y', crossed=True)]
    result = investigate_grounding(model, {'description': 'supplied target'}, graphs)
    assert set(result.best_scene_ids) == {g.image for g in graphs}
    ordinary = investigate_grounding(model, {'description': 'supplied target'}, [scene('z')])
    assert ordinary.complete and not ordinary.best_scene_ids and 'no_discriminating_scene' in ordinary.unresolved


def test_unvalidated_queries_are_predicted_and_marked_not_silently_removed():
    model = fit(validation=False)
    result = investigate_grounding(model, {'description': 'supplied target'}, [scene('cross', crossed=True)])
    assert result.complete and result.best_scene_ids
    assert result.predictions[0].predictions and all(not p.validated for p in result.predictions[0].predictions)


def test_partial_model_or_any_scene_match_blocks_all_preferences():
    incomplete = fit_scene_grounding([example('a'), example('b')], [example('held')], max_patterns=1)
    result = investigate_grounding(incomplete, {'description': 'supplied target'}, [scene('cross', crossed=True)])
    assert not result.complete and not result.best_scene_ids
    model = fit()
    # Authored resource restriction isolates matcher exhaustion, not predictions.
    model._max_matches = 1
    result = investigate_grounding(model, {'description': 'supplied target'}, [scene('cross', crossed=True)])
    assert not result.complete and not result.best_scene_ids
    assert any(not p.complete for s in result.predictions for p in s.predictions)


def test_opaque_names_and_whole_scene_roots_work_without_semantic_rules():
    model = fit(rename=True, whole=True)
    graph = scene('fresh', crossed=True, rename=True, whole=True)
    result = investigate_grounding(model, {'description': 'supplied target'}, [graph])
    assert result.complete and result.best_scene_ids == (graph.image,)
    assert any(graph.image in p.references for p in result.predictions[0].predictions)


def test_unknown_descriptions_conflicting_scene_identity_and_detached_evidence():
    model = fit()
    graph = scene('fresh', crossed=True)
    result = investigate_grounding(model, {'description': 'unknown'}, [graph])
    assert not result.complete and not result.best_scene_ids
    with pytest.raises(ValueError, match='inconsistent'):
        investigate_grounding(model, {'description': 'supplied target'}, [graph, scene('fresh')])
    result = investigate_grounding(model, {'description': 'supplied target'}, [graph, graph])
    assert len(result.scenes) == 1
    graph.propositions[0].roles.clear()
    assert result.scenes[0].propositions[0].roles


def test_known_description_without_rivals_is_complete_but_not_discriminating():
    model = fit_scene_grounding([example('a')], [example('held')])
    result = investigate_grounding(model, {'description': 'supplied target'}, [scene('new')])
    assert result.complete and not result.best_scene_ids
    assert 'no_supported_query' in result.unresolved and 'no_discriminating_scene' in result.unresolved


def test_conflicted_query_denotations_retain_opposing_evidence_without_recommendation():
    model = fit()
    graph = scene('crossed', crossed=True)
    graph = replace(graph, propositions=(*graph.propositions, replace(graph.propositions[0], polarity=False)))
    result = investigate_grounding(model, {'description': 'supplied target'}, [graph])
    assert not result.complete and not result.best_scene_ids
    rows = result.predictions[0].predictions
    conflicted = [row for row in rows if 'contradictory_match_evidence' in row.unresolved]
    assert conflicted and all(not row.complete for row in conflicted)
    assert any((0, (4,)) in match.conflicts for row in conflicted for match in row.matches)
