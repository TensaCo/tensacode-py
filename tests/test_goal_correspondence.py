"""Supplied teaching examples test structural learning, not inferred language intent."""
from dataclasses import replace
import pytest
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame
from tensorcode.records import Ref
from tensorcode.learning.goal_correspondence import GoalExample, fit_correspondences


def example(name, *, swap=False, repeated=False, invariant=False, negated=False, features=None):
    a, b = Ref(name + ':a'), Ref(name + ':b')
    if repeated: b = a
    frame = Frame('supplied', {'from': a, 'to': b}, features or {})
    goal = GoalSpec((Condition('related', {'x': b if swap else a, 'y': a if swap else b}, negated),),
                    invariants=(Condition('safe', {'x': a}),) if invariant else (),
                    label='authored teaching label', basis=('explicit teacher',))
    return GoalExample(name, frame, goal, ('supplied example',))


def test_learns_role_correspondence_for_unseen_entities_not_fixed_goal_label():
    model = fit_correspondences([example('a'), example('b')], [example('held')])
    prediction = model.propose(example('fresh').frame)
    assert prediction.complete and len(prediction.proposals) == 1
    proposal = prediction.proposals[0]
    assert proposal.goal.conditions == example('fresh').goal.conditions
    assert proposal.training_example_ids == ('a', 'b') and proposal.validation_example_ids == ('held',)
    assert proposal.goal.label == '' and proposal.goal.basis[0].startswith('learned-ref-correspondence:')


def test_role_swaps_learned_without_authored_mapping_and_competing_outputs_preserved():
    training = [example('a'), example('b'), example('c', swap=True), example('d', swap=True)]
    validation = [example('e'), example('f', swap=True)]
    model = fit_correspondences(training, validation)
    result = model.propose(example('new').frame)
    assert len(result.proposals) == 2
    assert {p.goal.conditions[0].args['x'] for p in result.proposals} == {Ref('new:a'), Ref('new:b')}
    assert all(len(p.validation_example_ids) == len(p.conflicting_validation_example_ids) == 1 for p in result.proposals)


def test_repeated_reference_equality_pattern_cannot_match_distinct_roles():
    model = fit_correspondences([example('a', repeated=True), example('b', repeated=True)],
                                [example('e', repeated=True)])
    assert model.propose(example('fresh', repeated=True).frame).proposals
    assert not model.propose(example('fresh').frame).proposals


def test_negation_invariants_and_all_qualifiers_preserved():
    kwargs = dict(invariant=True, negated=True, features={'mode': ('strict', 1)})
    model = fit_correspondences([example('a', **kwargs), example('b', **kwargs)], [example('e', **kwargs)])
    result = model.propose(example('f', **kwargs).frame).proposals[0].goal
    assert result.conditions == example('f', **kwargs).goal.conditions
    assert result.invariants == example('f', **kwargs).goal.invariants
    assert not model.propose(example('f').frame).proposals
    assert not model.propose(example('f', features={'mode': ('strict', True)}).frame).proposals


def test_rename_invariant_predictions_and_detached_model_examples():
    model = fit_correspondences([example('a'), example('b')], [example('e')])
    renamed = fit_correspondences([example('renamed:a'), example('renamed:b')], [example('renamed:e')])
    assert model.propose(example('f').frame).proposals[0].goal.conditions == renamed.propose(example('f').frame).proposals[0].goal.conditions
    model.training_examples[0].frame.roles.clear()
    assert model.training_examples[0].frame.roles


def test_validation_required_no_unsupported_template_admission():
    model = fit_correspondences([example('a'), example('b')], [example('e', swap=True)])
    assert model.templates and not model.templates[0].validation_ids
    assert not model.propose(example('fresh').frame).proposals
    assert not fit_correspondences([example('a')], [example('e')]).propose(example('f').frame).proposals


def test_split_leakage_and_duplicate_ids_rejected():
    with pytest.raises(ValueError, match='IDs'):
        fit_correspondences([example('a'), example('b')], [example('a')])
    with pytest.raises(ValueError, match='entities'):
        fit_correspondences([example('a'), example('b')], [replace(example('a'), id='different')])


def test_unsupported_shapes_and_pair_budget_are_explicit_incomplete():
    invalid = example('bad', features={'opaque': object()})
    model = fit_correspondences([example('a'), example('b'), invalid], [example('e')])
    assert not model.complete and any('unsupported_node_type' in s for s in model.propose(example('f').frame).unresolved)
    assert not model.propose(invalid.frame).complete
    bounded = fit_correspondences([example('a'), example('b'), example('c')], [example('e')], max_pairs=1)
    assert not bounded.complete and 'pair_budget_exhausted' in bounded.propose(example('f').frame).unresolved


def test_unbound_output_reference_cannot_be_invented_or_constant_generalized():
    a, b = example('a'), example('b')
    goal = GoalSpec((Condition('outside', {'x': Ref('entity:outside')}),))
    model = fit_correspondences([replace(a, goal=goal), replace(b, goal=goal)], [example('e')])
    assert not model.complete and not model.propose(example('fresh').frame).proposals
    same = replace(a, id='same-reference')
    model = fit_correspondences([a, same], [example('e')])
    assert not model.propose(example('fresh').frame).proposals


def contextual(name, room='room:shared'):
    device = Ref('device:' + name)
    context = Ref(room)
    return GoalExample(name, Frame('enable', {'object': device, 'context': context}),
        GoalSpec((Condition('enabled_in', {'device': device, 'room': context}),)))


def test_stable_context_is_retained_constant_with_variable_entity_transfer():
    model = fit_correspondences([contextual('a'), contextual('b')], [contextual('held')])
    proposal = model.propose(contextual('fresh').frame).proposals[0]
    assert proposal.goal.conditions == contextual('fresh').goal.conditions
    assert Ref('room:shared') in model.templates[0].constant_refs
    assert not model.propose(contextual('fresh', 'room:different').frame).proposals
    renamed = fit_correspondences([contextual('x', 'site:renamed'), contextual('y', 'site:renamed')],
                                 [contextual('z', 'site:renamed')])
    assert renamed.propose(contextual('new', 'site:renamed').frame).proposals[0].goal.conditions == contextual('new', 'site:renamed').goal.conditions


def test_shared_context_does_not_exempt_variable_entity_leakage():
    with pytest.raises(ValueError, match='variable entities'):
        fit_correspondences([contextual('a'), contextual('b')], [replace(contextual('a'), id='held')])


def test_unvalidated_matching_rival_retained_beside_validated_proposal():
    model = fit_correspondences([example('a'), example('b'), example('c', swap=True),
                                example('d', swap=True)], [example('held')])
    result = model.propose(example('fresh').frame)
    assert result.complete and len(result.proposals) == 1
    unsupported = next(t for t in model.templates if not t.validation_ids)
    assert unsupported.training_ids == ('c', 'd')
    assert unsupported.conflicting_ids == ('held',)
    assert 'unvalidated_correspondence:' + unsupported.id in result.unresolved
    assert model.unresolved == ()
