"""Supplied teaching examples test structural learning, not inferred language intent."""
from dataclasses import replace
import pytest
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame
from tensorcode.records import Ref
from tensorcode.learning.goal_correspondence import (
    GoalCorrespondenceTemplate,
    GoalExample,
    fit_correspondences,
)
from tensorcode.learning.structural_correspondence import StructuralTemplate


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


def test_goal_adapter_uses_shared_templates_without_changing_authenticated_model_shape():
    model = fit_correspondences([example('a'), example('b')], [example('held')])

    assert set(vars(model)) == {
        '_id', '_training', '_validation', '_templates', '_complete', '_unresolved'}
    assert model.templates and all(type(template) is GoalCorrespondenceTemplate
                                   for template in model.templates)
    assert model._templates and all(type(template) is StructuralTemplate
                                    for template in model._templates)
    assert model.propose(example('fresh').frame).proposals[0].goal.conditions == example('fresh').goal.conditions


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


def measured_example(name, *, desired=True, operation='opaque-operation', measurement='opaque-measurement', features=None):
    from tensorcode.goals import MeasuredActionGoal
    target = Ref('object:' + name)
    return GoalExample(name, Frame('supplied-command', {'object': target}, features or {}),
        MeasuredActionGoal(target, operation, measurement, desired, ('explicit teaching',)))


def test_measured_goal_transfers_target_with_taught_literals_only():
    from tensorcode.goals import MeasuredActionGoal
    model = fit_correspondences([measured_example('a'), measured_example('b')], [measured_example('held')])
    fresh = measured_example('fresh')
    result = model.propose(fresh.frame)
    assert result.complete and len(result.proposals) == 1
    goal = result.proposals[0].goal
    assert type(goal) is MeasuredActionGoal
    assert (goal.target, goal.operation, goal.measurement, goal.desired_outcome) == (
        fresh.goal.target, 'opaque-operation', 'opaque-measurement', True)
    assert result.proposals[0].training_example_ids == ('a', 'b')
    assert result.proposals[0].validation_example_ids == ('held',)
    assert not hasattr(goal, 'provider') and not hasattr(goal, 'model') and not hasattr(goal, 'token')
    assert 'opaque-operation' in goal.describe()


def test_measured_goal_preserves_qualifiers_and_competing_typed_outcomes():
    training = [measured_example('a', desired=True), measured_example('b', desired=True),
                measured_example('c', desired=1), measured_example('d', desired=1)]
    validation = [measured_example('e', desired=True), measured_example('f', desired=1)]
    model = fit_correspondences(training, validation)
    result = model.propose(measured_example('fresh').frame)
    assert {(type(p.goal.desired_outcome), p.goal.desired_outcome) for p in result.proposals} == {(bool, True), (int, 1)}
    assert all(p.conflicting_validation_example_ids for p in result.proposals)
    assert not model.propose(measured_example('fresh', features={'negated': True}).frame).proposals
    assert not model.propose(Frame('unseen-command', {'object': Ref('object:fresh')})).proposals


def test_measured_and_condition_goals_keep_distinct_template_tags():
    from tensorcode.goals import MeasuredActionGoal
    train = [measured_example('a'), measured_example('b')]
    held = [measured_example('e')]
    for name, destination in [('c', train), ('d', train), ('f', held)]:
        example = measured_example(name)
        destination.append(replace(example, goal=GoalSpec((Condition('opaque-operation', {'target': example.goal.target}),))))
    model = fit_correspondences(train, held)
    result = model.propose(measured_example('new').frame)
    assert {type(p.goal) for p in result.proposals} == {GoalSpec, MeasuredActionGoal}
    assert {t.goal[0] for t in model.templates} == {'GoalSpec', 'MeasuredActionGoal'}


@pytest.mark.parametrize('desired', [float('nan'), float('inf'), [], {}, object()])
def test_measured_goal_rejects_nonfinite_or_runtime_payloads(desired):
    from tensorcode.goals import MeasuredActionGoal
    with pytest.raises(ValueError, match='finite typed'):
        MeasuredActionGoal(Ref('object:a'), 'operation', 'measurement', desired)


def test_measured_goal_requires_grounded_target_and_nonempty_literal_names():
    from tensorcode.goals import MeasuredActionGoal
    from tensorcode.language import Entity
    for target, operation, measurement in [(Entity('noun', 'thing'), 'op', 'measure'),
                                          (Ref('object:a'), '', 'measure'), (Ref('object:a'), 'op', ' ')]:
        with pytest.raises(ValueError):
            MeasuredActionGoal(target, operation, measurement, True)
    assert MeasuredActionGoal(Ref('object:a'), 'op', 'measure', (None, True, 1, 1.5, 'literal'))


@pytest.mark.parametrize('reverse', [False, True])
def test_singleton_training_rival_remains_unresolved_with_source_identity(reverse):
    training = [example('a'), example('b'), example('rival', swap=True)]
    model = fit_correspondences(training[::-1] if reverse else training, [example('held')])
    result = model.propose(example('fresh').frame)
    assert len(result.proposals) == 1
    assert result.proposals[0].conflicting_training_example_ids == ('rival',)
    assert 'unrepresented_training_rival:rival' in result.unresolved
    assert model.templates[0].conflicting_training_example_ids == ('rival',)


def test_supported_training_rivals_remain_alternatives_without_unresolved_default():
    model = fit_correspondences(
        [example('a'), example('b'), example('c', swap=True), example('d', swap=True)],
        [example('held'), example('held-rival', swap=True)])
    result = model.propose(example('fresh').frame)
    assert len(result.proposals) == 2
    assert all(p.conflicting_training_example_ids for p in result.proposals)
    assert not result.unresolved
