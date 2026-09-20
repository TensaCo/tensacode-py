"""An explicit identity does not license erasing qualifications of a goal value."""

from dataclasses import dataclass

import pytest

from tensorcode.agent import Agent, Capability, Effect, Param, Plugin
from tensorcode.goals import Condition, GoalSpec, normalize_goal_value
from tensorcode.language import Entity, Frame, verbnet
from tensorcode.records import Ref


TARGET = Ref('world:target')


@pytest.mark.parametrize('features', [
    {'count': 3}, {'modifiers': (('amod', 'temporary'),)}, {'polarity': 'negative'},
    {'preserve': True}, {'noun': 'file'}, {'definite': True}, {'arbitrary-new-qualification': False},
])
def test_bound_entity_qualifiers_are_never_silently_discarded(features):
    value = Entity('description', 'target', features, TARGET)
    with pytest.raises(ValueError, match='unconsumed entity features'):
        normalize_goal_value(value, path='request.object')
    with pytest.raises(ValueError, match="conditions\\[0\\].args\\['target'\\]"):
        GoalSpec((Condition('gone', {'target': value}),))
    assert value.features == features


def test_invariant_qualifiers_require_projection_too():
    value = Entity('description', 'notes', {'count': 3}, TARGET)
    with pytest.raises(ValueError, match='invariants'):
        GoalSpec((Condition('ready', {'target': TARGET}),), invariants=(Condition('preserved', {'target': value}),))


def test_reference_only_wrapper_preserves_exact_identity():
    wrapped = Entity('name', 'any description', ref=TARGET)
    assert normalize_goal_value(wrapped) is TARGET
    goal = GoalSpec((Condition('gone', {'target': wrapped}),))
    assert goal.conditions[0].args == {'target': TARGET}


def test_reference_does_not_settle_remaining_entity_alternatives():
    value = Entity('name', 'target', ref=TARGET, candidates=(Entity('name', 'other', ref=Ref('world:other')),))
    with pytest.raises(ValueError, match='unresolved entity alternatives'):
        normalize_goal_value(value)


@pytest.mark.parametrize('kind,value', [('number', 3), ('literal', 'hello')])
def test_explicit_scalar_value_field_is_consumed_as_representation(kind, value):
    scalar = Entity(kind, 'surface spelling', {'value': value})
    assert normalize_goal_value(scalar) == value
    with pytest.raises(ValueError, match='unconsumed entity features'):
        normalize_goal_value(Entity(kind, 'surface spelling', {'value': value, 'unit': 'metre'}))


def test_value_feature_is_not_discarded_when_entity_has_world_identity():
    with pytest.raises(ValueError, match='unconsumed entity features'):
        normalize_goal_value(Entity('literal', 'three', {'value': 3}, ref=TARGET))


def test_nested_qualified_values_are_rejected_with_source_path():
    @dataclass(frozen=True)
    class Payload:
        targets: tuple
    wrapped = Payload((Entity('name', 'target', {'count': 3}, TARGET),))
    with pytest.raises(ValueError, match=r'goal.payload.targets\[0\]'):
        normalize_goal_value(wrapped, path='goal.payload')


def test_lexical_plans_share_the_preservation_boundary():
    class Eraser(Plugin):
        def capabilities(self):
            return (Capability('erase', (Param('target', 'thing'),),
                               effects=(Effect('gone', {'undergoer': 'target'}),)),)
    value = Entity('description', 'three targets', {'count': 3}, TARGET)
    frame = Frame('erase', {'object': value}, {'mood': 'imperative'})
    goal = verbnet.Goal('erase', 'authored:test', (Condition('gone', {'Theme': value}),), frame)
    plans, reasons = Agent([Eraser('eraser')]).plans(goal)
    assert plans == []
    assert any('unconsumed entity features' in reason for reason in reasons)


def test_explicitly_projected_domain_conditions_retain_quantity_and_preservation():
    goal = GoalSpec((Condition('count', {'collection': TARGET, 'value': 3}),),
                    invariants=(Condition('preserved', {'target': Ref('world:notes')}),),
                    basis=('authored-test:explicit-domain-projection',))
    assert goal.conditions[0].args == {'collection': TARGET, 'value': 3}
    assert goal.invariants[0].args == {'target': Ref('world:notes')}
