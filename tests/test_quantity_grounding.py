"""Quantity attachment requires occurrence-local, explicitly supplied identities."""

import pytest

from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.language import Entity, Frame
from tensorcode.quantity import Quantity, Unit
from tensorcode.records import Ref


def counted(value, *, ref=None):
    return Entity('description', f'{value} plants', {'count': str(value), 'noun': 'plant'}, ref=ref)


def clause(identity, amount):
    return Frame('has_possession', {
        'subject': Entity('name', 'Alex', ref=identity), 'object': counted(amount),
    })


def test_same_description_subjects_keep_separate_quantities():
    first, second = Ref('person:first'), Ref('person:second')
    frame = Frame('and', {'clauses': (clause(first, 3), clause(second, 7))})
    plugin = QuantityPlugin()
    claims = plugin.observe(frame)
    assert [(c.subject, c.object.value) for c in claims] == [(first, 3), (second, 7)]
    assert plugin.total(first, 'has_possession') == Quantity(3, Unit.of('plant'))
    assert plugin.total(second, 'has_possession') == Quantity(7, Unit.of('plant'))


@pytest.mark.parametrize('frame', [
    clause(None, 3),
    Frame('grow', {'subject': counted(3)}),
    Frame('grow', {'object': counted(3)}),
])
def test_unbound_or_subjectless_quantity_asserts_nothing(frame):
    plugin = QuantityPlugin()
    assert plugin.observe(frame) == []
    assert not plugin.mind.claims()


def test_counted_subject_uses_its_explicit_collection_identity():
    identity = Ref('collection:plants')
    plugin = QuantityPlugin()
    [claim] = plugin.observe(Frame('grow', {'subject': counted(3, ref=identity)}))
    assert claim.subject == identity
    assert claim.object == Quantity(3, Unit.of('plant'))


def test_bound_subject_does_not_ground_equal_wording_in_another_clause():
    identity = Ref('person:known')
    plugin = QuantityPlugin()
    claims = plugin.observe(Frame('and', {'clauses': (clause(identity, 3), clause(None, 7))}))
    assert len(claims) == 1
    assert plugin.total(identity, 'has_possession') == Quantity(3, Unit.of('plant'))


def test_direct_subject_reference_preserves_owner_without_surface_text():
    identity = Ref('person:direct')
    plugin = QuantityPlugin()
    [claim] = plugin.observe(Frame('has_possession', {'subject': identity, 'object': counted(4)}))
    assert claim.subject == identity
    assert claim.object == Quantity(4, Unit.of('plant'))


def test_quantity_plugin_has_no_description_resolution_overrides():
    assert 'refer' not in QuantityPlugin.__dict__
    assert 'denote' not in QuantityPlugin.__dict__
