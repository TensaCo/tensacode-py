"""Measurement identity and evidence must be supplied independently of prose."""
from datetime import datetime, timezone

import pytest

from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.language import Entity, Frame
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.records import Evidence, Ref
from quantity_fixtures import measurement, calculation


def counted(value, *, ref=None):
    return Entity('description', f'{value} plants', {'count': str(value), 'noun': 'plant'}, ref=ref)


def clause(identity, amount):
    return Frame('has_possession', {
        'subject': Entity('name', 'Alex', ref=identity), 'object': counted(amount),
    })


def test_same_description_subjects_keep_separate_explicit_measurement_identities():
    first, second = Ref('person:first'), Ref('person:second')
    plugin = QuantityPlugin()
    a = measurement(plugin, Ref('measurement:first-person'), first, 'has_possession', Quantity(3, Unit.of('plant')))
    b = measurement(plugin, Ref('measurement:second-person'), second, 'has_possession', Quantity(7, Unit.of('plant')))
    assert a.role('subject') == first and b.role('subject') == second
    for owner, operand, expected in ((first, a, 3), (second, b, 7)):
        chosen = calculation(plugin, 'sum', (operand,), CalculationContext(owner, 'has_possession'))
        result = plugin.calculate(chosen)
        assert not isinstance(result, Unknown), result
        assert result.proposition.role('object') == Quantity(expected, Unit.of('plant'))


@pytest.mark.parametrize('frame', [
    clause(None, 3), clause(Ref('person:bound'), 3),
    Frame('grow', {'subject': counted(3, ref=Ref('collection:plants'))}),
    Frame('grow', {'object': counted(3)}),
])
def test_even_grounded_language_does_not_authorize_measurement_extraction(frame):
    plugin = QuantityPlugin()
    assert not hasattr(plugin, 'observe')
    assert not plugin.mind.propositions()
    assert plugin.capabilities() == ()
    with pytest.raises(TypeError):
        plugin.remember(Ref('person:a'), 'has_possession', frame)
    assert not plugin.mind.propositions()


def test_evidence_and_measurement_reference_are_required_not_generated():
    plugin = QuantityPlugin()
    owner, amount = Ref('person:a'), Quantity(3, Unit.of('plant'))
    evidence = Evidence(Ref('fixture:observer'), datetime.now(timezone.utc), locator='explicit sensor record')
    with pytest.raises(TypeError):
        plugin.remember(owner, 'have', amount, evidence=evidence)
    with pytest.raises(TypeError):
        plugin.remember(owner, 'have', amount, measurement=Ref('measurement:a'))
    with pytest.raises(TypeError):
        plugin.remember(Entity('name', 'Alex'), 'have', amount,
                        measurement=Ref('measurement:a'), evidence=evidence)
    assert not plugin.mind.propositions()
    record = plugin.remember(owner, 'have', amount, measurement=Ref('measurement:a'), evidence=evidence)
    assert record.role('measurement') == Ref('measurement:a')
    assert record.role('subject') == owner
    assert record.role('predicate') == 'have'
    stored, = plugin.mind.propositions()
    assert stored.evidence == [evidence]


def test_counted_kind_is_supplied_separately_from_unit_spelling():
    plugin = QuantityPlugin()
    owner, kind = Ref('collection:plants'), Ref('kind:explicit')
    item = measurement(plugin, Ref('measurement:collection'), owner, 'grow', Quantity(3, Unit.of('plant')), kind=kind)
    assert item.role('kind') == kind
    assert item.role('predicate') == 'grow'
    assert item.role('kind') != Ref('kind:plant')
    assert plugin.capabilities() == ()


def test_quantity_plugin_has_no_description_resolution_or_statement_extraction_overrides():
    for retired in ('refer', 'denote', 'observe', 'total', 'total_of_kind', 'count_properties'):
        assert retired not in QuantityPlugin.__dict__
