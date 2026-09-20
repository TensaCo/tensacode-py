"""Chosen evidence, not unit spelling, supplies a direct conversion edge."""
from datetime import datetime, timezone
from dataclasses import replace

import pytest

from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.derivations import export_derivation, import_derivation, validate_record_support
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.records import Evidence, Interval, Proposition, Ref, Store

OWNER = Ref('owner:explicit')
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 2, 1, tzinfo=timezone.utc)
MIDDLE = datetime(2026, 1, 15, tzinfo=timezone.utc)


def evidence(name):
    return Evidence(Ref('source:' + name), START, locator='explicit supplied conversion evidence')


def setup(*, scope=None, valid=Interval()):
    plugin = QuantityPlugin()
    measurement = plugin.remember(OWNER, 'held', Quantity(2, Unit.of('opaque-pack')),
        measurement=Ref('measurement:original'), evidence=evidence('measurement'), scope=scope, valid=valid)
    definition = plugin.remember_conversion(Ref('definition:chosen'), Unit.of('opaque-pack'), Unit.of('opaque-item'), 4,
        scope=scope, valid=valid, evidence=evidence('definition'))
    return plugin, measurement, definition


def calculate(plugin, measurement, definition, *, scope=None, valid=Interval()):
    ref = plugin.register_calculation('convert', (measurement.id, definition.id),
        context=CalculationContext(OWNER, 'held'), params={'scope': scope, 'valid': valid},
        basis=('explicit direct edge and measurement selection',))
    plugin.select_calculation(ref, reason='explicit test selection')
    return plugin.calculate(ref)


def test_explicit_direct_edge_preserves_original_definition_and_authenticates_output():
    plugin, measurement, definition = setup()
    result = calculate(plugin, measurement, definition)
    assert result.proposition.role('object') == Quantity(8, Unit.of('opaque-item'))
    assert set(result.premise_ids) == {measurement.id, definition.id}
    assert plugin.mind.propositions('quantity_conversion')[0].proposition == definition
    world = Store()
    imported = import_derivation(world, export_derivation(plugin.mind, result))
    assert validate_record_support(world, imported.id) is True
    plugin.mind.supersede(definition, 'definition withdrawn')
    assert isinstance(validate_record_support(world, imported.id), Unknown)
    assert isinstance(calculate(plugin, measurement, definition), Unknown)


def test_selected_scope_and_application_interval_are_exact_and_retained():
    scope = Ref('scope:supplied')
    plugin, measurement, definition = setup(scope=scope, valid=Interval(START, END))
    application = Interval.at(MIDDLE)
    result = calculate(plugin, measurement, definition, scope=scope, valid=application)
    assert result.proposition.scope == scope and result.proposition.valid == application
    cap, = plugin.capabilities()
    assert cap.informs[0].query.scope == scope and cap.informs[0].query.valid == application
    assert isinstance(calculate(plugin, measurement, definition, scope=None, valid=application), Unknown)
    assert isinstance(calculate(plugin, measurement, definition, scope=scope, valid=Interval()), Unknown)
    assert isinstance(calculate(plugin, measurement, definition, scope=scope,
        valid=Interval.at(datetime(2027, 1, 1, tzinfo=timezone.utc))), Unknown)


@pytest.mark.parametrize('same_identity', [True, False])
def test_competing_rate_cannot_be_hidden_by_choosing_one_record(same_identity):
    plugin, measurement, definition = setup()
    first = calculate(plugin, measurement, definition)
    plugin.remember_conversion(Ref('definition:chosen' if same_identity else 'definition:rival'),
        Unit.of('opaque-pack'), Unit.of('opaque-item'), 5,
        scope=None, valid=Interval(), evidence=evidence('rival'))
    assert isinstance(validate_record_support(plugin.mind, first.record_id), Unknown)
    assert isinstance(calculate(plugin, measurement, definition), Unknown)


def test_revised_nonoverlapping_definition_is_not_a_competing_current_rate():
    plugin, measurement, definition = setup(valid=Interval(START, MIDDLE))
    plugin.remember_conversion(Ref('definition:later'), Unit.of('opaque-pack'), Unit.of('opaque-item'), 5,
        scope=None, valid=Interval(datetime(2026, 1, 16, tzinfo=timezone.utc), END), evidence=evidence('later'))
    assert calculate(plugin, measurement, definition, valid=Interval.at(START)).proposition.role('object').value == 8


def test_no_inverse_path_or_unprovided_unit_conversion_is_inferred():
    plugin, measurement, definition = setup()
    reversed_measurement = plugin.remember(OWNER, 'held', Quantity(8, Unit.of('opaque-item')),
        measurement=Ref('measurement:reverse'), evidence=evidence('reverse'))
    assert isinstance(calculate(plugin, reversed_measurement, definition), Unknown)
    missing = replace(definition, roles={**definition.roles, 'definition': Ref('definition:missing')})
    assert isinstance(calculate(plugin, measurement, missing), Unknown)
    with pytest.raises(ValueError, match='parameters'):
        plugin.register_calculation('convert', (measurement.id,), context=CalculationContext(OWNER, 'held'),
            params={'unit': Unit.of('opaque-item')}, basis=('old implicit conversion refused',))


def test_unrecognized_derived_definition_does_not_become_observed_rate():
    plugin, measurement, definition = setup()
    plugin.mind.supersede(definition, 'withdraw original')
    fake = replace(definition, roles={**definition.roles, 'definition': Ref('definition:unsupported')})
    plugin.mind.assert_(fake, replace(evidence('forged'), derived_from=('prop:missing',)))
    assert isinstance(calculate(plugin, measurement, fake), Unknown)


def test_scope_rival_is_not_a_cross_scope_conversion_license():
    plugin, measurement, definition = setup()
    plugin.remember_conversion(Ref('definition:other-context'), Unit.of('opaque-pack'), Unit.of('opaque-item'), 100,
        scope=Ref('scope:hypothesis'), valid=Interval(), evidence=evidence('hypothesis'))
    assert calculate(plugin, measurement, definition).proposition.role('object').value == 8


def test_scoped_timed_conversion_chains_without_broadening_source_context():
    scope = Ref('scope:chosen')
    plugin, measurement, definition = setup(scope=scope, valid=Interval(START, END))
    second_definition = plugin.remember_conversion(Ref('definition:second'), Unit.of('opaque-item'), Unit.of('other-unit'), 2,
        scope=scope, valid=Interval(START, END), evidence=evidence('second'))
    first = calculate(plugin, measurement, definition, scope=scope, valid=Interval.at(MIDDLE))
    ref = plugin.register_calculation('convert', (first.proposition.id, second_definition.id),
        context=CalculationContext(Ref('owner:second'), 'held'),
        params={'scope': scope, 'valid': Interval.at(MIDDLE)}, basis=('explicit second direct edge',))
    plugin.select_calculation(ref, reason='explicit chained conversion')
    result = plugin.calculate(ref)
    assert result.proposition.role('object') == Quantity(16, Unit.of('other-unit'))
    assert result.proposition.scope == scope and result.proposition.valid == Interval.at(MIDDLE)
    broad = plugin.register_calculation('convert', (first.proposition.id, second_definition.id),
        context=CalculationContext(Ref('owner:broad'), 'held'),
        params={'scope': scope, 'valid': Interval(START, END)}, basis=('attempt broader source interval',))
    plugin.select_calculation(broad, reason='test source validity refuses broadening')
    assert isinstance(plugin.calculate(broad), Unknown)
    unscoped = plugin.remember_conversion(Ref('definition:unscoped'), Unit.of('opaque-item'), Unit.of('other-unit'), 2,
        scope=None, valid=Interval(), evidence=evidence('unscoped'))
    assert isinstance(calculate(plugin, first.proposition, unscoped), Unknown)


@pytest.mark.parametrize('factor', [0, -1, True, float('inf'), float('nan')])
def test_conversion_equivalence_requires_positive_finite_factor(factor):
    plugin = QuantityPlugin()
    with pytest.raises(TypeError):
        plugin.remember_conversion(Ref('definition:bad'), Unit.of('a'), Unit.of('b'), factor,
            scope=None, valid=Interval(), evidence=evidence('bad'))
