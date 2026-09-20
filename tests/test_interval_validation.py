"""Temporal qualifiers must remain genuine datetime intervals through replay."""
from datetime import date, datetime, timedelta, timezone

import pytest

from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.derivations import validate_record_support
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.quantity_calculations import CalculationContext, calculation_operator
from tensorcode.records import Evidence, Interval, Ref


@pytest.mark.parametrize('value', [False, True, 0, 1, 0.5, '2026-01-01', date(2026, 1, 1)])
def test_non_datetime_endpoints_are_rejected(value):
    with pytest.raises(TypeError, match='datetimes'):
        Interval(value, None)
    with pytest.raises(TypeError, match='datetimes'):
        Interval(None, value)


def test_naive_and_aware_intervals_are_valid_separately_but_not_mixed():
    naive = datetime(2026, 1, 1)
    aware = naive.replace(tzinfo=timezone.utc)
    assert Interval(naive, naive + timedelta(days=1)).contains(naive)
    assert Interval(aware, aware + timedelta(days=1)).contains(aware)
    assert Interval().overlap(Interval.at(naive)) == Interval.at(naive)
    assert Interval().overlap(Interval.at(aware)) == Interval.at(aware)
    with pytest.raises(TypeError, match='awareness'):
        Interval(naive, aware)
    with pytest.raises(TypeError, match='awareness'):
        Interval(None, naive).overlap(Interval(aware, None))
    with pytest.raises(TypeError, match='awareness'):
        Interval.at(naive).contains(aware)


def test_order_and_closed_interval_overlap_remain_explicit():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    with pytest.raises(ValueError, match='precedes'):
        Interval(end, start)
    assert Interval(start, end).overlap(Interval(end, None)) == Interval.at(end)
    assert Interval(start, end).overlap(Interval(end + timedelta(seconds=1), None)) is None
    with pytest.raises(TypeError):
        Interval().contains(False)


def test_forged_interval_is_revalidated_before_temporal_operations():
    forged = Interval()
    object.__setattr__(forged, 'start', False)
    object.__setattr__(forged, 'end', True)
    with pytest.raises(TypeError, match='datetimes'):
        forged.overlap(Interval())
    with pytest.raises(TypeError, match='datetimes'):
        Interval().overlap(forged)
    with pytest.raises(TypeError, match='datetimes'):
        forged.contains(datetime(2026, 1, 1))


def test_conversion_cannot_authenticate_forged_boolean_validity_at_replay():
    plugin, owner = QuantityPlugin(), Ref('owner:test')
    evidence = Evidence(Ref('source:test'), datetime(2026, 1, 1, tzinfo=timezone.utc))
    measured = plugin.remember(owner, 'amount', Quantity(2, Unit.of('pack')),
        measurement=Ref('measurement:test'), evidence=evidence)
    conversion = plugin.remember_conversion(Ref('definition:test'), Unit.of('pack'), Unit.of('item'), 4,
        scope=None, valid=Interval(), evidence=evidence)
    context = CalculationContext(owner, 'converted')
    reference = plugin.register_calculation('convert', (measured.id, conversion.id),
        context=context, params={'scope': None, 'valid': Interval()}, basis=('explicit test conversion',))
    assert plugin.select_calculation(reference, reason='explicit test choice') is True
    result = plugin.calculate(reference)
    assert validate_record_support(plugin.mind, result.record_id) is True
    # An object assembled outside normal construction cannot bypass replay.
    forged = Interval()
    object.__setattr__(forged, 'start', False)
    object.__setattr__(forged, 'end', True)
    params = {'context': context, 'params': {'scope': None, 'valid': forged},
              'ordered_operand_ids': (measured.id, conversion.id)}
    try:
        outcome = calculation_operator('convert', (measured, conversion), params)
    except (TypeError, ValueError):
        pass  # The derivation boundary converts invalid replay inputs to Unknown.
    else:
        assert isinstance(outcome, Unknown)
    # Corrupt retained evidence rather than constructing a new valid interval.
    stored = next(record for record in plugin.mind.propositions() if record.id == measured.id)
    object.__setattr__(stored.proposition.valid, 'start', False)
    assert isinstance(plugin.calculate(reference), Unknown)
    assert isinstance(validate_record_support(plugin.mind, result.record_id), Unknown)
