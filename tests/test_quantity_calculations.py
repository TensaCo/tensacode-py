"""Measurements and arithmetic selection are explicit supplied evidence/policy."""
from datetime import datetime, timezone

import pytest

from tensorcode.agent.plugin import Call
from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.derivations import (export_derivation, import_derivation,
                                   validate_record_support, withdraw_operator)
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.records import Evidence, Ref, Store

OWNER, KIND = Ref('owner:a'), Ref('kind:a')


def remember(plugin, name, value, *, owner=OWNER, predicate='held', kind=KIND, unit='item'):
    return plugin.remember(owner, predicate, Quantity(value, Unit.of(unit)),
        measurement=Ref('measurement:' + name), kind=kind,
        evidence=Evidence(Ref('source:' + name), datetime.now(timezone.utc)))


def select(plugin, operation, operands, *, context=None, params=None):
    context = context or CalculationContext(OWNER, 'held', KIND)
    reference = plugin.register_calculation(operation, tuple(p.id for p in operands),
        context=context, params=params, basis=('explicit test operand/operation selection',))
    assert plugin.select_calculation(reference, reason='explicit test choice') is True
    return reference


def test_equal_valued_independent_measurements_count_as_two_explicit_addends():
    plugin = QuantityPlugin()
    a, b = remember(plugin, 'a', 3), remember(plugin, 'b', 3)
    assert a.id != b.id
    repeated = remember(plugin, 'a', 3)
    assert repeated.id == a.id and len(plugin.mind.propositions()) == 2
    assert len(next(r for r in plugin.mind.propositions() if r.id == a.id).evidence) == 2
    ref = select(plugin, 'sum', (a, b))
    result = plugin.calculate(ref)
    assert result.proposition.role('object') == Quantity(6, Unit.of('item'))
    assert validate_record_support(plugin.mind, result.record_id) is True


def test_rival_identity_even_with_other_predicate_cannot_be_selected_away():
    plugin = QuantityPlugin()
    a = remember(plugin, 'a', 3)
    reference = select(plugin, 'sum', (a,))
    first = plugin.calculate(reference)
    world = Store()
    imported = import_derivation(world, export_derivation(plugin.mind, first))
    remember(plugin, 'a', 4, predicate='different')
    assert isinstance(plugin.calculate(reference), Unknown)
    assert isinstance(validate_record_support(world, imported.id), Unknown)


def test_new_fact_invalidates_population_but_does_not_join_selected_addends():
    plugin = QuantityPlugin()
    a, b = remember(plugin, 'a', 3), remember(plugin, 'b', 4)
    reference = select(plugin, 'sum', (a, b))
    old = plugin.calculate(reference)
    remember(plugin, 'c', 2)
    assert isinstance(validate_record_support(plugin.mind, old.record_id), Unknown)
    fresh = plugin.calculate(reference)
    assert fresh.proposition.role('object') == Quantity(7, Unit.of('item'))


def test_rate_never_infers_multiplication_or_first_operand_pairing():
    plugin = QuantityPlugin()
    rate = plugin.remember(OWNER, 'rate', Quantity(6, Unit.of('ticket') / Unit.of('ride')),
        measurement=Ref('measurement:rate'), evidence=Evidence(Ref('source:rate'), datetime.now(timezone.utc)))
    count = remember(plugin, 'count', 10, kind=None, unit='ride')
    assert not plugin.capabilities()
    assert isinstance(plugin.calculate(select(plugin, 'sum', (rate, count))), Unknown)
    result = plugin.calculate(select(plugin, 'mul', (rate, count)))
    assert result.proposition.role('object') == Quantity(60, Unit.of('ticket'))
    assert not hasattr(plugin, '_total_claim') and not hasattr(plugin, 'total')
    assert not hasattr(plugin, 'observe') and not hasattr(plugin, 'count_properties')


def test_ordered_subtraction_and_explicit_conversion_have_authenticated_results():
    plugin = QuantityPlugin()
    a, b = remember(plugin, 'a', 9, unit='metre'), remember(plugin, 'b', 2, unit='metre')
    forward = plugin.calculate(select(plugin, 'sub', (a, b)))
    reverse = plugin.calculate(select(plugin, 'sub', (b, a)))
    assert forward.proposition.role('object').value == 7
    assert reverse.proposition.role('object').value == -7
    converted = plugin.calculate(select(plugin, 'convert', (a,), params={'unit': Unit.of('centimetre')}))
    assert converted.proposition.role('object') == Quantity(900, Unit.of('centimetre'))


def test_selected_context_replacement_invalidates_old_answer_and_reveal():
    plugin = QuantityPlugin()
    a, b, c = (remember(plugin, name, value) for name, value in [('a', 3), ('b', 4), ('c', 2)])
    old_ref = select(plugin, 'sum', (a, b))
    cap, = plugin.capabilities()
    action = Call(plugin.name, cap.name, (('owner', OWNER), ('kind', KIND)))
    receipt = plugin.execute(action)
    reference, = plugin.reveal(cap, dict(action.args), receipt)
    world = Store()
    imported = import_derivation(world, reference)
    new_ref = select(plugin, 'sum', (a, b, c))
    assert isinstance(validate_record_support(world, imported.id), Unknown)
    assert list(plugin.reveal(cap, dict(action.args), receipt)) == []
    assert isinstance(plugin.calculate(old_ref), Unknown)
    assert plugin.calculate(new_ref).proposition.role('object').value == 9
    assert len(plugin.selection_history) == 2


def test_explicit_empty_selected_count_is_not_world_zero():
    plugin = QuantityPlugin()
    reference = select(plugin, 'count_selected', ())
    result = plugin.calculate(reference)
    assert result.proposition.role('object') == Quantity(0, Unit.of('record'))
    assert validate_record_support(plugin.mind, result.record_id) is True
    assert withdraw_operator(plugin.mind, result.operator, reason='withdraw supplied policy') is True
    assert isinstance(validate_record_support(plugin.mind, result.record_id), Unknown)
    assert isinstance(plugin.calculate(reference), Unknown)


def test_missing_unsupported_or_withdrawn_measurement_refuses():
    plugin = QuantityPlugin()
    a = remember(plugin, 'a', 3)
    reference = select(plugin, 'sum', (a,))
    result = plugin.calculate(reference)
    plugin.mind.supersede(a, 'withdraw original evidence')
    assert isinstance(plugin.calculate(reference), Unknown)
    assert isinstance(validate_record_support(plugin.mind, result.record_id), Unknown)
    with pytest.raises(TypeError):
        plugin.remember(OWNER, 'held', Quantity(3))
    with pytest.raises(TypeError):
        plugin.remember(OWNER, 'held', Quantity(3), measurement=Ref('measurement:x'), evidence=None)


def test_calculation_and_context_must_be_explicitly_selected():
    plugin = QuantityPlugin()
    a = remember(plugin, 'a', 3)
    reference = plugin.register_calculation('sum', (a.id,), context=CalculationContext(OWNER, 'held', KIND), basis=('test',))
    assert isinstance(plugin.calculate(reference), Unknown)
    assert not plugin.capabilities()
    plugin.select_calculation(reference, reason='specific selection')
    cap, = plugin.capabilities()
    assert plugin.execute(Call(plugin.name, cap.name, (('owner', Ref('owner:other')), ('kind', KIND)))).status == 'rejected'


def test_calculated_spelling_never_authenticates_a_direct_observation_operand():
    from tensorcode.records import Proposition
    plugin = QuantityPlugin()
    spoof = Proposition('calculated:sum:held', {'subject': OWNER, 'object': Quantity(3, Unit.of('item'))})
    plugin.mind.assert_(spoof, Evidence(Ref('source:spoof'), datetime.now(timezone.utc)))
    reference = select(plugin, 'sum', (spoof,), context=CalculationContext(OWNER, 'other'))
    assert isinstance(plugin.calculate(reference), Unknown)


def test_derived_operand_cannot_survive_withdrawal_using_observed_copy():
    plugin = QuantityPlugin()
    a = remember(plugin, 'a', 3, unit='metre')
    source_ref = select(plugin, 'sum', (a,), context=CalculationContext(OWNER, 'source'))
    source = plugin.calculate(source_ref)
    converted_ref = select(plugin, 'convert', (source.proposition,),
        context=CalculationContext(OWNER, 'converted'), params={'unit': Unit.of('centimetre')})
    converted = plugin.calculate(converted_ref)
    assert converted.proposition.role('object') == Quantity(300, Unit.of('centimetre'))
    plugin.mind.assert_(source.proposition, Evidence(Ref('source:observed-copy'), datetime.now(timezone.utc)))
    withdraw_operator(plugin.mind, source.operator, reason='withdraw actual calculation policy')
    assert isinstance(validate_record_support(plugin.mind, converted.record_id), Unknown)
    assert isinstance(plugin.calculate(converted_ref), Unknown)


def test_measurement_envelope_cannot_omit_identity():
    from tensorcode.records import Proposition
    plugin = QuantityPlugin()
    malformed = Proposition('quantity_measurement', {'subject': OWNER, 'object': Quantity(3)})
    plugin.mind.assert_(malformed, Evidence(Ref('source:malformed'), datetime.now(timezone.utc)))
    reference = select(plugin, 'sum', (malformed,))
    assert isinstance(plugin.calculate(reference), Unknown)


@pytest.mark.parametrize('operation,params,expected', [
    ('div', None, 3), ('ratio', None, 3), ('percent_of', None, 300),
    ('scale', {'factor': 2}, 12), ('compare', None, 'greater')])
def test_explicit_registered_math_keeps_operation_and_parameters(operation, params, expected):
    plugin = QuantityPlugin()
    a, b = remember(plugin, 'a', 6), remember(plugin, 'b', 2)
    operands = (a,) if operation == 'scale' else (a, b)
    reference = select(plugin, operation, operands, params=params)
    result = plugin.calculate(reference)
    value = result.proposition.role('object')
    assert (value.value if isinstance(value, Quantity) else value) == expected
    assert result.proposition.predicate == 'calculated:' + operation + ':held'


def test_same_operation_predicate_can_chain_across_distinct_contexts():
    plugin = QuantityPlugin()
    a, b = remember(plugin, 'a', 3), remember(plugin, 'b', 4)
    first_ref = select(plugin, 'sum', (a,), context=CalculationContext(OWNER, 'held', KIND))
    first = plugin.calculate(first_ref)
    other = Ref('owner:combined')
    second_ref = select(plugin, 'sum', (first.proposition, b),
                        context=CalculationContext(other, 'held', KIND))
    second = plugin.calculate(second_ref)
    assert second.proposition.predicate == first.proposition.predicate
    assert second.proposition.role('subject') == other
    assert second.proposition.role('object') == Quantity(7, Unit.of('item'))
    assert validate_record_support(plugin.mind, second.record_id) is True
    withdraw_operator(plugin.mind, first.operator, reason='withdraw first selected calculation')
    assert isinstance(validate_record_support(plugin.mind, second.record_id), Unknown)
    assert isinstance(plugin.calculate(second_ref), Unknown)
