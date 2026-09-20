"""Selected arithmetic keeps its live measurement and selection authority."""
from datetime import datetime, timezone

from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.agent.plugin import Call
from tensorcode.derivations import (DerivationReference, import_derivation, validate_record_support,
                                   withdraw_operator)
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.records import Evidence, Proposition, Ref, Store

OWNER, KIND = Ref('owner:a'), Ref('kind:a')
CONTEXT = CalculationContext(OWNER, 'holds', KIND)


def remember(plugin, value, identity):
    return plugin.remember(OWNER, 'holds', Quantity(value, Unit.of('item')), kind=KIND,
        measurement=Ref(identity), evidence=Evidence(Ref('fixture:measurement'),
            datetime(2026, 9, 19, tzinfo=timezone.utc), locator=identity))


def choose(plugin, operands):
    reference = plugin.register_calculation('sum', tuple(p.id for p in operands),
        context=CONTEXT, basis=('Explicit independent addends supplied by fixture',))
    assert plugin.select_calculation(reference, reason='Fixture selects this ordered sum') is True
    return reference


def setup():
    plugin = QuantityPlugin()
    premises = (remember(plugin, 2, 'measurement:first'), remember(plugin, 3, 'measurement:second'))
    calculation = choose(plugin, premises)
    cap = next(cap for cap in plugin.capabilities() if cap.name == 'calculate_sum_kind_holds')
    action = Call(plugin.name, cap.name, (('owner', OWNER), ('kind', KIND)))
    return plugin, premises, calculation, cap, action


def perform():
    plugin, premises, calculation, cap, action = setup()
    receipt = plugin.execute(action)
    assert receipt.status == 'applied'
    reference, = plugin.reveal(cap, dict(action.args), receipt)
    return plugin, premises, calculation, cap, action, receipt, reference


def test_selected_sum_publishes_authenticated_result_not_measurement():
    plugin, premises, calculation, cap, action, receipt, reference = perform()
    assert type(reference) is DerivationReference
    assert reference.proposition == Proposition('calculated:sum:holds',
        {'subject': OWNER, 'kind': KIND, 'object': Quantity(5, Unit.of('item'))})
    assert cap.informs[0].query.predicate == 'calculated:sum:holds'
    derived, = plugin.mind.propositions('calculated:sum:holds')
    assert set(derived.evidence[0].derived_from) == {p.id for p in premises}
    assert validate_record_support(plugin.mind, derived.id) is True
    world = Store()
    imported = import_derivation(world, reference)
    assert not isinstance(imported, Unknown)
    assert validate_record_support(world, imported.id) is True
    assert not world.propositions('quantity_measurement')


def test_missing_operand_withdraws_source_import_and_pending_reveal():
    plugin, premises, calculation, cap, action, receipt, reference = perform()
    world = Store()
    imported = import_derivation(world, reference)
    plugin.mind.supersede(premises[0], 'Explicit measurement withdrawal')
    assert isinstance(validate_record_support(plugin.mind, reference.proposition.id), Unknown)
    assert isinstance(validate_record_support(world, imported.id), Unknown)
    assert list(plugin.reveal(cap, dict(action.args), receipt)) == []
    assert isinstance(import_derivation(Store(), reference), Unknown)


def test_new_measurement_does_not_implicitly_join_selected_operands():
    plugin, premises, calculation, cap, action, receipt, reference = perform()
    third = remember(plugin, 4, 'measurement:third')
    assert isinstance(validate_record_support(plugin.mind, reference.proposition.id), Unknown)
    assert list(plugin.reveal(cap, dict(action.args), receipt)) == []
    refreshed = plugin.calculate(calculation)
    assert refreshed.proposition.role('object') == Quantity(5, Unit.of('item'))
    assert validate_record_support(plugin.mind, refreshed.record_id) is True
    revised = choose(plugin, (*premises, third))
    assert isinstance(validate_record_support(plugin.mind, refreshed.record_id), Unknown)
    fresh = plugin.calculate(revised)
    assert fresh.proposition.role('object') == Quantity(9, Unit.of('item'))
    assert validate_record_support(plugin.mind, fresh.record_id) is True


def test_withdrawn_operator_cannot_be_recreated_by_execution():
    plugin, premises, calculation, cap, action, receipt, reference = perform()
    proof = plugin.calculate(calculation)
    assert withdraw_operator(plugin.mind, proof.operator, reason='Withdraw supplied policy') is True
    assert isinstance(validate_record_support(plugin.mind, reference.proposition.id), Unknown)
    assert list(plugin.reveal(cap, dict(action.args), receipt)) == []
    assert isinstance(plugin.calculate(calculation), Unknown)


def test_rival_value_for_existing_identity_blocks_selected_sum():
    plugin, premises, calculation, cap, action, receipt, reference = perform()
    remember(plugin, 200, 'measurement:first')
    assert isinstance(plugin.calculate(calculation), Unknown)
    assert isinstance(validate_record_support(plugin.mind, reference.proposition.id), Unknown)


def test_unrecognized_derived_operand_never_certifies_sum():
    plugin = QuantityPlugin()
    fake = Proposition('quantity_measurement', {'subject': OWNER, 'kind': KIND,
        'predicate': 'holds', 'measurement': Ref('measurement:fake'), 'object': Quantity(5, Unit.of('item'))})
    plugin.mind.assert_(fake, Evidence(Ref('test:unverified'), datetime.now(timezone.utc),
                                      derived_from=('proposition:missing',)))
    calculation = choose(plugin, (fake,))
    assert isinstance(plugin.calculate(calculation), Unknown)
    assert not plugin.mind.propositions('calculated:sum:holds')
