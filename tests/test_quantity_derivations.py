"""Supplied arithmetic is replayable evidence, never a new observed amount."""
from dataclasses import replace
from datetime import datetime, timezone

from tensorcode.agent.quantity_plugin import QuantityPlugin, _sum_kind_measurements
from tensorcode.agent.plugin import Call
from tensorcode.derivations import (DerivationReference, import_derivation, validate_record_support,
                                   withdraw_operator)
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.records import Evidence, Proposition, Ref, Store

OWNER, KIND = Ref('owner:a'), Ref('kind:a')


def setup():
    plugin = QuantityPlugin()
    premises = tuple(plugin.remember(OWNER, 'holds', Quantity(value, Unit.of('item')), kind=KIND)
                     for value in (2, 3))
    cap = next(cap for cap in plugin.capabilities() if cap.name == 'amount_of_kind_holds')
    action = Call(plugin.name, cap.name, (('owner', OWNER), ('kind', KIND)))
    return plugin, premises, cap, action


def perform():
    plugin, premises, cap, action = setup()
    receipt = plugin.execute(action)
    assert receipt.status == 'applied'
    reference, = plugin.reveal(cap, dict(action.args), receipt)
    return plugin, premises, cap, action, receipt, reference


def test_kind_sum_publishes_authenticated_derived_predicate_not_observation():
    plugin, premises, cap, action, receipt, reference = perform()
    assert type(reference) is DerivationReference
    assert reference.proposition == Proposition('total_kind:holds',
        {'subject': OWNER, 'kind': KIND, 'object': Quantity(5, Unit.of('item'))})
    assert cap.informs[0].query.predicate == 'total_kind:holds'
    derived, = plugin.mind.propositions('total_kind:holds')
    assert set(derived.evidence[0].derived_from) == {p.id for p in premises}
    assert validate_record_support(plugin.mind, derived.id) is True
    world = Store()
    imported = import_derivation(world, reference)
    assert not isinstance(imported, Unknown)
    assert validate_record_support(world, imported.id) is True
    assert not world.propositions('holds')


def test_missing_premise_withdraws_source_import_and_pending_reveal():
    plugin, premises, cap, action, receipt, reference = perform()
    world = Store()
    imported = import_derivation(world, reference)
    plugin.mind.supersede(premises[0], 'explicit measurement withdrawal')
    assert isinstance(validate_record_support(plugin.mind, reference.proposition.id), Unknown)
    assert isinstance(validate_record_support(world, imported.id), Unknown)
    assert list(plugin.reveal(cap, dict(action.args), receipt)) == []
    assert isinstance(import_derivation(Store(), reference), Unknown)


def test_new_matching_fact_invalidates_old_population_and_requires_fresh_sum():
    plugin, premises, cap, action, receipt, reference = perform()
    plugin.remember(OWNER, 'holds', Quantity(4, Unit.of('item')), kind=KIND)
    assert isinstance(validate_record_support(plugin.mind, reference.proposition.id), Unknown)
    assert list(plugin.reveal(cap, dict(action.args), receipt)) == []
    assert plugin.total_of_kind(OWNER, 'holds', KIND) == Quantity(9, Unit.of('item'))
    fresh = plugin._kind_derivations[(OWNER, 'holds', KIND)]
    assert validate_record_support(plugin.mind, fresh.record_id) is True


def test_withdrawn_operator_blocks_replay_export_and_new_calculation():
    plugin, premises, cap, action, receipt, reference = perform()
    assert withdraw_operator(plugin.mind, plugin._kind_sum_operator, reason='supplied policy withdrawn') is True
    assert isinstance(validate_record_support(plugin.mind, reference.proposition.id), Unknown)
    assert list(plugin.reveal(cap, dict(action.args), receipt)) == []
    assert isinstance(plugin.total_of_kind(OWNER, 'holds', KIND), Unknown)


def test_operator_replay_requires_exact_owner_kind_and_unqualified_operands():
    plugin, premises, _, _ = setup()
    params = {'owner': OWNER, 'kind': KIND, 'predicate': 'holds'}
    assert _sum_kind_measurements(premises, params).role('object') == Quantity(5, Unit.of('item'))
    for changed in (replace(premises[0], polarity=False), replace(premises[0], scope=Ref('scope:other')),
                    replace(premises[0], roles={**premises[0].roles, 'kind': Ref('kind:other')})):
        assert isinstance(_sum_kind_measurements((changed, premises[1]), params), Unknown)
    assert isinstance(_sum_kind_measurements((), params), Unknown)


def test_derived_only_unrecognized_operand_never_certifies_sum():
    plugin = QuantityPlugin()
    fake = Proposition('holds', {'subject': OWNER, 'kind': KIND, 'object': Quantity(5, Unit.of('item'))})
    plugin.mind.assert_(fake, Evidence(Ref('test:unverified'), datetime.now(timezone.utc),
                                      derived_from=('proposition:missing',)))
    assert isinstance(plugin.total_of_kind(OWNER, 'holds', KIND), Unknown)
    assert not plugin.mind.propositions('total_kind:holds')
