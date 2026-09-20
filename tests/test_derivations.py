"""Supplied operators replay exact detached premises; provenance is not a premise-ID claim."""
from dataclasses import replace
from datetime import datetime, timezone, timedelta
import pytest
from tensorcode.records import Store, Proposition, Ref, Evidence, Interval
from tensorcode.outcomes import Unknown
from tensorcode.derivations import (admit_operator, withdraw_operator, derive, validate_record_support,
    export_derivation, import_derivation, validate_derivation_reference)


def observed(store, value=2, *, predicate='measurement', valid=Interval()):
    return store.assert_(Proposition(predicate, {'value': value}, valid=valid),
        Evidence(Ref('source:fixture'), datetime.now(timezone.utc)))


def operator(premises, params):
    return Proposition('total', {'value': sum(p.roles['value'] for p in premises) + params['offset']})


def fixture(*, population=False):
    store = Store()
    a, b = observed(store, 2), observed(store, 3)
    handle = admit_operator(store, 'sum', operator, reason='supplied arithmetic operator')
    receipt = derive(store, handle, (a.id, b.id), params={'offset': 0}, basis=('explicit operands',),
        population_predicate='measurement' if population else None)
    assert not isinstance(receipt, Unknown), receipt
    return store, handle, receipt, (a, b)


def test_replayable_operator_receipt_and_explicit_population():
    store, _, receipt, _ = fixture(population=True)
    assert receipt.proposition.roles == {'value': 5}
    assert validate_record_support(store, receipt.record_id) is True
    observed(store, 9)
    assert isinstance(validate_record_support(store, receipt.record_id), Unknown)


@pytest.mark.parametrize('change', ['withdraw', 'version', 'erase', 'retract', 'changed', 'contradict'])
def test_changed_premise_or_operator_invalidates_receipt(change):
    store, handle, receipt, (a, b) = fixture()
    if change == 'withdraw': withdraw_operator(store, handle, reason='withdraw')
    if change == 'version': admit_operator(store, 'sum', operator, reason='new version')
    if change == 'erase': a.evidence.clear()
    if change == 'retract': store.supersede(a.proposition)
    if change == 'changed': a.evidence.append(Evidence(Ref('source:changed'), datetime.now(timezone.utc)))
    if change == 'contradict': store.assert_(replace(a.proposition, polarity=False), a.evidence[0])
    assert isinstance(validate_record_support(store, receipt.record_id), Unknown)


def test_forged_premise_ids_and_malformed_direct_evidence_are_not_support():
    store = Store()
    a = observed(store)
    fake = store.assert_(Proposition('fake', {'value': 2}),
        replace(a.evidence[0], derived_from=(a.id,), locator='derivation:invented'))
    assert isinstance(validate_record_support(store, fake.id), Unknown)
    a.evidence[:] = [replace(a.evidence[0], observed_at='today')]
    assert isinstance(validate_record_support(store, a.id), Unknown)


def test_independent_direct_evidence_survives_failed_derived_branch():
    store, handle, receipt, _ = fixture()
    store.assert_(receipt.proposition, Evidence(Ref('source:independent'), datetime.now(timezone.utc)))
    withdraw_operator(store, handle, reason='withdraw')
    assert validate_record_support(store, receipt.record_id) is True
    assert isinstance(export_derivation(store, receipt), Unknown)


def test_nested_derivations_validate_and_obey_budgets():
    store, _, receipt, _ = fixture()
    handle = admit_operator(store, 'double', lambda ps, params: Proposition('double', {'value': ps[0].roles['value'] * 2}), reason='supplied')
    second = derive(store, handle, (receipt.record_id,), basis=('explicit chain',))
    assert not isinstance(second, Unknown), second
    assert validate_record_support(store, second.record_id) is True
    assert isinstance(validate_record_support(store, second.record_id, max_depth=1), Unknown)
    assert isinstance(validate_record_support(store, second.record_id, max_nodes=1), Unknown)


def test_operator_only_receives_detached_values_and_mutating_store_fails_closed():
    store = Store()
    a = observed(store)
    def mutate(ps, params):
        ps[0].roles['value'] = 99
        return Proposition('copy', {'value': ps[0].roles['value']})
    handle = admit_operator(store, 'mutate-local', mutate, reason='explicit')
    receipt = derive(store, handle, (a.id,), basis=('test',))
    assert not isinstance(receipt, Unknown), receipt
    assert a.proposition.roles['value'] == 2
    def bad(ps, params):
        observed(store, 10)
        return Proposition('bad', {'value': 10})
    handle = admit_operator(store, 'bad', bad, reason='explicit')
    assert isinstance(derive(store, handle, (a.id,), basis=('test',)), Unknown)
    assert not store.propositions('bad')


def test_nondeterministic_and_reentrant_operator_cannot_publish():
    store = Store()
    a = observed(store)
    count = 0
    def unstable(ps, params):
        nonlocal count
        count += 1
        assert isinstance(validate_record_support(store, a.id), Unknown)
        return Proposition('unstable', {'value': count})
    handle = admit_operator(store, 'unstable', unstable, reason='explicit')
    assert isinstance(derive(store, handle, (a.id,), basis=('test',)), Unknown)
    assert not store.propositions('unstable')


def test_overlapping_opposite_validity_blocks_direct_support():
    store = Store()
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = observed(store, valid=Interval.at(at))
    observed(store, predicate='other')
    store.assert_(replace(a.proposition, polarity=False, valid=Interval(at + timedelta(days=1), None)), a.evidence[0])
    assert validate_record_support(store, a.id) is True
    store.assert_(replace(a.proposition, polarity=False, valid=Interval()), a.evidence[0])
    assert isinstance(validate_record_support(store, a.id), Unknown)


def test_cross_store_reference_retains_live_source_authority():
    store, handle, receipt, _ = fixture(population=True)
    reference = export_derivation(store, receipt)
    assert not isinstance(reference, Unknown), reference
    target = Store()
    imported = import_derivation(target, reference)
    assert not isinstance(imported, Unknown), imported
    assert validate_record_support(target, imported.id) is True
    assert validate_derivation_reference(reference) is True
    observed(store, 11)
    assert isinstance(validate_record_support(target, imported.id), Unknown)
    assert isinstance(validate_derivation_reference(reference), Unknown)


def test_forged_cross_store_reference_and_missing_receipt_evidence_reject():
    store, _, receipt, _ = fixture()
    reference = export_derivation(store, receipt)
    assert isinstance(import_derivation(Store(), replace(reference, receipt_id='derivation:forged')), Unknown)
    assert isinstance(import_derivation(Store(), replace(reference, proposition=Proposition('fake'))), Unknown)
    store._props[receipt.record_id].evidence.clear()
    assert isinstance(validate_derivation_reference(reference), Unknown)


def test_stripping_derived_from_does_not_relabel_receipt_as_direct_evidence():
    store, _, receipt, _ = fixture()
    record = store._props[receipt.record_id]
    record.evidence[:] = [replace(receipt.evidence, derived_from=())]
    assert isinstance(validate_record_support(store, record.id), Unknown)


def test_cross_store_bridge_can_feed_a_further_replayable_derivation():
    source, handle, receipt, _ = fixture()
    target = Store()
    imported = import_derivation(target, export_derivation(source, receipt))
    local = admit_operator(target, 'double', lambda ps, params: Proposition('double', {'value': ps[0].roles['value'] * 2}), reason='explicit')
    nested = derive(target, local, (imported.id,), basis=('use imported receipt',))
    assert not isinstance(nested, Unknown), nested
    assert validate_record_support(target, nested.record_id) is True
    withdraw_operator(source, handle, reason='source withdrawn')
    assert isinstance(validate_record_support(target, nested.record_id), Unknown)


def test_derived_branch_survives_an_independent_sibling_operator_withdrawal():
    store, first, receipt, premises = fixture()
    second = admit_operator(store, 'independent-sum', operator, reason='independent operator admission')
    other = derive(store, second, tuple(p.id for p in premises), params={'offset': 0}, basis=('second derivation',))
    assert not isinstance(other, Unknown), other
    withdraw_operator(store, first, reason='withdraw first')
    assert validate_record_support(store, receipt.record_id) is True
    assert isinstance(export_derivation(store, receipt), Unknown)
    assert not isinstance(export_derivation(store, other), Unknown)


@pytest.mark.parametrize('entrypoint', ['validate', 'derive', 'export', 'import'])
def test_outer_operator_cannot_invalidate_previously_visited_source(entrypoint):
    source, _, receipt, premises = fixture()
    target = Store()
    reference = export_derivation(source, receipt)
    imported = import_derivation(target, reference)
    armed = [False]
    def outer(ps, params):
        if armed[0]:
            armed[0] = False
            source.supersede(premises[0].proposition, why='withdraw during outer replay')
        return Proposition('outer', {'value': ps[0].roles['value']})
    handle = admit_operator(target, 'outer', outer, reason='explicit supplied procedure')
    outer_receipt = derive(target, handle, (imported.id,), basis=('explicit',))
    assert not isinstance(outer_receipt, Unknown), outer_receipt
    outer_ref = export_derivation(target, outer_receipt)
    armed[0] = True
    result = {'validate': lambda: validate_record_support(target, outer_receipt.record_id),
              'derive': lambda: derive(target, handle, (imported.id,), basis=('explicit',)),
              'export': lambda: export_derivation(target, outer_receipt),
              'import': lambda: import_derivation(Store(), outer_ref)}[entrypoint]()
    assert isinstance(result, Unknown), result


def test_operator_admission_and_withdrawal_reasons_remain_auditable():
    from tensorcode.derivations import operator_history
    store, handle, _, _ = fixture()
    withdraw_operator(store, handle, reason='teacher retracts procedure')
    history = operator_history(store)
    assert history[0].reason == 'supplied arithmetic operator'
    assert history[-1].reason == 'teacher retracts procedure'
    assert history[-1].status == 'withdrawn'


def test_batch_reference_validation_checks_earlier_store_after_later_replay():
    from tensorcode.derivations import validate_derivation_references
    source, _, receipt, premises = fixture()
    first = export_derivation(source, receipt)
    other = Store()
    operand = observed(other, 8)
    armed = [False]
    def later(ps, params):
        if armed[0]:
            armed[0] = False
            source.supersede(premises[0].proposition)
        return Proposition('later', {'value': 8})
    handle = admit_operator(other, 'later', later, reason='explicit')
    second_receipt = derive(other, handle, (operand.id,), basis=('explicit',))
    second = export_derivation(other, second_receipt)
    armed[0] = True
    assert isinstance(validate_derivation_references((first, second)), Unknown)


def test_branch_budget_bounds_replay_callbacks_across_failed_alternatives():
    store = Store()
    premise = observed(store)
    armed, calls = [False], [0]
    def alternative(ps, params):
        calls[0] += 1
        return Proposition('answer', {'value': 99 if armed[0] else 2})
    for index in range(8):
        handle = admit_operator(store, 'alternative-' + str(index), alternative, reason='explicit alternative')
        receipt = derive(store, handle, (premise.id,), basis=('supplied',))
        assert not isinstance(receipt, Unknown), receipt
    armed[0], calls[0] = True, 0
    assert isinstance(validate_record_support(store, receipt.record_id, max_nodes=5), Unknown)
    assert calls[0] <= 1


@pytest.mark.parametrize('budget', [True, 0, -1])
def test_export_import_reject_invalid_budget_types_and_values(budget):
    source, _, receipt, _ = fixture()
    reference = export_derivation(source, receipt)
    assert isinstance(export_derivation(source, receipt, max_nodes=budget), Unknown)
    assert isinstance(import_derivation(Store(), reference, max_depth=budget), Unknown)
