"""Literal immutable units survive structural copying and registered store persistence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json
import pytest

from tensorcode.records import (Claim, Evidence, Proposition, Ref, Store, TypeRegistry,
    encode, decode)
from tensorcode.quantity import Quantity, Unit
from tensorcode.learning.experience import _same
from tensorcode.derivations import (admit_operator, derive, validate_record_support,
    export_derivation, import_derivation)
from tensorcode.outcomes import Unknown


def registry():
    result = TypeRegistry()
    result.register(Unit)
    result.register(Quantity)
    return result


def evidence():
    return Evidence(Ref('source:literal-measurement'), datetime(2026, 1, 1, tzinfo=timezone.utc))


def test_literal_unit_copy_and_mapping_alias_cannot_mutate_unit():
    authored = {' USD. ': 1, 'hours': -1}
    unit = Unit(authored)
    authored[' USD. '] = 99
    assert dict(unit.powers) == {' USD. ': 1, 'hours': -1}
    copied = deepcopy(unit)
    assert _same(copied, unit)
    assert hash(copied) == hash(unit)
    with pytest.raises((TypeError, AttributeError, FrozenInstanceError)):
        copied.powers[' USD. '] = 2
    assert dict(unit.powers) == {' USD. ': 1, 'hours': -1}


@pytest.mark.parametrize('symbol', ['kg', 'kilogram', 'USD', '$', 'Boxes', 'boxes', ' hours. '])
def test_registered_json_roundtrip_preserves_literal_symbol(symbol):
    original = Quantity(3, Unit.of(symbol))
    encoded = encode(original, registry())
    restored, report = decode(json.loads(json.dumps(encoded.data)), registry())
    assert report.lossless
    assert _same(original, restored)
    assert dict(restored.unit.powers) == {symbol: 1}


def test_preexisting_serialized_unit_spellings_are_reconstructed_without_translation():
    data = {'$type': 'Unit', 'fields': {'powers': {'kg': 1, 'hours': -1}}}
    restored, report = decode(data, registry())
    assert report.lossless
    assert dict(restored.powers) == {'kg': 1, 'hours': -1}
    # Already-normalized historical data cannot recover the original spelling.
    historical, _ = decode({'$type': 'Unit', 'fields': {'powers': {'kilogram': 1}}}, registry())
    assert dict(historical.powers) == {'kilogram': 1}
    assert historical != Unit.of('kg')


def test_nested_quantity_claim_and_proposition_identity_survive_store_json():
    store = Store(registry())
    quantity = Quantity(3, Unit({'USD': 1, 'hours': -1}))
    claim = Claim(Ref('owner:one'), 'quoted', quantity)
    store.tell(claim, evidence())
    inner = Proposition('measurement', {'value': quantity})
    outer = Proposition('retained', {'content': inner, 'alternate': (Quantity(3, Unit.of('usd')), quantity)})
    original = store.assert_(outer, evidence())
    reloaded, report = Store.from_json(json.loads(json.dumps(store.to_json())), registry())
    assert report.lossless
    assert _same(reloaded.propositions()[0].proposition, original.proposition)
    assert reloaded.propositions()[0].id == original.id
    assert _same(reloaded.claims()[0].claim, claim)
    assert reloaded.claims()[0].id == claim.id
    assert _same(reloaded.propositions()[0].evidence, original.evidence)


def test_immutable_unit_proof_survives_copy_and_cross_store_import():
    store = Store(registry())
    unit = Unit.of('Hours')
    premise = store.assert_(Proposition('measurement', {'value': Quantity(3, unit)}), evidence())
    def supplied_double(premises, params):
        value = premises[0].roles['value']
        return Proposition('calculated', {'value': Quantity(value.value * params['factor'], value.unit)})
    handle = admit_operator(store, 'double-literal', supplied_double, reason='explicit supplied arithmetic')
    receipt = derive(store, handle, (premise.id,), params={'factor': 2}, basis=('selected measurement',))
    assert not isinstance(receipt, Unknown), receipt
    assert _same(deepcopy(receipt), receipt)
    assert validate_record_support(store, receipt.record_id) is True
    reference = export_derivation(store, deepcopy(receipt))
    assert not isinstance(reference, Unknown), reference
    target = Store(registry())
    imported = import_derivation(target, deepcopy(reference))
    assert not isinstance(imported, Unknown), imported
    assert validate_record_support(target, imported.id) is True
    assert dict(imported.proposition.roles['value'].unit.powers) == {'Hours': 1}
