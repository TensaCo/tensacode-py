"""Temporal contents and explicit withdrawal evidence survive store persistence."""
from datetime import datetime, timezone
import json

import pytest

from tensorcode.records import Evidence, Interval, Proposition, Ref, Store, Var


def instant(year):
    return Interval.at(datetime(year, 1, 1, tzinfo=timezone.utc))


def evidence(name):
    return Evidence(Ref('source:' + name), datetime(2026, 1, 1, tzinfo=timezone.utc),
                    locator='retained:' + name, method='explicit observation')


def fact(year):
    return Proposition('located', {'entity': Ref('item:a'), 'place': Ref('place:b')}, valid=instant(year))


def roundtrip(store):
    restored, report = Store.from_json(json.loads(json.dumps(store.to_json())), store.registry)
    assert not report.opaque
    return restored


def test_validity_is_part_of_proposition_identity_and_evidence_does_not_merge_across_time():
    store = Store()
    first, later = fact(2001), fact(2026)
    assert first.id != later.id
    old = store.assert_(first, evidence('old'))
    new = store.assert_(later, evidence('new'))
    assert old is not new
    assert store.assert_(fact(2001), evidence('corroboration')) is old
    assert old.evidence == [evidence('old'), evidence('corroboration')]
    assert new.evidence == [evidence('new')]
    restored = roundtrip(store)
    rows = {r.id: r for r in restored.propositions()}
    assert set(rows) == {first.id, later.id}
    assert rows[first.id].proposition.valid == instant(2001)
    assert rows[later.id].proposition.valid == instant(2026)
    assert rows[first.id].evidence == old.evidence
    assert rows[later.id].evidence == new.evidence


@pytest.mark.parametrize('nested', [lambda p: p, lambda p: {'claim': [p]}])
def test_nested_temporal_contents_are_distinct_and_survive_serialization(nested):
    store = Store()
    first = Proposition('reported', {'content': nested(fact(2001))})
    later = Proposition('reported', {'content': nested(fact(2026))})
    assert first.id != later.id
    store.assert_(first, evidence('old'))
    store.assert_(later, evidence('new'))
    restored = roundtrip(store)
    assert {r.id for r in restored.propositions()} == {first.id, later.id}
    assert {r.id: r.proposition for r in restored.propositions()} == {first.id: first, later.id: later}


def test_typed_scalar_contents_do_not_collapse_even_when_python_equality_aliases():
    store = Store()
    values = (True, 1, 1.0, '1', False, 0, 0.0, None)
    rows = [store.assert_(Proposition('measured', {'value': value}, valid=instant(2026)), evidence(str(i)))
            for i, value in enumerate(values)]
    assert len({row.id for row in rows}) == len(values)
    restored = roundtrip(store)
    assert {row.id for row in restored.propositions()} == {row.id for row in rows}
    assert {row.id: type(row.proposition.roles['value']) for row in restored.propositions()} == {
        row.id: type(row.proposition.roles['value']) for row in rows}


def test_supersede_retains_explicit_evidence_and_roundtrips_withdrawal():
    store = Store()
    record = store.assert_(fact(2001), evidence('old'))
    withdrawal = evidence('withdrawal')
    assert store.supersede(Proposition('located', {'entity': Ref('item:a'), 'place': Var('place')}),
                           'source corrected', evidence=(withdrawal,)) == [record.id]
    assert record.retracted.reason == 'source corrected'
    assert record.retracted.evidence == (withdrawal,)
    assert not store.propositions() and not store.find(Proposition('located'))
    restored = roundtrip(store)
    assert not restored.propositions()
    serialized = restored.to_json()['propositions'][0]
    assert serialized['retracted'] == store.to_json()['propositions'][0]['retracted']
    # Corroborating historical evidence must not silently undo a withdrawal.
    assert restored.assert_(fact(2001), evidence('later assertion')).retracted == record.retracted
    assert not restored.propositions()


def test_supersede_without_evidence_does_not_fabricate_it_and_invalid_evidence_is_atomic():
    store = Store()
    record = store.assert_(fact(2001), evidence('old'))
    with pytest.raises(TypeError, match='tuple of Evidence'):
        store.supersede(Proposition('located'), evidence=datetime.now(timezone.utc))
    assert record.retracted is None
    assert store.supersede(Proposition('located')) == [record.id]
    assert record.retracted.evidence == ()
    assert not roundtrip(store).propositions()


@pytest.mark.parametrize('invalid', [None, (), False, 'unbounded'])
def test_invalid_interval_values_cannot_alias_unbounded_content(invalid):
    with pytest.raises(TypeError, match='validity must be an Interval'):
        Proposition('located', valid=invalid)
