"""Quantities: units carried, dimensions checked, arithmetic shown."""

import pytest

from datetime import datetime, timezone

from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.derivations import validate_record_support
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.outcomes import Unknown
from tensorcode.quantity import (Quantity, Unit, add, compare, convert, div, mul,
                                percent_of, ratio, scale, sub)
from tensorcode.records import Evidence, Ref


def test_adding_different_dimensions_is_refused_not_approximated():
    sheep = Quantity(12, Unit.of("sheep"))
    price = Quantity(5, Unit.of("coin") / Unit.of("sheep"))
    refused = add(sheep, price)
    assert isinstance(refused, Unknown) and refused.reason == "dimension_mismatch"
    assert "sheep" in refused.detail and "coin" in refused.detail
    # the same pair multiplies perfectly well, and the unit says what came out
    assert str(mul(sheep, price)) == "60 coin"


def test_different_literal_unit_spellings_require_explicit_conversion_evidence():
    assert isinstance(add(Quantity(1, Unit.of("kg")), Quantity(500, Unit.of("gram"))), Unknown)
    assert isinstance(compare(Quantity(1, Unit.of("kg")), Quantity(1, Unit.of("lb"))), Unknown)
    assert isinstance(convert(Quantity(1, Unit.of("kg")), Unit.of("kilogram")), Unknown)
    assert add(Quantity(1, Unit.of("kg")), Quantity(2, Unit.of("kg"))) == Quantity(3, Unit.of("kg"))


def test_rates_compose_and_cancel():
    per_hour = Quantity(12, Unit.of("dollar") / Unit.of("hour"))
    worked = Quantity(3, Unit.of("hour"))
    paid = mul(worked, per_hour)
    assert str(paid) == "36 dollar"
    assert str(div(paid, worked)) == "12 dollar/hour"


def test_ratios_and_percentages_are_dimensionless_and_checked():
    part, whole = Quantity(3, Unit.of("sheep")), Quantity(12, Unit.of("sheep"))
    assert ratio(part, whole).value == pytest.approx(0.25)
    assert percent_of(part, whole).value == pytest.approx(25.0)
    assert isinstance(ratio(part, Quantity(2, Unit.of("coin"))), Unknown)


def test_dividing_by_zero_refuses():
    assert isinstance(div(Quantity(1, Unit.of("coin")), Quantity(0, Unit.of("sheep"))), Unknown)


@pytest.mark.parametrize("left,right", [
    ("gas", "ga"), ("news", "new"), ("MS", "ms"), ("box", "boxes"),
    ("feet", "foot"), ("kg", "kilogram"), ("usd", "dollar"), ("x.", "x"), (" x", "x"),
])
def test_literal_symbols_are_not_normalized(left, right):
    a, b = Unit.of(left), Unit.of(right)
    assert dict(a.powers) == {left: 1} and dict(b.powers) == {right: 1}
    assert a != b
    assert isinstance(add(Quantity(1, a), Quantity(2, b)), Unknown)


def test_unit_inputs_and_nested_powers_are_immutable():
    from dataclasses import FrozenInstanceError
    powers = {"MS": 1, "gas": -2}
    unit = Unit(powers)
    powers["MS"] = 5
    assert unit.powers == {"MS": 1, "gas": -2}
    with pytest.raises(TypeError):
        unit.powers["MS"] = 5
    with pytest.raises(FrozenInstanceError):
        unit.powers.entries = (("MS", 5),)
    assert isinstance(unit.powers.entries, tuple)
    assert hash(unit) == hash(Unit({"gas": -2, "MS": 1}))


def test_no_global_registry_can_reinterpret_a_retained_unit(monkeypatch):
    import tensorcode.quantity as module
    unit = Unit.of("MS")
    monkeypatch.setattr(module, "BASE_UNITS", {"MS": ("time", 1000)}, raising=False)
    assert unit.dimension == (("MS", 1),)
    assert Unit.of("MS") == unit
    assert isinstance(convert(Quantity(1, unit), Unit.of("second")), Unknown)


def test_formal_products_cancel_without_choosing_a_base_symbol():
    left, right = Unit.of("MS"), Unit.of("ms")
    assert ((left / right) * right) == left
    assert left / left == Unit()
    assert left ** 0 == Unit()
    assert left * right == right * left
    assert left != right


@pytest.mark.parametrize("powers", [{"x": True}, {"x": 1.5}, {"": 1}, {1: 1}, {"x": []}])
def test_invalid_unit_symbols_or_exponents_fail(powers):
    with pytest.raises(TypeError):
        Unit(powers)


@pytest.mark.parametrize("value", [True, "3", float("nan"), float("inf"), None])
def test_quantity_values_are_finite_typed_numbers(value):
    with pytest.raises(ValueError):
        Quantity(value, Unit.of("x"))
    with pytest.raises(ValueError):
        scale(Quantity(1), value)


def test_comparison_does_not_invent_an_equality_tolerance():
    assert compare(Quantity(1), Quantity(1 + 1e-12)) == "less"


def measured(plugin, owner, predicate, quantity, identity):
    return plugin.remember(owner, predicate, quantity, measurement=Ref(identity),
        evidence=Evidence(Ref("obs:note"), datetime.now(timezone.utc), method="supplied-test-measurement"))


def calculation(plugin, operation, premises, owner, predicate, params=None):
    reference = plugin.register_calculation(operation, tuple(p.id for p in premises),
        context=CalculationContext(owner, predicate), params=params,
        basis=("Explicit arithmetic operands and operation supplied by the test",))
    assert plugin.select_calculation(reference, reason="Explicit test calculation selection") is True
    return plugin.calculate(reference)


def test_a_derivation_records_its_working_and_falls_with_its_premises():
    plugin, owner = QuantityPlugin(), Ref("entity:Anem")
    held = measured(plugin, owner, "has", Quantity(12, Unit.of("sheep")), "measurement:held")
    price = measured(plugin, Ref("entity:market"), "price",
                     Quantity(5, Unit.of("coin") / Unit.of("sheep")), "measurement:price")
    revenue = calculation(plugin, "mul", [held, price], owner, "revenue")
    assert revenue.proposition.role("object") == Quantity(60, Unit.of("coin"))
    assert revenue.proposition.predicate == "calculated:mul:revenue"
    assert set(revenue.premise_ids) == {held.id, price.id}
    assert price.role("object") == Quantity(5, Unit.of("coin") / Unit.of("sheep"))
    assert validate_record_support(plugin.mind, revenue.record_id) is True
    plugin.mind.supersede(price)
    assert isinstance(validate_record_support(plugin.mind, revenue.record_id), Unknown)


def test_a_derivation_over_mismatched_units_refuses_and_records_nothing():
    plugin, owner = QuantityPlugin(), Ref("entity:Anem")
    sheep = measured(plugin, owner, "has", Quantity(12, Unit.of("sheep")), "measurement:sheep")
    coins = measured(plugin, owner, "holds", Quantity(5, Unit.of("coin")), "measurement:coins")
    before = tuple(plugin.mind.propositions())
    got = calculation(plugin, "sum", [sheep, coins], owner, "total")
    assert isinstance(got, Unknown)
    assert isinstance(add(sheep.role("object"), coins.role("object")), Unknown)
    assert tuple(plugin.mind.propositions()) == before


def test_scale_and_sum_are_recorded_like_any_other_operation():
    plugin = QuantityPlugin()
    field, farm = Ref("entity:field"), Ref("entity:farm")
    a = measured(plugin, field, "yield", Quantity(10, Unit.of("bushel")), "measurement:a")
    b = measured(plugin, Ref("entity:field2"), "yield", Quantity(4, Unit.of("bushel")), "measurement:b")
    total = calculation(plugin, "sum", [a, b], farm, "yield")
    assert total.proposition.role("object") == Quantity(14, Unit.of("bushel"))
    doubled = calculation(plugin, "scale", [a], field, "doubled", {"factor": 2})
    assert doubled.proposition.role("object") == Quantity(20, Unit.of("bushel"))
    assert doubled.proposition.predicate == "calculated:scale:doubled"
    assert set(doubled.premise_ids) == {a.id, b.id}
    selected = next(registration for registration in plugin.registrations
                    if registration.context == CalculationContext(field, "doubled"))
    assert selected.operand_ids == (a.id,) and selected.params == {"factor": 2}
    assert validate_record_support(plugin.mind, doubled.record_id) is True


def test_unsafe_claim_quantity_admission_apis_are_removed():
    import tensorcode.quantity as quantity
    assert not hasattr(quantity, "derive")
    assert not hasattr(quantity, "tell_quantity")
    for name in ("BASE_UNITS", "ALIASES", "normalize_unit", "_base_unit"):
        assert not hasattr(quantity, name)
    assert not hasattr(Quantity, "parse")


@pytest.mark.parametrize("left,right", [
    ("dollar", "euro"), ("dollar", "pound_sterling"), ("coin", "dollar"),
    ("cent", "dollar"), ("cent", "euro"),
])
def test_distinct_currencies_have_no_implicit_exchange_parity(left, right):
    a, b = Quantity(2, Unit.of(left)), Quantity(3, Unit.of(right))
    for result in (add(a, b), sub(a, b), compare(a, b), convert(a, b.unit), ratio(a, b)):
        assert isinstance(result, Unknown) and result.reason == "dimension_mismatch"
    assert a.unit.dimension != b.unit.dimension


def test_same_currency_arithmetic_preserves_its_supplied_symbol():
    assert add(Quantity(2, Unit.of("euro")), Quantity(3, Unit.of("euro"))) == Quantity(5, Unit.of("euro"))
    assert add(Quantity(2, Unit.of("cent")), Quantity(3, Unit.of("cent"))) == Quantity(5, Unit.of("cent"))
