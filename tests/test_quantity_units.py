"""Quantities: units carried, dimensions checked, arithmetic shown."""

import pytest

from datetime import datetime, timezone

from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.derivations import validate_record_support
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.outcomes import Unknown
from tensorcode.quantity import (BASE_UNITS, Quantity, Unit, add, compare, convert, div, mul, normalize_unit,
                                percent_of, ratio, scale, sub)
from tensorcode.records import Evidence, Ref


def test_adding_different_dimensions_is_refused_not_approximated():
    sheep = Quantity(12, Unit.of("sheep"))
    price = Quantity(5, Unit.of("coin") / Unit.of("sheep"))
    refused = add(sheep, price)
    assert isinstance(refused, Unknown) and refused.reason == "dimension_mismatch"
    assert "count:sheep" in refused.detail and "currency" in refused.detail
    # the same pair multiplies perfectly well, and the unit says what came out
    assert str(mul(sheep, price)) == "60 coin"


def test_one_dimension_spelled_two_ways_still_adds():
    total = add(Quantity(1, Unit.of("kg")), Quantity(500, Unit.of("gram")))
    assert isinstance(total, Quantity) and total.value == pytest.approx(1.5)
    assert compare(Quantity(1, Unit.of("kg")), Quantity(1, Unit.of("lb"))) == "greater"


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


@pytest.mark.parametrize("singular,plural", [
    ("box", "boxes"), ("glass", "glasses"), ("house", "houses"), ("sheep", "sheep"),
    ("coin", "coins"), ("minute", "minutes"), ("bush", "bushes"), ("penny", "pennies"),
    ("apple", "apples"), ("inch", "inches"), ("bushel", "bushels"),
])
def test_a_things_singular_and_plural_are_the_same_unit(singular, plural):
    """Otherwise two mentions of one thing would refuse to add to each other."""
    assert normalize_unit(singular) == normalize_unit(plural)


def test_the_one_plural_this_cannot_settle_is_recorded_not_hidden():
    """English spelling leaves "bus"/"buses" genuinely ambiguous: bus+es or buse+s.

    Stripping "-es" after a single s would break "houses" (hous); not stripping it breaks
    "buses" (buse). This records the residual rather than pretending the rule is complete.
    """
    assert normalize_unit("bus") == "bus"
    assert normalize_unit("buses") == "buse"  # known-wrong, and it is one word, not a class


def test_a_known_unit_beats_the_plural_rule():
    assert normalize_unit("minutes") == "minute" and normalize_unit("feet") == "foot"
    assert all(normalize_unit(u) == u for u in BASE_UNITS)


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
