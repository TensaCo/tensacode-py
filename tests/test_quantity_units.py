"""Quantities: units carried, dimensions checked, arithmetic shown."""

import pytest

from tensacode.cognition import explain
from tensacode.outcomes import Unknown
from tensacode.quantity import (BASE_UNITS, Quantity, Unit, add, compare, derive, div, mul, normalize_unit,
                                percent_of, ratio, scale, sub, tell_quantity)
from tensacode.records import Ref, Store


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


def test_a_derivation_records_its_working_and_falls_with_its_premises():
    mind, source = Store(), Ref("obs:note")
    held = tell_quantity(mind, Ref("entity:Anem"), "has", Quantity(12, Unit.of("sheep")), source=source)
    price = tell_quantity(mind, Ref("entity:market"), "price", Quantity(5, Unit.of("coin") / Unit.of("sheep")), source=source)
    revenue = derive(mind, Ref("entity:Anem"), "revenue", "mul", [held, price])
    assert revenue.object == Quantity(60, Unit.of("coin"))
    lines = "\n".join(explain(mind, revenue.id))
    assert "arithmetic:mul" in lines, lines  # the operation, not just the premises
    assert "60 coin" in lines and "5 coin/sheep" in lines, lines  # units survive into the explanation
    mind.apply(__import__("tensacode").Patch((__import__("tensacode").Retract(price.id, "price withdrawn"),), mind.revision))
    assert mind.claims(Ref("entity:Anem"), "revenue") == []


def test_a_derivation_over_mismatched_units_refuses_and_records_nothing():
    mind, source = Store(), Ref("obs:note")
    sheep = tell_quantity(mind, Ref("entity:Anem"), "has", Quantity(12, Unit.of("sheep")), source=source)
    coins = tell_quantity(mind, Ref("entity:Anem"), "holds", Quantity(5, Unit.of("coin")), source=source)
    got = derive(mind, Ref("entity:Anem"), "total", "add", [sheep, coins])
    assert isinstance(got, Unknown) and got.reason == "dimension_mismatch"
    assert mind.claims(Ref("entity:Anem"), "total") == []


def test_scale_and_sum_are_recorded_like_any_other_operation():
    mind, source = Store(), Ref("obs:note")
    a = tell_quantity(mind, Ref("entity:field"), "yield", Quantity(10, Unit.of("bushel")), source=source)
    b = tell_quantity(mind, Ref("entity:field2"), "yield", Quantity(4, Unit.of("bushel")), source=source)
    assert derive(mind, Ref("entity:farm"), "yield", "sum", [a, b]).object == Quantity(14, Unit.of("bushel"))
    doubled = derive(mind, Ref("entity:field"), "doubled", "scale", [a], factor=2)
    assert doubled.object == Quantity(20, Unit.of("bushel"))
    assert "×2" in "\n".join(explain(mind, doubled.id))
