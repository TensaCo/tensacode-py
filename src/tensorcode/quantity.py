"""Typed quantities over immutable products of literal unit symbols.

Additive operations require exact unit equality. Physical relationships require
explicit evidence and selected operations. Missing unit equivalence returns
:class:`~tensorcode.outcomes.Unknown`; no spelling, scale, or dimension is inferred.

    >>> sheep = Quantity(12, Unit.of("sheep"))
    >>> price = Quantity(5, Unit.of("coin") / Unit.of("sheep"))
    >>> mul(sheep, price)
    Quantity(value=60, unit=Unit(coin))
    >>> isinstance(add(sheep, price), Unknown)
    True

These functions compute values only. Retained measurements and authenticated
calculation lineage belong to the explicit quantity calculation interface.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from collections.abc import Mapping

from .outcomes import Unknown


@dataclass(frozen=True, eq=False)
class _UnitPowers(Mapping):
    """Immutable mapping with transparent tuple storage for authenticated records."""
    entries: tuple[tuple[str, int], ...]

    def __getitem__(self, key):
        for symbol, power in self.entries:
            if symbol == key:
                return power
        raise KeyError(key)

    def __iter__(self):
        return (symbol for symbol, _ in self.entries)

    def __len__(self):
        return len(self.entries)

    def __hash__(self):
        return hash(self.entries)


@dataclass(frozen=True)
class Unit:
    """A product of unit symbols with integer exponents. ``Unit()`` is dimensionless."""

    powers: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.powers, Mapping):
            raise TypeError("unit powers require a mapping of literal symbols to integer exponents")
        entries = []
        for symbol, power in self.powers.items():
            if type(symbol) is not str or not symbol or type(power) is not int:
                raise TypeError("unit symbols must be nonempty literal strings and exponents exact integers")
            if power:
                entries.append((symbol, power))
        object.__setattr__(self, "powers", _UnitPowers(tuple(sorted(entries))))

    @classmethod
    def of(cls, symbol: str, power: int = 1) -> "Unit":
        """Construct one explicitly supplied literal symbol; perform no text interpretation."""
        return cls({symbol: power})

    @property
    def dimension(self) -> tuple[tuple[str, int], ...]:
        """Formal symbol signature, with no inferred physical dimensions or conversions."""
        return tuple(self.powers.items())

    @property
    def dimensionless(self) -> bool:
        return not self.powers

    def __mul__(self, other: "Unit") -> "Unit":
        merged = Counter(self.powers)
        merged.update(other.powers)
        return Unit(dict(merged))

    def __truediv__(self, other: "Unit") -> "Unit":
        merged = Counter(self.powers)
        merged.subtract(other.powers)
        return Unit(dict(merged))

    def __pow__(self, n: int) -> "Unit":
        if type(n) is not int:
            raise TypeError("unit exponent must be an exact integer")
        return Unit({s: p * n for s, p in self.powers.items()})

    def __hash__(self) -> int:
        return hash(tuple(self.powers.items()))

    def __str__(self) -> str:
        if not self.powers:
            return ""
        parts = [s if p == 1 else f"{s}^{p}" for s, p in self.powers.items() if p > 0]
        under = [s if p == -1 else f"{s}^{-p}" for s, p in self.powers.items() if p < 0]
        head = "·".join(parts) or "1"
        return head + ("/" + "·".join(under) if under else "")

    def __repr__(self) -> str:
        return f"Unit({self})"


@dataclass(frozen=True)
class Quantity:
    """A measured value: how much, of what unit."""

    value: int | float
    unit: Unit = field(default_factory=Unit)

    def __post_init__(self) -> None:
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("quantity value must be a finite int or float, not a coerced value")
        if type(self.unit) is not Unit:
            raise TypeError("quantity requires an explicit Unit")

    @property
    def dimension(self) -> tuple[tuple[str, int], ...]:
        return self.unit.dimension

    def comparable(self, other: "Quantity") -> bool:
        return self.unit == other.unit

    def __str__(self) -> str:
        shown = f"{self.value:g}"
        return f"{shown} {self.unit}".strip()

    def __repr__(self) -> str:
        return f"Quantity(value={self.value}, unit={self.unit!r})"


# ----------------------------------------------------------------- arithmetic

MISMATCH = "dimension_mismatch"


def _mismatch(op: str, a: Quantity, b: Quantity) -> Unknown:
    return Unknown(MISMATCH, f"cannot {op} {a} and {b}: {_dim(a)} vs {_dim(b)}")


def _dim(q: Quantity) -> str:
    return "·".join(f"{d}^{p}" if p != 1 else d for d, p in q.dimension) or "dimensionless"


def add(a: Quantity, b: Quantity) -> Quantity | Unknown:
    if not a.comparable(b):
        return _mismatch("add", a, b)
    return Quantity(a.value + b.value, a.unit)


def sub(a: Quantity, b: Quantity) -> Quantity | Unknown:
    if not a.comparable(b):
        return _mismatch("subtract", a, b)
    return Quantity(a.value - b.value, a.unit)


def mul(a: Quantity, b: Quantity) -> Quantity:
    return Quantity(a.value * b.value, a.unit * b.unit)


def div(a: Quantity, b: Quantity) -> Quantity | Unknown:
    if b.value == 0:
        return Unknown("division_by_zero", f"cannot divide {a} by zero")
    return Quantity(a.value / b.value, a.unit / b.unit)


def scale(a: Quantity, factor: float) -> Quantity:
    if type(factor) not in (int, float) or not math.isfinite(factor):
        raise ValueError("scale factor must be a finite int or float")
    return Quantity(a.value * factor, a.unit)


def convert(a: Quantity, unit: Unit) -> Quantity | Unknown:
    """Identity conversion only; different units require a selected evidenced definition."""
    if type(unit) is not Unit:
        raise TypeError("conversion target requires an explicit Unit")
    if a.unit != unit:
        return _mismatch("convert without an evidenced definition", a, Quantity(1, unit))
    return a


def ratio(a: Quantity, b: Quantity) -> Quantity | Unknown:
    """A dimensionless ratio, only between comparable quantities."""
    if not a.comparable(b):
        return _mismatch("compare", a, b)
    if b.value == 0:
        return Unknown("division_by_zero", f"cannot divide {a} by zero")
    return Quantity(a.value / b.value, Unit())


def percent_of(part: Quantity, whole: Quantity) -> Quantity | Unknown:
    got = ratio(part, whole)
    return got if isinstance(got, Unknown) else Quantity(got.value * 100, Unit.of("percent"))


OPS = {"add": add, "sub": sub, "mul": mul, "div": div, "ratio": ratio, "percent_of": percent_of}


def compare(a: Quantity, b: Quantity) -> str | Unknown:
    """'greater', 'less' or 'equal' — or a refusal when the dimensions differ."""
    if not a.comparable(b):
        return _mismatch("compare", a, b)
    x, y = a.value, b.value
    if x == y:
        return "equal"
    return "greater" if x > y else "less"
