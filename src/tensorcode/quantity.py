"""Quantities with units, and arithmetic that refuses rather than guesses.

A number in a sentence is not a number: "3 sheep", "3 coins per sheep" and "3%" compose
differently, and adding the first two is not a small error but a category mistake. So a
quantity carries its unit, a unit carries its dimension, and every operation checks that
the dimensions line up. A mismatch returns :class:`~tensorcode.outcomes.Unknown` — the
same refusal the rest of the library uses — never a number that looks fine.

    >>> sheep = Quantity(12, Unit.of("sheep"))
    >>> price = Quantity(5, Unit.of("coin") / Unit.of("sheep"))
    >>> mul(sheep, price)
    Quantity(value=60.0, unit=Unit(coin))
    >>> isinstance(add(sheep, price), Unknown)
    True

Derivations are recorded, not just computed: :func:`derive` writes the result as a claim
whose evidence names the premises and the operation, so ``explain`` shows the working.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .outcomes import Score, Unknown
from .records import Claim, Evidence, Ref, Store

# --------------------------------------------------------------------------- units

#: base dimension per known unit symbol, with the factor into that dimension's base unit.
#: "item" is the dimension of anything counted; a bare count has no unit of its own.
BASE_UNITS: dict[str, tuple[str, float]] = {
    "item": ("item", 1.0),
    # currency
    "coin": ("currency", 1.0), "dollar": ("currency", 1.0), "cent": ("currency", 0.01),
    "euro": ("currency", 1.0), "pound_sterling": ("currency", 1.0),
    # mass
    "kilogram": ("mass", 1.0), "gram": ("mass", 0.001), "pound": ("mass", 0.45359237), "ounce": ("mass", 0.0283495),
    # volume
    "litre": ("volume", 1.0), "millilitre": ("volume", 0.001), "bushel": ("volume", 35.2391), "gallon": ("volume", 3.78541),
    # length
    "metre": ("length", 1.0), "centimetre": ("length", 0.01), "kilometre": ("length", 1000.0),
    "inch": ("length", 0.0254), "foot": ("length", 0.3048), "mile": ("length", 1609.344),
    # time
    "second": ("time", 1.0), "minute": ("time", 60.0), "hour": ("time", 3600.0),
    "day": ("time", 86400.0), "week": ("time", 604800.0), "year": ("time", 31557600.0),
}

#: surface spellings that mean a known unit. Plurals are stripped before lookup.
ALIASES: dict[str, str] = {
    "$": "dollar", "usd": "dollar", "buck": "dollar", "€": "euro", "£": "pound_sterling", "penny": "cent", "pennies": "cent",
    "kg": "kilogram", "g": "gram", "lb": "pound", "lbs": "pound", "oz": "ounce",
    "l": "litre", "ml": "millilitre", "m": "metre", "cm": "centimetre", "km": "kilometre",
    "ft": "foot", "feet": "foot", "in": "inch", "mi": "mile",
    "s": "second", "sec": "second", "min": "minute", "hr": "hour", "hrs": "hour", "h": "hour",
    "mins": "minute", "yr": "year",
}


def normalize_unit(word: str) -> str:
    """A surface word to a unit symbol. Unknown words become their own count unit."""
    w = word.strip().lower().rstrip(".")
    w = ALIASES.get(w, w)
    if w in BASE_UNITS:
        return w
    # try each way this could be a plural, and take the first that names a unit we know;
    # "minutes" is minute (not "minut"), while an unknown word keeps its singular stem
    if w.endswith(("us", "is", "ss")) or len(w) < 3:
        return w  # a singular that merely ends in s: bus, iris, glass — nothing to strip
    stems = []
    if w.endswith("ies") and len(w) > 4:
        stems.append(w[:-3] + "y")
    if w.endswith("es") and len(w) > 3:
        stems += [w[:-1], w[:-2]]
    if w.endswith("s") and len(w) > 2:
        stems.append(w[:-1])
    for stem in stems:
        candidate = ALIASES.get(stem, stem)
        if candidate in BASE_UNITS:
            return candidate
    if not stems:
        return w
    # An unknown count noun. "-es" is the plural marker after a sibilant ("boxes" -> box,
    # "glasses" -> glass), but a word ending in silent e takes a bare "-s" ("houses" ->
    # house). Testing for a doubled s keeps those apart: "hous" ends in one s and is
    # rejected, "glass" in two and is kept. Getting this wrong would file a thing's
    # singular and plural as different units and then refuse to add them together.
    if w.endswith("es") and len(w) > 3:
        stripped = w[:-2]
        return stripped if stripped.endswith(("ss", "x", "z", "ch", "sh")) else w[:-1]
    return stems[0]


@dataclass(frozen=True)
class Unit:
    """A product of unit symbols with integer exponents. ``Unit()`` is dimensionless."""

    powers: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "powers", {k: v for k, v in sorted(self.powers.items()) if v})

    @classmethod
    def of(cls, symbol: str, power: int = 1) -> "Unit":
        return cls({normalize_unit(symbol): power}) if symbol else cls()

    @property
    def dimension(self) -> tuple[tuple[str, int], ...]:
        """The dimension signature: what may be added to what."""
        dims: Counter[str] = Counter()
        for symbol, power in self.powers.items():
            dims[BASE_UNITS.get(symbol, ("item", 1.0))[0] if symbol in BASE_UNITS else f"count:{symbol}"] += power
        return tuple(sorted((d, p) for d, p in dims.items() if p))

    @property
    def dimensionless(self) -> bool:
        return not self.powers

    def factor(self) -> float:
        """Scale into base units of each dimension, so comparable quantities compare."""
        out = 1.0
        for symbol, power in self.powers.items():
            out *= BASE_UNITS.get(symbol, (None, 1.0))[1] ** power
        return out

    def __mul__(self, other: "Unit") -> "Unit":
        merged = Counter(self.powers)
        merged.update(other.powers)
        return Unit(dict(merged))

    def __truediv__(self, other: "Unit") -> "Unit":
        merged = Counter(self.powers)
        merged.subtract(other.powers)
        return Unit(dict(merged))

    def __pow__(self, n: int) -> "Unit":
        return Unit({s: p * n for s, p in self.powers.items()})

    def __hash__(self) -> int:
        """Hashable, because a quantity ends up inside a set.

        ``powers`` is a dict, and a frozen dataclass hashes its fields, so the generated
        ``__hash__`` raised ``unhashable type: 'dict'``. That only surfaces once a
        :class:`~tensorcode.records.Claim` carries a :class:`Quantity` as its object: the
        agent's retrieval builds ``{claim.subject, claim.object}`` to check that everything
        the question bound appears in the claim, and the whole lookup died with a
        ``TypeError`` — which is to say a plugin could record a quantity but the agent could
        never read one back. The powers are already normalized and sorted in
        ``__post_init__``, so the tuple of items is a faithful key.
        """
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

    value: float
    unit: Unit = field(default_factory=Unit)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))

    @classmethod
    def parse(cls, value: float, unit_word: str | None = None, power: int = 1) -> "Quantity":
        return cls(value, Unit.of(unit_word, power) if unit_word else Unit())

    @property
    def dimension(self) -> tuple[tuple[str, int], ...]:
        return self.unit.dimension

    def base(self) -> float:
        """The value in base units, for comparison across spellings of one dimension."""
        return self.value * self.unit.factor()

    def comparable(self, other: "Quantity") -> bool:
        return self.dimension == other.dimension

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
    return Quantity(a.base() + b.base(), _base_unit(a.unit)) if a.unit != b.unit else Quantity(a.value + b.value, a.unit)


def sub(a: Quantity, b: Quantity) -> Quantity | Unknown:
    if not a.comparable(b):
        return _mismatch("subtract", a, b)
    return Quantity(a.base() - b.base(), _base_unit(a.unit)) if a.unit != b.unit else Quantity(a.value - b.value, a.unit)


def mul(a: Quantity, b: Quantity) -> Quantity:
    return Quantity(a.value * b.value, a.unit * b.unit)


def div(a: Quantity, b: Quantity) -> Quantity | Unknown:
    if b.value == 0:
        return Unknown("division_by_zero", f"cannot divide {a} by zero")
    return Quantity(a.value / b.value, a.unit / b.unit)


def scale(a: Quantity, factor: float) -> Quantity:
    return Quantity(a.value * factor, a.unit)


def convert(a: Quantity, unit: Unit) -> Quantity | Unknown:
    """The same amount said in another unit of the same dimension.

    ``add`` already rescales when two spellings of one dimension meet, but it picks the
    dimension's base unit, so asking for "45 minutes" back gave "2700 second". A question
    names the unit it wants its answer in ("how many minutes …"), and answering in a
    different one is a wrong answer however right the number is. Refuses across dimensions
    and refuses a unit whose scale is zero, rather than returning something plausible.
    """
    if a.dimension != unit.dimension:
        return _mismatch("convert", a, Quantity(1.0, unit))
    factor = unit.factor()
    if factor == 0:
        return Unknown("unscalable_unit", f"cannot express {a} in {unit}: that unit has no scale")
    return Quantity(a.base() / factor, unit)


def ratio(a: Quantity, b: Quantity) -> Quantity | Unknown:
    """A dimensionless ratio, only between comparable quantities."""
    if not a.comparable(b):
        return _mismatch("compare", a, b)
    if b.base() == 0:
        return Unknown("division_by_zero", f"cannot divide {a} by zero")
    return Quantity(a.base() / b.base(), Unit())


def percent_of(part: Quantity, whole: Quantity) -> Quantity | Unknown:
    got = ratio(part, whole)
    return got if isinstance(got, Unknown) else Quantity(got.value * 100, Unit.of("percent"))


def _base_unit(unit: Unit) -> Unit:
    """The base spelling of each dimension in a unit, used when two spellings are added."""
    out: Counter[str] = Counter()
    for symbol, power in unit.powers.items():
        dim = BASE_UNITS.get(symbol, (None, None))[0]
        base = next((s for s, (d, f) in BASE_UNITS.items() if d == dim and f == 1.0), symbol) if dim else symbol
        out[base] += power
    return Unit(dict(out))


OPS = {"add": add, "sub": sub, "mul": mul, "div": div, "ratio": ratio, "percent_of": percent_of}


def compare(a: Quantity, b: Quantity) -> str | Unknown:
    """'greater', 'less' or 'equal' — or a refusal when the dimensions differ."""
    if not a.comparable(b):
        return _mismatch("compare", a, b)
    x, y = a.base(), b.base()
    if math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12):
        return "equal"
    return "greater" if x > y else "less"


# --------------------------------------------------------------------- claims


def tell_quantity(mind: Store, subject: Ref, predicate: str, quantity: Quantity, *, source: Ref,
                  observed_at: datetime | None = None, method: str = "quantity", confidence: Score | None = None) -> Claim:
    """Record a quantity as a claim, keeping the unit with the number."""
    claim = Claim(subject, predicate, quantity)
    mind.tell(claim, Evidence(source=source, observed_at=observed_at or datetime.now(timezone.utc), method=method, confidence=confidence))
    return claim


def derive(mind: Store, subject: Ref, predicate: str, op: str, premises: Iterable[Claim], *,
           source: Ref | None = None, observed_at: datetime | None = None,
           factor: float | None = None) -> Claim | Unknown:
    """Compute ``op`` over the premises' quantities and record the result with its working.

    The evidence names the operation and the premise claim ids, so ``explain`` shows the
    arithmetic and retracting a premise withdraws the conclusion.
    """
    premises = list(premises)
    values = [p.object for p in premises]
    if not all(isinstance(v, Quantity) for v in values):
        return Unknown("not_quantities", f"{op} needs quantities, got {[type(v).__name__ for v in values]}")
    if op == "scale":
        if factor is None or len(values) != 1:
            return Unknown("bad_arity", "scale takes one quantity and a factor")
        result: Any = scale(values[0], factor)
    elif op in OPS:
        if len(values) != 2:
            return Unknown("bad_arity", f"{op} takes two quantities, got {len(values)}")
        result = OPS[op](values[0], values[1])
    elif op == "sum":
        result = values[0]
        for v in values[1:]:
            result = add(result, v)
            if isinstance(result, Unknown):
                break
    else:
        return Unknown("unknown_operation", op)
    if isinstance(result, Unknown):
        return result
    claim = Claim(subject, predicate, result)
    mind.tell(claim, Evidence(
        source=source or Ref("reasoning:arithmetic"),
        observed_at=observed_at or datetime.now(timezone.utc),
        method=f"arithmetic:{op}" + (f"×{factor:g}" if factor is not None else ""),
        derived_from=tuple(p.id for p in premises),
    ))
    return claim
