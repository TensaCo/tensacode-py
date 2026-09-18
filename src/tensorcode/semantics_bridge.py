"""From a parsed sentence to the structures that make it usable.

The grammar already recovers more than the projection keeps. "Anem has 12 sheep" parses
with ``count=Entity(number, '12')`` inside the noun's features, and
:func:`tensorcode.language.to_claims` then emits ``Anem have sheep`` — the number is
dropped on the floor. Conditionals and causal connectives fare worse: both clauses parse,
but they come back as two unrelated readings with the connective gone, so "if I press Send
an announcement appears" is indistinguishable from two separate remarks.

This module is the projection those structures deserve:

* :func:`quantities_in` lifts numbers and their units out of the parse into
  :class:`~tensorcode.quantity.Quantity` values;
* :func:`link_in` recovers the connective from the surface string and pairs the readings it
  joined, giving a conditional, a causal or a temporal link;
* :func:`probability_in` reads adverbs of likelihood as an *uncalibrated* score, because
  "probably" is not a calibrated number and must not pretend to be.

Where it reads the surface string rather than the parse, that is a workaround for a gap in
the grammar, not a design choice; ``needed_from_grammar`` lists them so the gaps stay
visible instead of becoming permanent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .outcomes import Score, Unknown
from .quantity import Quantity, Unit, normalize_unit
from .records import Claim, Evidence, Ref, Store
from .temporal import CONNECTIVES

#: what the grammar drops and this module recovers from the surface string instead
needed_from_grammar = (
    "numerals are kept only as a `count` feature on the noun; to_claims discards them",
    "conditionals return two unlinked readings; `if`/`then` is not in the parse",
    "causal connectives (`because`, `so`, `caused ... to`) parse as a noun or a plain verb",
    "temporal connectives (`before`, `after`, `while`) collapse the clauses into one frame",
    "comparatives (`twice as much as`) drop below full coverage and lose the multiplier",
    "adverbs of likelihood (`probably`) can become the predicate",
)

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000,
    "half": 0.5, "quarter": 0.25, "twice": 2, "double": 2, "triple": 3,
}

LIKELIHOOD = {
    "certainly": 0.95, "definitely": 0.95, "surely": 0.9, "undoubtedly": 0.95,
    "probably": 0.75, "likely": 0.75, "presumably": 0.7, "apparently": 0.6,
    "maybe": 0.5, "perhaps": 0.5, "possibly": 0.4, "might": 0.4, "could": 0.4,
    "unlikely": 0.2, "doubtfully": 0.15, "never": 0.02,
}

CAUSAL_CUES = {"because": "effect_first", "since": "effect_first", "as": "effect_first",
               "so": "cause_first", "therefore": "cause_first", "thus": "cause_first",
               "caused": "cause_first", "causes": "cause_first", "cause": "cause_first"}

MONEY = {"$": "dollar", "€": "euro", "£": "pound_sterling"}
_NUMBER = re.compile(r"(?P<sym>[$€£])?\s*(?P<num>\d+(?:[.,]\d+)?)\s*(?P<pct>%)?\s*(?P<unit>[a-zA-Z][a-zA-Z_-]*)?")


@dataclass(frozen=True)
class Mention:
    """A quantity as it was said, with whatever the sentence attached it to."""

    quantity: Quantity
    of: str  # the noun it counted or measured, as written
    owner: str | None = None  # the subject it was predicated of, when there is one
    predicate: str | None = None
    per: str | None = None  # the unit it was stated per, for rates

    def describe(self) -> str:
        head = f"{self.quantity}"
        of = "" if normalize_unit(self.of) in str(self.quantity.unit) else f" of {self.of}"
        return head + of + (f" ({self.owner} {self.predicate})" if self.owner else "")


@dataclass(frozen=True)
class Link:
    """Two clauses and the relation the connective asserted between them."""

    kind: str  # "conditional" | "causal" | "temporal"
    relation: str  # "if_then" | "causes" | "before" | "after" | "during"
    antecedent: Any  # a Frame (the if-side, the cause, the earlier event)
    consequent: Any  # a Frame (the then-side, the effect, the later event)
    cue: str

    def describe(self) -> str:
        left = getattr(self.antecedent, "describe", lambda: str(self.antecedent))()
        right = getattr(self.consequent, "describe", lambda: str(self.consequent))()
        return f"{self.kind}({self.relation}): {left} ⇒ {right} [“{self.cue}”]"


# ----------------------------------------------------------------- quantities


def _number_of(entity: Any) -> float | None:
    text = getattr(entity, "text", None)
    if text is None:
        return None
    word = str(text).strip().lower()
    if word in WORD_NUMBERS:
        return float(WORD_NUMBERS[word])
    try:
        return float(word.replace(",", ""))
    except ValueError:
        return None


def quantities_in(frame: Any) -> list[Mention]:
    """Every quantity the parse carries, with its unit and what it was said about.

    Reads the ``count`` feature the grammar puts on a noun — the thing ``to_claims`` throws
    away — and the ``per`` phrasing that makes a rate.
    """
    out: list[Mention] = []
    for sub in frame.walk() if hasattr(frame, "walk") else [frame]:
        subject = sub.roles.get("subject") if hasattr(sub, "roles") else None
        owner = getattr(subject, "text", None)
        for role, value in (sub.roles.items() if hasattr(sub, "roles") else ()):
            for entity in _entities(value):
                count = entity.features.get("count") if hasattr(entity, "features") else None
                amount = _number_of(count) if count is not None else None
                if amount is None:
                    continue
                # the noun feature carries the head word; the surface text carries a rate
                # phrase ("coins per bushel"), which is the part that fixes the unit
                written = str(entity.text or "")
                noun = written if re.search(r"\bper\b|/", written) else str(entity.features.get("noun") or written)
                head, per = _split_rate(noun)
                unit = Unit.of(head)
                if per:
                    unit = unit / Unit.of(per)
                out.append(Mention(Quantity(amount, unit), of=head, owner=owner if role != "subject" else None,
                                   predicate=getattr(sub, "predicate", None), per=per))
    return out


def _entities(value: Any, depth: int = 0) -> Iterable[Any]:
    if depth > 6:
        return
    if hasattr(value, "features") and hasattr(value, "text"):
        yield value
        for inner in value.features.values():
            yield from _entities(inner, depth + 1)
    elif isinstance(value, tuple):
        for item in value:
            yield from _entities(item, depth + 1)


def _split_rate(noun: str) -> tuple[str, str | None]:
    parts = re.split(r"\s+per\s+|\s*/\s*", noun.strip(), maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return noun.strip(), None


def quantities_in_text(text: str) -> list[Mention]:
    """Quantities read straight off the string, for sentences the grammar does not cover.

    Used by the arithmetic evaluation, where coverage matters more than structure. It is a
    weaker reader than :func:`quantities_in` — it does not know what owns what.
    """
    out: list[Mention] = []
    for match in _NUMBER.finditer(text):
        raw = match.group("num").replace(",", "")
        try:
            amount = float(raw)
        except ValueError:
            continue
        if match.group("pct"):
            out.append(Mention(Quantity(amount, Unit.of("percent")), of="percent"))
            continue
        symbol, word = match.group("sym"), match.group("unit")
        if symbol:
            out.append(Mention(Quantity(amount, Unit.of(MONEY[symbol])), of=MONEY[symbol]))
            continue
        noun = normalize_unit(word) if word else "item"
        out.append(Mention(Quantity(amount, Unit.of(noun)), of=noun))
    return out


# ---------------------------------------------------------------------- links


def _overlap(frame: Any, span: str) -> int:
    """How much of a frame's wording appears in a stretch of the sentence."""
    words = set(re.findall(r"[a-z0-9]+", span.lower()))
    score = 0
    predicate = str(getattr(frame, "predicate", "") or "").lower()
    if predicate and any(w.startswith(predicate[:4]) for w in words if len(predicate) >= 4):
        score += 2
    for entity in getattr(frame, "entities", lambda: ())():
        text = str(getattr(entity, "text", "")).lower()
        if text and text in words:
            score += 1
    return score


def link_in(text: str, meanings: Sequence[Any]) -> Link | Unknown:
    """The relation a connective asserted, and which reading sits on each side.

    The connective comes from the string because the grammar drops it. Which clause is
    which comes from matching each reading's wording against the two halves of the
    sentence, so the direction of "because" is read, not assumed.
    """
    frames = [m for m in meanings if hasattr(m, "predicate")]
    if len(frames) < 2:
        return Unknown("one_reading", "a link needs two clauses; the parse gave fewer")
    lowered = text.lower()

    for cue in ("if",):
        match = re.search(rf"\b{cue}\b", lowered)
        if match:
            left, right = _sides(text, match.end(), r"\bthen\b")
            a, b = _assign(frames, left, right)
            return Link("conditional", "if_then", a, b, cue)

    for cue, orientation in CAUSAL_CUES.items():
        match = re.search(rf"\b{cue}\b", lowered)
        if match:
            before_cue, after_cue = text[: match.start()], text[match.end():]
            first, second = _assign(frames, before_cue, after_cue)
            cause, effect = (second, first) if orientation == "effect_first" else (first, second)
            return Link("causal", "causes", cause, effect, cue)

    for cue in CONNECTIVES:
        match = re.search(rf"\b{cue}\b", lowered)
        if match:
            relation = CONNECTIVES[cue]
            before_cue, after_cue = text[: match.start()], text[match.end():]
            first, second = _assign(frames, before_cue, after_cue)
            if relation == "before":
                return Link("temporal", "before", first, second, cue)
            if relation == "after":
                return Link("temporal", "before", second, first, cue)
            return Link("temporal", "during", first, second, cue)
    return Unknown("no_connective", "no linking word found in the sentence")


def _sides(text: str, start: int, closer: str) -> tuple[str, str]:
    rest = text[start:]
    match = re.search(closer, rest.lower())
    return (rest[: match.start()], rest[match.end():]) if match else (rest, text[:start])


def _assign(frames: Sequence[Any], left: str, right: str) -> tuple[Any, Any]:
    """Put the reading that matches the left span on the left, the other on the right."""
    best_left = max(frames, key=lambda f: (_overlap(f, left), -_overlap(f, right)))
    rest = [f for f in frames if f is not best_left] or list(frames)
    best_right = max(rest, key=lambda f: _overlap(f, right))
    return best_left, best_right


# ---------------------------------------------------------------- likelihood


def probability_in(text: str) -> Score | Unknown:
    """An adverb of likelihood as an uncalibrated score, never a calibrated probability."""
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in LIKELIHOOD:
            return Score(LIKELIHOOD[word], "uncalibrated", basis="")
    return Unknown("no_likelihood_adverb", "nothing in the sentence states how likely it is")


# -------------------------------------------------------------------- claims


def tell_mentions(mind: Store, mentions: Iterable[Mention], *, source: Ref, subject: Ref | None = None,
                  observed_at: datetime | None = None, method: str = "bridge:quantity") -> list[Claim]:
    """Record quantities as claims, keeping each number with its unit."""
    at = observed_at or datetime.now(timezone.utc)
    out: list[Claim] = []
    for mention in mentions:
        owner = subject or (Ref(f"entity:{mention.owner}") if mention.owner else Ref(f"entity:{mention.of}"))
        predicate = mention.predicate or "amount"
        claim = Claim(owner, predicate, mention.quantity)
        mind.tell(claim, Evidence(source=source, observed_at=at, method=method))
        out.append(claim)
    return out


def tell_link(mind: Store, link: Link, *, source: Ref, observed_at: datetime | None = None) -> Ref:
    """Record a link between clauses as a claim about the relation itself.

    A conditional is *not* asserted as its consequent: "if I press Send an announcement
    appears" must never enter the world as "an announcement appears".
    """
    from .language.semantics import _mint  # the same event identity the projection uses

    at = observed_at or datetime.now(timezone.utc)
    evidence = Evidence(source=source, observed_at=at, method=f"bridge:{link.kind}")
    antecedent, consequent = _mint(link.antecedent), _mint(link.consequent)
    ref = Ref(f"link:{antecedent.id.split(':')[1]}-{consequent.id.split(':')[1]}")
    mind.tell(Claim(ref, "is_a", link.kind), evidence)
    mind.tell(Claim(ref, "relation", link.relation), evidence)
    mind.tell(Claim(ref, "antecedent", antecedent), evidence)
    mind.tell(Claim(ref, "consequent", consequent), evidence)
    mind.tell(Claim(ref, "cue", link.cue), evidence)
    mind.tell(Claim(antecedent, "is_a", getattr(link.antecedent, "predicate", "event")), evidence)
    mind.tell(Claim(consequent, "is_a", getattr(link.consequent, "predicate", "event")), evidence)
    return ref
