"""Legacy authored clause-link and likelihood projections.

These remaining helpers use supplied connective tables and lexical overlap;
they are not learned semantic interpretation. Quantity extraction, lexical unit
guessing, and automatic mention-to-claim identity creation were removed.
Measurements and arithmetic now require explicit evidence, literal unit identities,
and independently selected calculation operands and operations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from .outcomes import Score, Unknown
from .records import Claim, Evidence, Ref, Store
from .temporal import CONNECTIVES

#: what the grammar drops and this module recovers from the surface string instead
needed_from_grammar = (
    "conditionals return two unlinked readings; `if`/`then` is not in the parse",
    "causal connectives (`because`, `so`, `caused ... to`) parse as a noun or a plain verb",
    "temporal connectives (`before`, `after`, `while`) collapse the clauses into one frame",
    "adverbs of likelihood (`probably`) can become the predicate",
)


LIKELIHOOD = {
    "certainly": 0.95, "definitely": 0.95, "surely": 0.9, "undoubtedly": 0.95,
    "probably": 0.75, "likely": 0.75, "presumably": 0.7, "apparently": 0.6,
    "maybe": 0.5, "perhaps": 0.5, "possibly": 0.4, "might": 0.4, "could": 0.4,
    "unlikely": 0.2, "doubtfully": 0.15, "never": 0.02,
}

CAUSAL_CUES = {"because": "effect_first", "since": "effect_first", "as": "effect_first",
               "so": "cause_first", "therefore": "cause_first", "thus": "cause_first",
               "caused": "cause_first", "causes": "cause_first", "cause": "cause_first"}


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
