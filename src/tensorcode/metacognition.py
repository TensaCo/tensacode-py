"""Knowing what one knows: competence, decomposed confidence, repair, and agency.

Four faculties that a system needs about *itself*, each built because a measurement said
the thing it replaces does not work:

* :class:`SelfModel` — competence per *kind* of thing attempted, learned from outcomes and
  consulted before trying. This is not the per-item capability router that
  ``docs/revival/13`` measured at AUC 0.55 (SQuAD) and 0.46 (HotpotQA, below chance): that
  asked "will I get *this item* right" from features of the item. A competence prior asks
  the cheaper question "how do I do on *this kind of thing*", which needs no per-item
  signal at all — only that the kind is knowable before attempting, and that accuracy
  actually varies by kind. Where it does not vary, the prior is worthless, and
  :func:`SelfModel.competence` says so rather than inventing a number.

* :class:`Confidences` — one scalar cannot say "I am sure what you asked for but unsure
  which file you meant" (``docs/revival/15`` §15.8). Confidence is carried per *commitment*,
  so a gate can act on the part that is weak: an uncertain slot becomes a question about
  that slot instead of a refusal of the whole request.

* :class:`Monitor` — noticing "that did not work", naming the failure, and choosing a
  repair from a repertoire. The pathology it exists to stop is measured: a teacher loop
  that re-issued an identical failing command three times, and an assistant that stopped
  dead rather than trying differently.

* :func:`attribute` — every observed change is mine (predicted from my own action) or the
  world's (unpredicted). Without this distinction surprise is meaningless, because every
  consequence of one's own action looks like news.

Nothing here introspects on a model's own posterior: ``docs/revival/15`` measured that at
+0.9 points of selective accuracy for 2.8% refusals, with 446 of 464 items in the top
confidence bin. The signals here are external — outcome history, tier disagreement,
observed failure, and efference copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Iterable, Literal, Mapping, Sequence

from .outcomes import Score, Unknown, Verdict

MIN_TRIALS = 8  # below this a rate is not a competence estimate; back off to a coarser kind


# --------------------------------------------------------------- competence


def _wilson_lower(correct: int, attempts: int, z: float = 1.96) -> float:
    """The conservative end of a Wilson interval: what I can claim, not what I hope."""
    if attempts == 0:
        return 0.0
    p = correct / attempts
    d = 1 + z * z / attempts
    centre = p + z * z / (2 * attempts)
    spread = z * sqrt(p * (1 - p) / attempts + z * z / (4 * attempts * attempts))
    return max(0.0, (centre - spread) / d)


@dataclass(frozen=True)
class Competence:
    """How I have done at one kind of thing, and how sure that estimate is."""

    kind: str
    attempts: int
    correct: int
    backed_off_from: str | None = None  # the finer kind that had too little history

    @property
    def rate(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.0

    @property
    def lower(self) -> float:
        return _wilson_lower(self.correct, self.attempts)

    def score(self) -> Score:
        """The conservative estimate, as a probability whose basis names the history."""
        basis = f"self-model:{self.kind}@{self.attempts}"
        if self.backed_off_from:
            basis += f" (backed off from {self.backed_off_from})"
        return Score(self.lower, "probability", basis=basis)

    def describe(self) -> str:
        via = f", via {self.backed_off_from}" if self.backed_off_from else ""
        return f"{self.kind}: {self.correct}/{self.attempts} = {self.rate:.3f} (lower {self.lower:.3f}{via})"


@dataclass
class SelfModel:
    """Outcome history per kind, with backoff from a specific kind to a coarser family.

    ``kinds`` are caller-chosen strings ordered specific-to-general, e.g.
    ``("squad2/how_many", "squad2", "extractive_qa")``. Backoff is what makes the model
    usable on a kind it has never seen: a never-attempted question type still inherits the
    dataset's rate, and an unknown dataset inherits the family's.
    """

    name: str = "self-model"
    attempts: dict[str, int] = field(default_factory=dict)
    correct: dict[str, int] = field(default_factory=dict)

    def record(self, kinds: Sequence[str], ok: bool) -> None:
        """Credit an outcome to every level of the hierarchy at once."""
        for kind in kinds:
            self.attempts[kind] = self.attempts.get(kind, 0) + 1
            self.correct[kind] = self.correct.get(kind, 0) + (1 if ok else 0)

    def competence(self, kinds: Sequence[str], *, min_trials: int = MIN_TRIALS) -> Competence | Unknown:
        """The most specific kind with enough history to speak for, or ``Unknown``."""
        finest = kinds[0] if kinds else "?"
        for i, kind in enumerate(kinds):
            n = self.attempts.get(kind, 0)
            if n >= min_trials:
                return Competence(kind, n, self.correct.get(kind, 0), backed_off_from=finest if i else None)
        return Unknown("no_competence_history", f"fewer than {min_trials} attempts at any of {list(kinds)}")

    def worth_attempting(self, kinds: Sequence[str], *, floor: float, **kw: Any) -> Verdict:
        """Should I try? ``unknown`` when I have no history — which is not the same as no.

        A caller that treats ``unknown`` as a refusal will never attempt anything new; one
        that treats it as permission is merely uninformed. The distinction is the point.
        """
        got = self.competence(kinds, **kw)
        if isinstance(got, Unknown):
            return Verdict("unknown", (got.detail,))
        if got.lower >= floor:
            return Verdict("holds", (f"{got.describe()} at or above floor {floor:.2f}",))
        return Verdict("fails", (f"{got.describe()} below floor {floor:.2f}",))

    def table(self, *, min_trials: int = MIN_TRIALS) -> list[Competence]:
        rows = [Competence(k, n, self.correct.get(k, 0)) for k, n in self.attempts.items() if n >= min_trials]
        return sorted(rows, key=lambda c: c.lower)

    def spread(self, kinds: Iterable[str], *, min_trials: int = MIN_TRIALS) -> float:
        """How much competence varies across these kinds — how much a prior could buy.

        Near zero means every kind is alike and a competence prior is worthless however
        well estimated. This is the number to look at *before* building a gate.
        """
        rates = [self.correct.get(k, 0) / self.attempts[k] for k in kinds if self.attempts.get(k, 0) >= min_trials]
        return max(rates) - min(rates) if len(rates) > 1 else 0.0


# ------------------------------------------------------ decomposed confidence

Commitment = Literal["speech_act", "act", "slot", "value", "evidence"]
#: what a gate should do about the weakest commitment of each kind
REPAIRABLE: dict[str, str] = {"slot": "ask", "value": "refuse", "act": "ask", "speech_act": "refuse", "evidence": "refuse"}


@dataclass(frozen=True)
class Belief:
    """One thing the system has committed to, with its own confidence."""

    commitment: Commitment
    about: str            # which slot / which act / which value
    value: Any
    score: Score | None = None  # None means "no confidence reported", not zero

    @property
    def strength(self) -> float:
        return self.score.value if self.score else 0.0

    def describe(self) -> str:
        s = f"{self.score.value:.2f}" if self.score else "unreported"
        return f"{self.commitment}:{self.about}={self.value!r} ({s})"


@dataclass(frozen=True)
class Gate:
    """What to do, and about what."""

    decision: Literal["act", "ask", "refuse"]
    about: str = ""
    why: str = ""

    def describe(self) -> str:
        target = f" about {self.about}" if self.about else ""
        return f"{self.decision}{target}: {self.why}"


@dataclass(frozen=True)
class Confidences:
    """Confidence per commitment, and a gate that acts on the weakest one."""

    parts: tuple[Belief, ...]

    def weakest(self) -> Belief | None:
        return min(self.parts, key=lambda b: b.strength) if self.parts else None

    def of(self, commitment: str) -> tuple[Belief, ...]:
        return tuple(b for b in self.parts if b.commitment == commitment)

    def gate(self, *, act_at: float, ask_below: float) -> Gate:
        """Act when every commitment is strong; otherwise repair the weakest.

        The difference from a single scalar: a weak *slot* asks a question about that slot,
        while a weak *value* or unreadable *speech act* refuses. A flat gate cannot tell
        those apart and refuses all three.
        """
        weak = self.weakest()
        if weak is None:
            return Gate("refuse", why="nothing was committed to")
        if weak.strength >= act_at:
            return Gate("act", why=f"weakest commitment {weak.describe()} at or above {act_at:.2f}")
        how = REPAIRABLE.get(weak.commitment, "refuse")
        if how == "ask" and weak.strength < ask_below:
            return Gate("refuse", weak.about, f"{weak.describe()} too weak even to ask about")
        if how == "ask":
            return Gate("ask", weak.about, f"{weak.describe()} is the weak part; the rest is fine")
        return Gate("refuse", weak.about, f"{weak.describe()} below {act_at:.2f} and not a question I can ask")


def agreement(readings: Sequence[Any], *, basis: str) -> Score:
    """Confidence from independent readers agreeing, not from one reader's posterior.

    ``docs/revival/15`` measured a trained model's own posterior as nearly useless for
    abstention (+0.9 points for 2.8% refusals). Disagreement between differently-built
    readers is an external signal about the same commitment.
    """
    kept = [r for r in readings if r is not None]
    if not kept:
        return Score(0.0, "vote_share", basis=f"{basis}: nothing read")
    top = max(kept.count(r) for r in kept)
    return Score(top / len(readings), "vote_share", basis=f"{basis}: {top}/{len(readings)} readers agree")


# -------------------------------------------------------- error and repair

FailureKind = Literal[
    "no_effect",         # the act went through and nothing changed
    "error_output",      # the environment said no
    "target_missing",    # what I aimed at is not there
    "stale_reference",   # it was there and has moved or changed
    "timeout",           # it never finished
    "wrong_result",      # it finished and the result is not what was expected
    "crashed",           # my own machinery broke
]
RepairKind = Literal["retry", "retry_differently", "reperceive", "ask", "give_up"]

#: first repair to consider per failure, before history is taken into account
FIRST_REPAIR: dict[str, RepairKind] = {
    "no_effect": "retry_differently",
    "error_output": "retry_differently",
    "target_missing": "ask",
    "stale_reference": "reperceive",
    "timeout": "retry",
    "wrong_result": "retry_differently",
    "crashed": "give_up",
}


@dataclass(frozen=True)
class Attempt:
    action: str
    failure: FailureKind | None = None
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.failure is not None


@dataclass(frozen=True)
class Repair:
    kind: RepairKind
    why: str
    avoid: tuple[str, ...] = ()  # actions already known not to work

    def describe(self) -> str:
        skip = f" (not {', '.join(self.avoid)})" if self.avoid else ""
        return f"{self.kind}: {self.why}{skip}"


@dataclass
class Monitor:
    """A record of what has already failed, and a repertoire of what to do next.

    The one rule it will not break: never propose repeating an action that has already
    failed the same way. That is the measured pathology — an identical failing command
    re-issued three times — and it is a property of the *repertoire*, not of any model.
    """

    history: list[Attempt] = field(default_factory=list)
    patience: int = 2  # identical-failure retries allowed before escalating the repair

    def note(self, action: str, failure: FailureKind | None = None, detail: str = "") -> Attempt:
        attempt = Attempt(action, failure, detail)
        self.history.append(attempt)
        return attempt

    def failed_before(self, action: str, failure: FailureKind | None = None) -> int:
        return sum(1 for a in self.history if a.action == action and a.failed and (failure is None or a.failure == failure))

    def dead_ends(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(a.action for a in self.history if a.failed))

    def repair(self, action: str, failure: FailureKind, *, detail: str = "") -> Repair:
        """Choose a repair, escalating as the same failure recurs."""
        seen = self.failed_before(action, failure)
        first = FIRST_REPAIR.get(failure, "retry_differently")
        avoid = self.dead_ends()
        if seen > self.patience:
            return Repair("give_up", f"{action!r} failed {seen}x with {failure}; no repair left to try", avoid)
        if first == "retry" and seen >= 1:
            return Repair("retry_differently", f"{action!r} already failed {seen}x with {failure}; a plain retry is the same act", avoid)
        if first == "retry_differently" and seen >= self.patience:
            return Repair("ask", f"{action!r} failed {seen}x with {failure}; I cannot find a different way alone", avoid)
        why = f"{failure}" + (f": {detail}" if detail else "")
        return Repair(first, why, avoid)


# ---------------------------------------------------------------- agency


Attribution = Literal["self", "world", "both", "unexplained"]


def attribute(observed: Mapping[str, Any], *, before: Mapping[str, Any], predicted: Mapping[str, Any] | None = None,
              acted: bool = True) -> dict[str, Attribution]:
    """Which changes were mine, and which the world's.

    ``predicted`` is the efference copy: what my own action said would change. A changed
    aspect that my action predicted is mine; one it did not predict is the world's; an
    aspect that changed while I did nothing is the world's by construction. Predicting a
    change that did not happen is *not* a change, and is the business of
    :func:`tensorcode.expectation.check`, not of this function.
    """
    predicted = predicted or {}
    out: dict[str, Attribution] = {}
    for aspect, now in observed.items():
        if aspect in before and before[aspect] == now:
            continue  # nothing changed; nothing to attribute
        if not acted:
            out[aspect] = "world"
            continue
        if aspect not in predicted:
            out[aspect] = "world"
        elif predicted[aspect] == now:
            out[aspect] = "self"
        else:
            # my action touched this aspect but the result is not what it predicted:
            # something of mine and something else both moved it
            out[aspect] = "both"
    return out


def surprising(attributions: Mapping[str, Attribution]) -> tuple[str, ...]:
    """The aspects worth attention: the ones I did not cause."""
    return tuple(a for a, kind in attributions.items() if kind in ("world", "both", "unexplained"))
