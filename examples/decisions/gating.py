"""Confidence gating: turning an answer plus a confidence into auto / confirm / escalate.

The pattern is the ordinary one — a threshold per consequence, low-stakes actions run at a
lower bar than money-moving ones. Two things here are not ordinary, and both come from
``tensacode.outcomes``:

1. **A gate refuses to threshold a number that is not a probability.** ``Score`` carries its
   ``kind``, so a similarity, a relevance score or a model's own uncalibrated confidence
   cannot be silently compared against 0.85. Gating on a retrieval score as if it were
   P(correct) is a common way to ship a confidently wrong automation; here it is a type
   error, reported as ``Gate("escalate", why="not a calibrated probability: similarity")``.

2. **``Unknown`` escalates, and is not zero.** An abstention is not a 0.0 confidence, so it
   never lands in the "low confidence but still act" band.

Thresholds are configuration, not opinion: ``Gate.from_measurement`` builds them from a
measured selective-accuracy curve, so "auto above 0.93" means "0.93 is where measured
accuracy on held-out data reached the accuracy this action requires".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from tensacode.outcomes import Score, Unknown

Action = Literal["auto", "confirm", "escalate"]


@dataclass(frozen=True)
class Decided:
    """What the gate concluded, and why — both go in the audit record."""

    action: Action
    confidence: float | None
    why: str
    basis: str = ""


@dataclass(frozen=True)
class Thresholds:
    """One consequence class. ``auto`` must be at least ``confirm``."""

    name: str
    auto: float
    confirm: float
    basis: str = "hand-set"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confirm <= self.auto <= 1.0:
            raise ValueError(f"{self.name}: need 0 <= confirm ({self.confirm}) <= auto ({self.auto}) <= 1")


#: Defaults. Reversible work runs at a lower bar than money movement; the numbers are
#: replaced by measured ones in eval/decisions/measure.py, which writes the basis string.
DEFAULT_THRESHOLDS: dict[str, Thresholds] = {
    "routing": Thresholds("routing", auto=0.60, confirm=0.30, basis="hand-set default: routing is reversible"),
    "reply": Thresholds("reply", auto=0.80, confirm=0.50, basis="hand-set default: a wrong reply is visible to a customer"),
    "money": Thresholds("money", auto=0.95, confirm=0.70, basis="hand-set default: money movement"),
}


class Gate:
    """Decides auto / confirm / escalate for one consequence class."""

    def __init__(self, thresholds: Thresholds, *, require_probability: bool = True) -> None:
        self.t = thresholds
        self.require_probability = require_probability

    @classmethod
    def from_measurement(cls, name: str, curve: Sequence[tuple[float, float, int]], *, target_accuracy: float, confirm_at: float, basis: str) -> "Gate":
        """Lowest threshold whose measured selective accuracy reaches ``target_accuracy``.

        ``curve`` is (threshold, accuracy among answers above it, n). If no threshold on the
        curve reaches the target, the gate never auto-acts — it does not round down to the
        best available, because "the best we measured" is not "good enough to act".
        """
        ok = [(threshold, accuracy, n) for threshold, accuracy, n in curve if accuracy >= target_accuracy and n > 0]
        auto = min(t for t, _, _ in ok) if ok else 1.0
        note = f"{basis}; auto={auto:.3f} is the lowest measured threshold reaching accuracy {target_accuracy}"
        if not ok:
            note = f"{basis}; NO measured threshold reached accuracy {target_accuracy}, so auto is disabled"
        return cls(Thresholds(name, auto=auto, confirm=min(confirm_at, auto), basis=note))

    def decide(self, answer: object, score: Score | None) -> Decided:
        if isinstance(answer, Unknown):
            return Decided("escalate", None, f"no answer: {answer.reason}", self.t.basis)
        if score is None:
            return Decided("escalate", None, "answer carried no confidence", self.t.basis)
        if self.require_probability and score.kind != "probability":
            return Decided("escalate", None, f"not a calibrated probability: {score.kind}", self.t.basis)
        p = score.value
        if p >= self.t.auto:
            return Decided("auto", p, f"{p:.3f} >= auto {self.t.auto:.3f} ({self.t.name})", score.basis or self.t.basis)
        if p >= self.t.confirm:
            return Decided("confirm", p, f"{p:.3f} in [{self.t.confirm:.3f}, {self.t.auto:.3f}) ({self.t.name})", score.basis or self.t.basis)
        return Decided("escalate", p, f"{p:.3f} < confirm {self.t.confirm:.3f} ({self.t.name})", score.basis or self.t.basis)


def gates(overrides: dict[str, Thresholds] | None = None) -> dict[str, Gate]:
    table = dict(DEFAULT_THRESHOLDS)
    table.update(overrides or {})
    return {name: Gate(t) for name, t in table.items()}
