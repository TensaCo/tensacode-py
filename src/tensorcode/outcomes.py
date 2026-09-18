"""Outcome values returned by operations.

These are ordinary values, not execution wrappers. Each one encodes a semantic
distinction that a caller must not silently collapse:

* ``Unknown`` is not ``False`` and not a low-confidence guess.
* ``Verdict`` has three states; ``fails`` and ``unknown`` are different.
* ``Receipt`` distinguishes "did not happen" from "may have happened".
* ``Score`` says what kind of number it is (a similarity is not a probability).

Execution metadata (backend, timing, cost, escalation) lives in the trace, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def _no_truth_value(self: Any) -> bool:
    raise TypeError(
        f"{type(self).__name__} has no truth value; handle it explicitly "
        "(e.g. `isinstance(x, Unknown)` or `verdict.status == 'holds'`)"
    )


@dataclass(frozen=True)
class Unknown:
    """An operation could not produce an answer it is entitled to give.

    ``candidates`` are best-effort guesses with their scores. They are *not* answers.
    """

    reason: str
    detail: str = ""
    candidates: tuple[tuple[Any, "Score"], ...] = ()

    __bool__ = _no_truth_value


ScoreKind = Literal[
    "probability",  # calibrated on a named dataset (see ``basis``)
    "uncalibrated",  # model-internal confidence; ordering only
    "similarity",  # geometric closeness; not a probability of anything
    "relevance",  # retrieval score; comparable only within one ranking
    "utility",  # objective value under a declared objective
    "vote_share",  # fraction of voters/samples
]


@dataclass(frozen=True)
class Score:
    value: float
    kind: ScoreKind
    basis: str = ""  # e.g. "banking77/validation@2026-09-16" for a calibrated probability

    def __post_init__(self) -> None:
        if self.kind == "probability" and not self.basis:
            raise ValueError("a calibrated probability must name its calibration basis")


@dataclass(frozen=True)
class Verdict:
    """Result of evaluating a proposition, candidate, or constraint against evidence."""

    status: Literal["holds", "fails", "unknown"]
    reasons: tuple[str, ...] = ()
    evidence: tuple[Any, ...] = ()  # Refs or observations actually consulted

    __bool__ = _no_truth_value

    @property
    def holds(self) -> bool:
        return self.status == "holds"


ReceiptStatus = Literal[
    "applied",  # the effect happened and the executor observed it
    "rejected",  # the executor refused before any effect (policy, validation, auth)
    "failed",  # attempted; known not to have taken effect
    "indeterminate",  # attempted; may or may not have taken effect (timeouts, lost replies)
]


@dataclass(frozen=True)
class Receipt:
    """What an executor can honestly say about one invocation of an action."""

    action: Any
    status: ReceiptStatus
    retryable: bool = False  # safe to retry *as far as the executor knows*
    idempotency_key: str | None = None
    effect_id: str | None = None
    error: str | None = None
    retry_after_s: float | None = None
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    __bool__ = _no_truth_value
