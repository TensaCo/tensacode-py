"""Fitting an abstention threshold, and failing loudly when the target cannot be met.

The first version of this returned a threshold of 1.0 when no operating point reached the
target selective accuracy — which silently refuses every item and looks like a designed
abstention. A fit that cannot reach its target is a fact about the model, not a threshold,
so it is reported as ``reachable: False`` and the caller decides. Nothing here guesses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TargetUnreachable(RuntimeError):
    """No operating point on the calibration slice met the requested selective accuracy."""


@dataclass(frozen=True)
class Fit:
    threshold: float
    reachable: bool
    target: float
    best_selective_accuracy: float  # the best any threshold achieved on the fitting slice
    coverage_at_threshold: float
    n: int
    note: str = ""

    @property
    def refuses_everything(self) -> bool:
        return self.coverage_at_threshold == 0.0


def fit_threshold(confidence: np.ndarray, correct: np.ndarray, *, target: float,
                  min_coverage: float = 0.05, strict: bool = False) -> Fit:
    """Lowest threshold whose selective accuracy meets ``target`` while still answering something.

    ``min_coverage`` keeps a threshold that answers almost nothing from counting as a success:
    one lucky item at 100% is not a calibrated operating point. When the target cannot be met,
    ``reachable`` is False and the threshold falls back to answering everything, so the caller
    sees the model's real accuracy instead of an empty coverage row. ``strict`` raises instead.
    """
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=float)
    if confidence.size == 0:
        raise TargetUnreachable("no calibration items")
    order = np.argsort(-confidence)
    ranked = correct[order]
    running = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    coverage = np.arange(1, len(ranked) + 1) / len(ranked)
    eligible = (running >= target) & (coverage >= min_coverage)
    best = float(running.max()) if len(running) else 0.0
    if not eligible.any():
        note = (f"no threshold reached {target:.2f} selective accuracy with at least {min_coverage:.0%} "
                f"coverage on {len(ranked)} calibration items (best {best:.3f}); "
                f"falling back to answering everything")
        if strict:
            raise TargetUnreachable(note)
        return Fit(0.0, False, target, best, 1.0, len(ranked), note)
    k = int(np.nonzero(eligible)[0].max()) + 1
    return Fit(float(confidence[order][k - 1]), True, target, best, float(k / len(ranked)), len(ranked))
