"""Expectations, and the prediction errors that follow from them.

A modality feature on a parsed frame ("an announcement *should* appear") is inert: it
records that someone said "should" and predicts nothing. An expectation here is a
commitment — a cue, what should then hold, and how sure — which means it can be *wrong*,
and being wrong is the useful part:

* :func:`check` compares what was expected against what was observed and, on a mismatch,
  writes a :class:`Violation` claim naming both sides and their provenance;
* a violation is a claim like any other, so attention (``awareness``) can be seeded from
  it and rules can fire on it — surprise becomes a first-class input rather than a log line;
* :class:`Predictor` learns the probability from its own record of hits and misses, so the
  number on an expectation is a frequency with a named basis rather than a hand-set prior.

Nothing here guesses: an expectation with too little evidence reports
:class:`~tensorcode.outcomes.Unknown` rather than a made-up probability.
"""

from __future__ import annotations

import hashlib
import json
from math import exp, log, log2
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .outcomes import Score, Unknown, Verdict
from .records import Claim, Evidence, Ref, Store

MIN_TRIALS = 3  # below this, a frequency is not a probability worth reporting


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Expectation:
    """If ``cue`` happens, ``predicted`` should hold afterwards, with probability ``p``."""

    cue: str  # a description of the triggering act or event, e.g. "click:Send message"
    predicted: tuple[tuple[str, Any], ...]  # (aspect, value) pairs that should be observed
    p: Score | None = None
    scope: Ref | None = None
    source: str = "stated"  # "stated" (someone said so) | "learned" (from its own record)

    @property
    def ref(self) -> Ref:
        return Ref(f"expectation:{_digest([self.cue, list(self.predicted), self.source])}")

    def describe(self) -> str:
        body = ", ".join(f"{a}={v!r}" for a, v in self.predicted)
        odds = f" p={self.p.value:.2f}" if self.p else ""
        return f"after {self.cue}: {body}{odds}"


@dataclass(frozen=True)
class Violation:
    """A prediction error: what was expected, what happened, and how surprising it was."""

    expectation: Expectation
    aspect: str
    expected: Any
    observed: Any
    surprise: float  # -log2(p assigned to what actually happened), 0 when unsurprising

    @property
    def ref(self) -> Ref:
        return Ref(f"violation:{_digest([self.expectation.ref.id, self.aspect, str(self.expected), str(self.observed)])}")

    def describe(self) -> str:
        return f"{self.aspect}: expected {self.expected!r}, observed {self.observed!r} (surprise {self.surprise:.2f} bits)"


def expect(mind: Store, expectation: Expectation, *, source: Ref, observed_at: datetime | None = None,
           method: str = "expectation") -> Ref:
    """Record an expectation as claims, so it can be inspected, cited and retracted."""
    at = observed_at or datetime.now(timezone.utc)
    ref = expectation.ref
    evidence = Evidence(source=source, observed_at=at, method=method, confidence=expectation.p)
    mind.tell(Claim(ref, "is_a", "expectation", scope=expectation.scope), evidence)
    mind.tell(Claim(ref, "cue", expectation.cue, scope=expectation.scope), evidence)
    mind.tell(Claim(ref, "held_because", expectation.source, scope=expectation.scope), evidence)
    for aspect, value in expectation.predicted:
        mind.tell(Claim(ref, f"predicts:{aspect}", value, scope=expectation.scope), evidence)
    return ref


def check(mind: Store, expectation: Expectation, observed: Mapping[str, Any], *, source: Ref,
          observed_at: datetime | None = None) -> tuple[Verdict, tuple[Violation, ...]]:
    """Compare an expectation against what was observed; record any prediction error.

    Returns a verdict and the violations. An aspect the observation does not mention is
    ``unknown``, not a violation: unseen is not disconfirmed.
    """
    at = observed_at or datetime.now(timezone.utc)
    violations: list[Violation] = []
    unseen: list[str] = []
    for aspect, value in expectation.predicted:
        if aspect not in observed:
            unseen.append(aspect)
            continue
        got = observed[aspect]
        if got != value:
            p = expectation.p.value if expectation.p else 0.5
            surprise = -log2(max(1e-6, 1.0 - p))
            violations.append(Violation(expectation, aspect, value, got, surprise))
    for violation in violations:
        mind.tell(
            Claim(violation.ref, "is_a", "violation"),
            Evidence(source=source, observed_at=at, method="prediction-error", derived_from=()),
        )
        for predicate, value in (("of", expectation.ref), ("aspect", violation.aspect),
                                 ("expected", violation.expected), ("observed", violation.observed),
                                 ("surprise", violation.surprise)):
            mind.tell(Claim(violation.ref, predicate, value), Evidence(source=source, observed_at=at, method="prediction-error"))
    if violations:
        return Verdict("fails", tuple(v.describe() for v in violations), (expectation.ref,)), tuple(violations)
    if unseen and len(unseen) == len(expectation.predicted):
        return Verdict("unknown", (f"nothing observed about {', '.join(unseen)}",), (expectation.ref,)), ()
    return Verdict("holds", (expectation.describe(),), (expectation.ref,)), ()


# ------------------------------------------------------------------ learning


@dataclass
class Predictor:
    """Learns what follows a cue from its own hits and misses.

    The probability it reports is a smoothed frequency over trials it actually saw, with
    the basis naming the record it came from. Below :data:`MIN_TRIALS` it reports
    ``Unknown`` instead of a number, because three coin flips are not a calibration.
    """

    name: str = "expectation/observed"
    counts: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))
    trials: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def observe(self, cue: str, aspect: str, value: Any) -> None:
        key = (cue, aspect)
        self.counts[key][_hashable(value)] += 1
        self.trials[key] += 1

    def predict(self, cue: str, aspect: str) -> tuple[Any, Score] | Unknown:
        """The most frequent outcome for this cue and aspect, with its frequency."""
        key = (cue, aspect)
        trials = self.trials.get(key, 0)
        if trials < MIN_TRIALS:
            return Unknown("too_few_trials", f"{trials} observations of {aspect} after {cue}")
        value, hits = self.counts[key].most_common(1)[0]
        # Laplace over the outcomes seen *and one that has not been*: with a single observed
        # outcome the seen-only denominator cancels and twenty hits would report certainty,
        # which is the failure this smoothing exists to prevent.
        outcomes = max(2, len(self.counts[key]))
        p = (hits + 1) / (trials + outcomes)
        return value, Score(p, "probability", basis=f"{self.name}:{cue}/{aspect}@{trials}")

    def expectation(self, cue: str, aspects: Iterable[str]) -> Expectation | Unknown:
        """An expectation over several aspects, at the weakest of their probabilities."""
        predicted: list[tuple[str, Any]] = []
        weakest: Score | None = None
        for aspect in aspects:
            got = self.predict(cue, aspect)
            if isinstance(got, Unknown):
                return got
            value, score = got
            predicted.append((aspect, value))
            if weakest is None or score.value < weakest.value:
                weakest = score
        if not predicted:
            return Unknown("nothing_to_predict", cue)
        return Expectation(cue, tuple(predicted), weakest, source="learned")


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, set):
        return tuple(sorted(_hashable(v) for v in value))
    return value


# -------------------------------------------------------------- probability


def combine(scores: Sequence[Score], *, assume: str = "independent") -> Score | Unknown:
    """Pool several probabilities about one proposition, naming the assumption used.

    Independent evidence pools in log-odds. That assumption is usually false — two sources
    reading the same notice are one source — so it is written into the basis rather than
    left implicit, and a caller that cannot justify it should not use the number.

    Refuses when any input is not a calibrated probability: a similarity and a vote share do
    not pool into a probability, and pretending they do is how a confident wrong number gets
    made.
    """
    scores = list(scores)
    if not scores:
        return Unknown("no_evidence", "nothing to combine")
    wrong = [s.kind for s in scores if s.kind != "probability"]
    if wrong:
        return Unknown("not_probabilities", f"cannot pool {', '.join(sorted(set(wrong)))} into a probability")
    if assume != "independent":
        return Unknown("unsupported_assumption", f"only independent pooling is implemented, not {assume!r}")
    total = 0.0
    for score in scores:
        p = min(max(score.value, 1e-6), 1 - 1e-6)
        total += log(p / (1 - p))
    pooled = 1 / (1 + exp(-total))
    bases = "+".join(sorted({s.basis for s in scores if s.basis}))
    return Score(pooled, "probability", basis=f"pooled(independent,n={len(scores)}):{bases}"[:200])


def disagreement(scores: Sequence[Score]) -> float:
    """How far apart the evidence is, as the spread of its probabilities.

    Pooling hides this: two sources at 0.1 and 0.9 pool to 0.5, and so do two at 0.5. A
    caller that needs to know whether its belief is settled or contested needs the spread as
    well as the pooled number.
    """
    values = [s.value for s in scores]
    return (max(values) - min(values)) if values else 0.0


@dataclass(frozen=True)
class Calibration:
    """How well stated probabilities matched what happened."""

    bins: tuple[tuple[float, float, int], ...]  # (stated, observed, n) per occupied bin
    ece: float  # expected calibration error: average gap, weighted by how often each bin was used
    n: int

    def describe(self) -> str:
        rows = ", ".join(f"{stated:.2f}→{observed:.2f}({count})" for stated, observed, count in self.bins)
        return f"ECE {self.ece:.3f} over {self.n}: {rows}"


def calibration(pairs: Sequence[tuple[float, bool]], *, bins: int = 10) -> Calibration | Unknown:
    """Bin stated probabilities against observed frequencies."""
    if not pairs:
        return Unknown("no_predictions", "nothing to calibrate")
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for p, hit in pairs:
        buckets[min(bins - 1, int(p * bins))].append((p, hit))
    rows: list[tuple[float, float, int]] = []
    error = 0.0
    for index in sorted(buckets):
        group = buckets[index]
        stated = sum(p for p, _ in group) / len(group)
        observed = sum(1 for _, hit in group if hit) / len(group)
        rows.append((round(stated, 4), round(observed, 4), len(group)))
        error += len(group) * abs(stated - observed)
    return Calibration(tuple(rows), round(error / len(pairs), 4), len(pairs))


def surprises(mind: Store, *, since: datetime | None = None, least: float = 0.0) -> list[Claim]:
    """Violations on record, most surprising first — what attention should be drawn to."""
    out: list[tuple[float, Claim]] = []
    for record in mind.claims(predicate="surprise"):
        if since is not None and all(e.observed_at < since for e in record.evidence):
            continue
        value = record.claim.object
        if isinstance(value, (int, float)) and value >= least:
            out.append((float(value), record.claim))
    return [claim for _, claim in sorted(out, key=lambda pair: -pair[0])]
