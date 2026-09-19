"""Causal claims, kept apart from evidential ones, and earned by intervention.

Provenance already answers "why do I believe this" — which source said it. It does not
answer "what makes this happen", and the two are routinely confused: a claim derived from
another is *supported* by it, not *caused* by it. So causation is its own record, and it
carries how it was learned:

* ``observational`` support means the two were seen together. That is a correlation, and
  this module names it one.
* ``interventional`` support means the same state was run twice, once with the act and once
  without, and the effect differed. An agent's own action is a real ``do(·)``, and a world
  that can fork gives the untaken branch for free.

:func:`experiment` runs that controlled pair, :func:`learn` turns contrasts into claims
with an effect size, and :func:`counterfactual` records the branch that did not happen in a
scope of its own, so "what would have happened" never leaks into the shared world.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from .outcomes import Score, Unknown
from .records import Claim, Evidence, Ref, Store

Support = str  # "interventional" | "observational"


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Contrast:
    """One aspect, observed with the act and without it, from the same starting state."""

    cause: str
    aspect: str
    with_act: tuple[Any, ...]
    without_act: tuple[Any, ...]

    @property
    def pairs(self) -> tuple[tuple[Any, Any], ...]:
        """Comparable paired observations; absent or unresolved values are not outcomes."""
        return tuple((a, b) for a, b in zip(self.with_act, self.without_act)
                     if not isinstance(a, Unknown) and not isinstance(b, Unknown))

    @property
    def trials(self) -> int:
        return len(self.pairs)

    @property
    def outcomes(self) -> tuple[tuple[Any, int], ...]:
        """Treated outcome counts over comparable trials, without inventing probabilities."""
        counts: list[tuple[Any, int]] = []
        for value, _ in self.pairs:
            for index, (seen, count) in enumerate(counts):
                if value == seen:
                    counts[index] = (seen, count + 1)
                    break
            else:
                counts.append((value, 1))
        return tuple(counts)

    @property
    def changed_with(self) -> float:
        """Fraction of with-act runs where this aspect ended up different from its control."""
        if not self.trials:
            return 0.0
        return sum(1 for a, b in self.pairs if a != b) / self.trials

    @property
    def effect(self) -> float:
        """The average causal effect: how often the act, and only the act, moved this aspect."""
        return self.changed_with

    def describe(self) -> str:
        return f"{self.cause} → {self.aspect}: {self.with_act[:1]} vs control {self.without_act[:1]} (effect {self.effect:.2f}, n={self.trials})"


@dataclass(frozen=True)
class Causal:
    """``cause`` brings about ``effect``, with the evidence that says so."""

    cause: str
    aspect: str
    effect: Any
    support: Support
    strength: Score
    trials: int
    mechanism: str | None = None
    enabling: tuple[str, ...] = ()  # conditions that must hold for the link to fire
    contrast: Contrast | None = None  # paired runs, including incomplete observations

    @property
    def ref(self) -> Ref:
        return Ref(f"causal:{_digest([self.cause, self.aspect, str(self.effect), self.support, self.contrast])}")

    def describe(self) -> str:
        how = f" via {self.mechanism}" if self.mechanism else ""
        needs = f" when {', '.join(self.enabling)}" if self.enabling else ""
        effect = (f"changes {self.aspect}; outcome unresolved ({self.effect.detail})"
                  if isinstance(self.effect, Unknown) else f"causes {self.aspect}={self.effect!r}")
        return f"{self.cause} {effect}{how}{needs} [{self.support}, {self.strength.kind}={self.strength.value:.2f}, n={self.trials}]"


# ------------------------------------------------------------- interventions


def experiment(
    *,
    prepare: Callable[[], Any],
    act: Callable[[Any], None],
    observe: Callable[[Any], Mapping[str, Any]],
    cause: str,
    trials: int = 1,
    settle: Callable[[Any], None] | None = None,
    control: Callable[[Any], None] | None = None,
) -> list[Contrast]:
    """Run the same starting state with and without ``act``; report what differed.

    ``prepare`` returns a fresh copy of the world (a fork, a restored checkpoint), so the
    two branches differ in exactly one thing: whether the act happened. Anything that moves
    on its own — a clock, an animation, a scheduled event — moves in both branches and so
    cancels out of the contrast, which is the whole point of running the control.

    ``control`` is what the untreated branch does *instead*: an innocuous act of the same
    kind (a click on empty space, a step that changes nothing). Without it the control
    branch does nothing at all, and then everything that follows from merely acting — a
    logical clock, a step counter, a repaint — is scored as an effect of this act. Which
    control is right depends on the question: leave it out to ask "what follows from doing
    this rather than nothing", pass one to ask "what does *this* act do that any act would
    not".
    """
    with_runs: list[Mapping[str, Any]] = []
    without_runs: list[Mapping[str, Any]] = []
    for _ in range(max(1, trials)):
        treated = prepare()
        act(treated)
        if settle:
            settle(treated)
        with_runs.append(dict(observe(treated)))

        untreated = prepare()
        if control:
            control(untreated)
        if settle:
            settle(untreated)
        without_runs.append(dict(observe(untreated)))
    aspects = sorted({k for run in with_runs + without_runs for k in run})
    return [
        Contrast(cause, aspect,
                 tuple(run.get(aspect, Unknown("unobserved", aspect)) for run in with_runs),
                 tuple(run.get(aspect, Unknown("unobserved", aspect)) for run in without_runs))
        for aspect in aspects
    ]


def learn(contrasts: Iterable[Contrast], *, least_effect: float = 0.5, basis: str = "intervention") -> list[Causal]:
    """Summarize observed intervention contrasts without choosing a variable outcome.

    A scalar effect means all comparable treated outcomes agreed in this sample;
    it is not a guarantee about future trials. Variable outcomes remain Unknown,
    with their empirical shares and full paired evidence retained. Strength is the
    observed share of changed pairs, not a calibrated future-outcome probability.
    """
    out: list[Causal] = []
    for contrast in contrasts:
        if contrast.trials == 0 or contrast.effect < least_effect:
            continue
        outcomes = contrast.outcomes
        value = outcomes[0][0] if len(outcomes) == 1 else Unknown(
            "variable_outcome", f"{len(outcomes)} treated outcomes across {contrast.trials} comparable trials",
            tuple((outcome, Score(count / contrast.trials, "vote_share",
                                  basis=f"{basis}:treated-outcomes@n={contrast.trials}"))
                  for outcome, count in outcomes),
        )
        out.append(Causal(
            cause=contrast.cause, aspect=contrast.aspect, effect=value, support="interventional",
            strength=Score(contrast.effect, "vote_share", basis=f"{basis}:changed-pairs@n={contrast.trials}"),
            trials=contrast.trials, contrast=contrast,
        ))
    return out


def correlations(observations: Sequence[tuple[set[str], Mapping[str, Any]]], *, cause: str,
                 least: float = 0.5, basis: str = "co-occurrence") -> list[Causal]:
    """The baseline an intervention has to beat: what merely co-occurs with the act.

    Each observation is ``(acts_that_happened, aspects_observed)``. No control, so a
    consequence of something else that always accompanies the act scores just as highly —
    which is exactly the mistake :func:`experiment` exists to avoid.
    """
    seen: dict[tuple[str, Any], int] = {}
    total = 0
    for acts, aspects in observations:
        if cause not in acts:
            continue
        total += 1
        for aspect, value in aspects.items():
            seen[(aspect, _hashable(value))] = seen.get((aspect, _hashable(value)), 0) + 1
    if not total:
        return []
    out: list[Causal] = []
    for (aspect, value), hits in sorted(seen.items(), key=lambda kv: -kv[1]):
        share = hits / total
        if share >= least:
            out.append(Causal(cause=cause, aspect=aspect, effect=value, support="observational",
                              strength=Score(share, "probability", basis=f"{basis}@n={total}"), trials=total))
    return out


def moved(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, str]:
    """Which aspects moved, as events rather than values.

    Co-occurrence has to be counted over movements, not over exact values: an aspect whose
    value is different every time (a clock, a counter) never repeats, so keying on the value
    makes a perfectly reliable co-occurrence look like noise. This is the same lesson the
    prediction measurement gave — what changed is learnable, what it became often is not.
    """
    return {key: "changed" for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)}


def _hashable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


# -------------------------------------------------------------------- claims


def tell_causal(mind: Store, causal: Causal, *, source: Ref, observed_at: datetime | None = None,
                derived_from: Sequence[str] = ()) -> Ref:
    """Record a causal link as claims that say how it was learned."""
    at = observed_at or datetime.now(timezone.utc)
    ref = causal.ref
    evidence = Evidence(source=source, observed_at=at, method=f"causal:{causal.support}",
                        confidence=causal.strength, derived_from=tuple(derived_from))
    mind.tell(Claim(ref, "is_a", "causal_link"), evidence)
    mind.tell(Claim(ref, "cause", causal.cause), evidence)
    mind.tell(Claim(ref, "aspect", causal.aspect), evidence)
    if isinstance(causal.effect, Unknown):
        mind.tell(Claim(ref, "effect_unresolved", causal.effect.reason), evidence)
    else:
        mind.tell(Claim(ref, "effect", causal.effect), evidence)
    if causal.contrast is not None:
        mind.tell(Claim(ref, "treated_observations", causal.contrast.with_act), evidence)
        mind.tell(Claim(ref, "control_observations", causal.contrast.without_act), evidence)
        mind.tell(Claim(ref, "treated_outcome_counts", causal.contrast.outcomes), evidence)
    mind.tell(Claim(ref, "support", causal.support), evidence)
    mind.tell(Claim(ref, "trials", causal.trials), evidence)
    if causal.mechanism:
        mind.tell(Claim(ref, "mechanism", causal.mechanism), evidence)
    for condition in causal.enabling:
        mind.tell(Claim(ref, "enabled_by", condition), evidence)
    return ref


def causes_of(mind: Store, aspect: str, *, interventional_only: bool = False) -> list[Ref]:
    """Recorded causes of an aspect, strongest support first."""
    links = [r.claim.subject for r in mind.claims(predicate="aspect", object=aspect)]
    out = []
    for ref in links:
        support = next((r.claim.object for r in mind.claims(ref, "support")), None)
        if interventional_only and support != "interventional":
            continue
        out.append((0 if support == "interventional" else 1, ref))
    return [ref for _, ref in sorted(out, key=lambda pair: pair[0])]


def counterfactual(mind: Store, *, name: str, facts: Iterable[tuple[Ref, str, Any]], source: Ref,
                   because: str, observed_at: datetime | None = None) -> Ref:
    """Record the branch that did not happen, in a scope of its own.

    A counterfactual is real knowledge — it is what makes an effect size meaningful — but it
    is not true of the world, so it lives in its own scope and never answers a question
    about what is.
    """
    at = observed_at or datetime.now(timezone.utc)
    scope = Ref(f"scope:counterfactual:{name}")
    evidence = Evidence(source=source, observed_at=at, method="counterfactual")
    mind.tell(Claim(scope, "is_a", "counterfactual"), evidence)
    mind.tell(Claim(scope, "because", because), evidence)
    for subject, predicate, value in facts:
        mind.tell(Claim(subject, predicate, value, scope=scope), evidence)
    return scope


def distinguish(contrasts: Iterable[Contrast], correlated: Iterable[Causal]) -> dict[str, str]:
    """Which co-occurring aspects the intervention actually vindicated.

    Returns aspect → "caused" | "merely_correlated", which is the judgement a correlational
    record cannot make on its own.
    """
    caused = {c.aspect for c in contrasts if c.trials and c.effect >= 0.5}
    seen = {c.aspect for c in correlated}
    return {aspect: ("caused" if aspect in caused else "merely_correlated") for aspect in sorted(seen | caused)}
