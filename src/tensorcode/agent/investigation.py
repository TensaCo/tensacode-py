"""Bounded observation of supplied, world-conditional interpretation hypotheses.

Agreement with predictions supports a supplied model; it does not establish a
speaker's intent or discover the model. This module neither selects workspace
candidates nor writes beliefs or invokes actions. Providers must honor their
read-only observation contract.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterable

from ..goals import Condition
from ..outcomes import Unknown
from .plugin import Plugin


@dataclass(frozen=True)
class CandidateHypothesis:
    candidate_id: str
    predictions: tuple[Condition, ...]
    basis: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("a hypothesis requires a candidate id")
        if not self.basis or any(not isinstance(x, str) or not x.strip() for x in self.basis):
            raise ValueError("a hypothesis requires explicit nonempty basis")
        for condition in self.predictions:
            if not isinstance(condition, Condition):
                raise TypeError("predictions must be Conditions")
            if type(condition.negated) is not bool:
                raise TypeError("condition negation must be bool")
            if any(condition.pred == other.pred and condition.args == other.args
                   and condition.negated != other.negated for other in self.predictions):
                raise ValueError("contradictory hypothesis predictions")


@dataclass(frozen=True)
class ProviderEvidence:
    provider: str
    status: str
    value: bool | None
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True)
class ObservationRecord:
    # Copy-on-access keeps mutable domain references usable without exposing the
    # retained snapshot. A frozen dataclass alone cannot protect nested mappings.
    _condition: Condition = field(repr=False)
    providers: tuple[ProviderEvidence, ...]
    _value: bool | Unknown = field(repr=False)
    order: int

    @property
    def condition(self) -> Condition:
        return deepcopy(self._condition)

    @property
    def value(self) -> bool | Unknown:
        return deepcopy(self._value)


@dataclass(frozen=True)
class HypothesisAssessment:
    candidate_id: str
    basis: tuple[str, ...]
    _confirmed: tuple[Condition, ...] = field(repr=False)
    _contradicted: tuple[Condition, ...] = field(repr=False)
    _unresolved: tuple[Condition, ...] = field(repr=False)

    @property
    def confirmed(self) -> tuple[Condition, ...]:
        return deepcopy(self._confirmed)

    @property
    def contradicted(self) -> tuple[Condition, ...]:
        return deepcopy(self._contradicted)

    @property
    def unresolved(self) -> tuple[Condition, ...]:
        return deepcopy(self._unresolved)

    @property
    def viable(self) -> bool:
        return not self._contradicted


@dataclass(frozen=True)
class InvestigationResult:
    selected_id: str | None
    observations: tuple[ObservationRecord, ...]
    assessments: tuple[HypothesisAssessment, ...]
    reason: str


def investigate(
    hypotheses: Iterable[CandidateHypothesis], providers: Iterable[Plugin], *, max_probes: int = 8,
) -> InvestigationResult:
    """Observe distinct positive atoms, preferring balanced discriminating probes.

    Missing predictions leave a candidate viable under either outcome. Unknown
    providers abstain; contradictory bool observations and malformed responses
    make the aggregate unknown. Selection requires every survivor prediction to
    be confirmed and every rival to have an observed contradiction. It remains
    relative to the supplied hypotheses and observation contracts.
    """
    if type(max_probes) is not int or max_probes < 0:
        raise ValueError("max_probes must be a nonnegative integer")
    models = deepcopy(tuple(hypotheses))
    observers = tuple(providers)
    if any(not isinstance(model, CandidateHypothesis) for model in models):
        raise TypeError("hypotheses must be CandidateHypothesis records")
    for model in models:
        model.__post_init__()  # Validate potentially mutated nested source data.
    if len({m.candidate_id for m in models}) != len(models):
        raise ValueError("candidate ids must be unique")
    names = tuple(p.name for p in observers)
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("observation providers require unique nonempty names")
    atoms: list[Condition] = []
    predictions: list[dict[int, bool]] = []
    for model in models:
        bound: dict[int, bool] = {}
        for condition in model.predictions:
            positive = Condition(condition.pred, deepcopy(condition.args))
            if positive not in atoms:
                atoms.append(positive)
            bound[atoms.index(positive)] = not condition.negated
        predictions.append(bound)
    values: dict[int, bool | Unknown] = {}
    observations: list[ObservationRecord] = []

    def assess() -> tuple[HypothesisAssessment, ...]:
        result = []
        for model, expected in zip(models, predictions):
            confirmed, contradicted, unresolved = [], [], []
            for index, wanted in expected.items():
                condition = Condition(atoms[index].pred, deepcopy(atoms[index].args), not wanted)
                actual = values.get(index)
                target = unresolved if type(actual) is not bool else confirmed if actual is wanted else contradicted
                target.append(condition)
            result.append(HypothesisAssessment(model.candidate_id, tuple(model.basis),
                                               tuple(confirmed), tuple(contradicted), tuple(unresolved)))
        return tuple(result)

    while True:
        assessments = assess()
        viable = [i for i, assessment in enumerate(assessments) if assessment.viable]
        selected = None
        if len(viable) == 1:
            survivor = assessments[viable[0]]
            if survivor.confirmed and not survivor.unresolved:
                selected = survivor.candidate_id
        if selected is not None:
            reason = "supported_unique_hypothesis"
            break
        if not viable:
            reason = "all_hypotheses_contradicted" if models else "no_hypotheses"
            break
        pending = [j for j in range(len(atoms)) if j not in values
                   and any(j in predictions[i] for i in viable)]
        if not pending:
            reason = "insufficient_evidence" if len(viable) == 1 else "unresolved_alternatives"
            break
        if len(observations) >= max_probes:
            reason = "probe_budget_exhausted"
            break

        def rank(index: int) -> tuple[int, int, int]:
            positive = sum(predictions[i].get(index) is True for i in viable)
            negative = sum(predictions[i].get(index) is False for i in viable)
            missing = len(viable) - positive - negative
            return (0 if positive and negative else 1, max(positive, negative) + missing, index)

        index = min(pending, key=rank)
        evidence = []
        known: set[bool] = set()
        invalid = False
        for provider in observers:
            try:
                observed = provider.observe_condition(deepcopy(atoms[index]))
            except Exception as exc:
                observed = Unknown("observation_error", f"{type(exc).__name__}: {exc}")
                invalid = True
            if type(observed) is bool:
                known.add(observed)
                evidence.append(ProviderEvidence(provider.name, "observed", observed))
            elif isinstance(observed, Unknown):
                evidence.append(ProviderEvidence(provider.name, "unknown", None, observed.reason, observed.detail))
            else:
                invalid = True
                evidence.append(ProviderEvidence(provider.name, "invalid", None, "invalid_observation",
                                                 "provider returned neither bool nor Unknown"))
        value = (Unknown("invalid_observation") if invalid else
                 Unknown("conflicting_observations") if len(known) > 1 else
                 next(iter(known)) if known else Unknown("unobserved_condition"))
        values[index] = value
        observations.append(ObservationRecord(deepcopy(atoms[index]), tuple(evidence), value, len(observations)))
    return InvestigationResult(selected, tuple(observations), assessments, reason)
