"""Induce guarded transition predictions from retained execution observations.

Observation projection and validation policy are supplied, not learned. Only rule
conditions and outcome associations are induced. Predictions are proposals with
sample provenance; they never install capabilities or certify future effects.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Iterable, Mapping

from ..agent.interpretation import InterpretationSource
from ..agent.plugin import Call
from ..outcomes import Receipt, Unknown
from .induce import DecisionList, decision_list
from .literals import candidate_literals


@dataclass(frozen=True)
class Transition:
    attempt_id: str
    provider: str
    source_ids: tuple[str, str]
    before: Any
    action: Any
    after: Any
    receipt: Receipt


@dataclass(frozen=True)
class ExcludedTransition:
    attempt_id: str
    source_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class TransitionBatch:
    transitions: tuple[Transition, ...]
    exclusions: tuple[ExcludedTransition, ...]


def _same(left: Any, right: Any) -> bool:
    """Structural action identity, including array payloads without importing numpy."""
    if type(left) is not type(right):
        return False
    if is_dataclass(left):
        return all(_same(getattr(left, f.name), getattr(right, f.name)) for f in fields(left))
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    if hasattr(left, "shape") and hasattr(left, "dtype") and callable(getattr(left, "tolist", None)):
        return left.shape == right.shape and str(left.dtype) == str(right.dtype) and _same(left.tolist(), right.tolist())
    result = left == right
    if not isinstance(result, bool):
        raise ValueError("action payload equality is not a scalar truth value")
    return result


def _action_matches(action: Any, other: Any, receipt: Any) -> bool:
    try:
        return isinstance(action, Call) and _same(action, other) and _same(action, receipt)
    except (TypeError, ValueError, AttributeError, RecursionError):
        return False


def extract_transitions(sources: Iterable[InterpretationSource], *, provider: str) -> TransitionBatch:
    """Pair one observed before/after source per applied attempt and selected provider."""
    if not provider or not provider.startswith("plugin:"):
        raise ValueError("select an explicit plugin:<name> observation provider")
    grouped: dict[str, list[InterpretationSource]] = {}
    exclusions: list[ExcludedTransition] = []
    seen: set[str] = set()
    for source in sources:
        if source.provider != provider or source.modality != "observation":
            continue
        if source.metadata.get("stage") not in ("before_action", "after_action"):
            continue
        if source.id in seen:
            raise ValueError(f"duplicate source ID: {source.id}")
        seen.add(source.id)
        attempt = source.metadata.get("attempt_id")
        if not isinstance(attempt, str) or not attempt:
            exclusions.append(ExcludedTransition("", (source.id,), "missing attempt ID"))
            continue
        grouped.setdefault(attempt, []).append(source)
    transitions: list[Transition] = []
    for attempt, rows in grouped.items():
        before = [s for s in rows if s.metadata["stage"] == "before_action"]
        after = [s for s in rows if s.metadata["stage"] == "after_action"]
        reason = ""
        if len(before) != 1 or len(after) != 1:
            reason = "requires exactly one before and one after observation"
        elif any(s.metadata.get("status") != "observed" for s in rows):
            reason = "observation unavailable or failed"
        else:
            receipt = after[0].metadata.get("receipt")
            action = before[0].metadata.get("action")
            if not isinstance(receipt, Receipt) or receipt.status != "applied":
                reason = "receipt does not establish an applied action"
            elif not _action_matches(action, after[0].metadata.get("action"), receipt.action):
                reason = "action identity disagrees across observations and receipt"
            elif before[0].metadata.get("receipt") is not None:
                reason = "before observation already contains a receipt"
        if reason:
            exclusions.append(ExcludedTransition(attempt, tuple(s.id for s in rows), reason))
            continue
        transitions.append(deepcopy(Transition(attempt, provider, (before[0].id, after[0].id),
                                               before[0].payload, action, after[0].payload, receipt)))
    return TransitionBatch(tuple(transitions), tuple(exclusions))


@dataclass(frozen=True)
class Projection:
    name: str
    features: Callable[[Any, Any], Mapping[str, Any]]
    outcome: Callable[[Any], Any]
    provenance: tuple[str, ...]
    kind: str = "authored"

    def __post_init__(self) -> None:
        if not self.name or not self.provenance or self.kind != "authored":
            raise ValueError("projection must name its supplied authored semantics and provenance")


@dataclass(frozen=True)
class ValidationPolicy:
    min_training_support: int = 2
    min_evaluation_support: int = 1
    min_accuracy: float = 1.0

    def __post_init__(self) -> None:
        if self.min_training_support < 1 or self.min_evaluation_support < 1 or not 0 < self.min_accuracy <= 1:
            raise ValueError("validation requires positive support and accuracy in (0, 1]")


@dataclass(frozen=True)
class RuleEvidence:
    rule_index: int
    training_attempt_ids: tuple[str, ...]
    evaluation_attempt_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    training_correct: int
    evaluation_correct: int
    reasons: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class Evaluation:
    total: int
    predicted: int
    correct: int

    @property
    def coverage(self) -> float:
        return self.predicted / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.predicted if self.predicted else None


@dataclass(frozen=True)
class TransitionPrediction:
    outcome: Any
    rule_index: int
    evidence: RuleEvidence
    projection: str
    projection_provenance: tuple[str, ...]


def _features(projection: Projection, before: Any, action: Any) -> frozenset:
    mapping = projection.features(deepcopy(before), deepcopy(action))
    if not isinstance(mapping, Mapping) or not mapping or any(not isinstance(k, str) for k in mapping):
        raise ValueError("projection must return a nonempty mapping with string feature names")
    if any(isinstance(value, Unknown) for value in mapping.values()):
        raise ValueError("unknown feature values cannot become observed facts")
    return frozenset(mapping.items())


def _matched(artifact: DecisionList, facts: frozenset) -> int | None:
    mapping = dict(facts)
    for index, rule in enumerate(artifact.rules):
        # Negated equality must not make an absent observation count as false.
        if any(condition.predicate not in mapping for condition in rule.conditions):
            return None
        if rule.matches(facts):
            return index
    return None


class LearnedTransitionModel:
    """Sample-validated induced rules; neither defaults nor unsupported rules answer."""

    def __init__(self, artifact: DecisionList, projection: Projection, evidence: tuple[RuleEvidence, ...],
                 evaluation: Evaluation, policy: ValidationPolicy, provider: str,
                 feature_values: Mapping[str, frozenset]) -> None:
        self._artifact = deepcopy(artifact)
        self.projection = projection
        self.evidence = evidence
        self.evaluation = evaluation
        self.policy = policy
        self.provider = provider
        self._feature_values = deepcopy(dict(feature_values))

    @property
    def artifact(self) -> DecisionList:
        return deepcopy(self._artifact)

    def predict(self, before: Any, action: Any) -> TransitionPrediction | Unknown:
        try:
            facts = _features(self.projection, before, action)
        except (ValueError, TypeError, KeyError) as error:
            return Unknown("unprojectable_observation", str(error))
        if any(name in self._feature_values and value not in self._feature_values[name]
               for name, value in facts):
            return Unknown("unseen_feature_value", "prediction lies outside observed training feature values")
        index = _matched(self._artifact, facts)
        if index is None:
            return Unknown("unsupported_transition", "no explicit induced rule covers the observed features")
        evidence = self.evidence[index]
        if not evidence.verified:
            return Unknown("unverified_transition", "; ".join(evidence.reasons))
        return TransitionPrediction(deepcopy(self._artifact.rules[index].label), index, evidence,
                                    self.projection.name, self.projection.provenance)


def fit_transitions(transitions: Iterable[Transition], *, projection: Projection,
                    train_attempt_ids: Iterable[str], evaluation_attempt_ids: Iterable[str],
                    policy: ValidationPolicy = ValidationPolicy()) -> LearnedTransitionModel:
    """Fit on explicit training attempts; validate rules on disjoint heldout attempts.

    Every supplied transition must be assigned to exactly one split. Evaluation
    gates learned rules but never changes their conditions or labels. Its scores
    describe this validation set, not an untouched final test-set estimate.
    """
    rows = list(deepcopy(tuple(transitions)))
    by_id = {row.attempt_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate attempt IDs")
    train_ids, eval_ids = tuple(train_attempt_ids), tuple(evaluation_attempt_ids)
    train, held = set(train_ids), set(eval_ids)
    if not train or not held or len(train) != len(train_ids) or len(held) != len(eval_ids):
        raise ValueError("both splits require unique, nonempty attempt IDs")
    if train & held or train | held != set(by_id):
        raise ValueError("splits must be disjoint and partition all supplied attempts")
    providers = {row.provider for row in rows}
    source_ids = [sid for row in rows for sid in row.source_ids]
    if len(providers) != 1 or len(source_ids) != len(set(source_ids)):
        raise ValueError("require one provider and independently identified observation sources")
    if any(not isinstance(row.receipt, Receipt) or row.receipt.status != "applied" or not _action_matches(row.action, row.action, row.receipt.action) for row in rows):
        raise ValueError("training requires applied action evidence")
    cases = {}
    for row in rows:
        outcome = projection.outcome(deepcopy(row.after))
        if isinstance(outcome, Unknown):
            raise ValueError("unknown projected outcomes cannot be training labels")
        hash(outcome)
        cases[row.attempt_id] = (_features(projection, row.before, row.action), outcome)
    training = [cases[i] for i in train_ids]
    artifact = decision_list(training, candidate_literals(training), min_support=policy.min_training_support)
    domains: dict[str, set] = {}
    for facts, _ in training:
        for name, value in facts:
            domains.setdefault(name, set()).add(value)
    assignments = {i: (_matched(artifact, facts) if all(name not in domains or value in domains[name]
                                                        for name, value in facts) else None)
                   for i, (facts, _) in cases.items()}
    evidence: list[RuleEvidence] = []
    for index, rule in enumerate(artifact.rules):
        tr = tuple(i for i in train_ids if assignments[i] == index)
        ev = tuple(i for i in eval_ids if assignments[i] == index)
        tc = sum(cases[i][1] == rule.label for i in tr)
        ec = sum(cases[i][1] == rule.label for i in ev)
        reasons = []
        if len(tr) < policy.min_training_support:
            reasons.append("insufficient training support")
        if len(ev) < policy.min_evaluation_support:
            reasons.append("insufficient heldout support")
        if tr and tc / len(tr) < policy.min_accuracy:
            reasons.append("training contradictions exceed policy")
        if ev and ec / len(ev) < policy.min_accuracy:
            reasons.append("heldout contradictions exceed policy")
        evidence.append(RuleEvidence(index, tr, ev, tuple(s for i in tr + ev for s in by_id[i].source_ids), tc, ec, tuple(reasons)))
    predicted = [i for i in eval_ids if assignments[i] is not None and evidence[assignments[i]].verified]
    evaluation = Evaluation(len(eval_ids), len(predicted), sum(cases[i][1] == artifact.rules[assignments[i]].label for i in predicted))
    return LearnedTransitionModel(artifact, projection, tuple(evidence), evaluation, policy, next(iter(providers)),
                                  {name: frozenset(values) for name, values in domains.items()})
