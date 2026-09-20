"""Induce guarded transition predictions from retained execution observations.

Observation projection and validation policy are supplied, not learned. Only rule
conditions and outcome associations are induced. Predictions are proposals with
sample provenance; they never install capabilities or certify future effects.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from ..agent.interpretation import InterpretationSource
from ..agent.plugin import Call
from ..outcomes import Receipt, Unknown
from .induce import DecisionList, Rule, decision_list
from .literals import Literal, candidate_literals


@dataclass(frozen=True, order=True)
class ActionFamily:
    plugin: str
    capability: str
    argument_names: tuple[str, ...]


def action_family(action: Any) -> ActionFamily:
    if not isinstance(action, Call) or not isinstance(action.plugin, str) or not action.plugin or not isinstance(action.capability, str) or not action.capability:
        raise ValueError("an action family requires a named plugin and capability Call")
    try:
        pairs = tuple(action.args)
        if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
            raise ValueError("action arguments must be name/value pairs")
        names = tuple(pair[0] for pair in pairs)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("action argument names must be nonempty and unique")
    except TypeError as error:
        raise ValueError("invalid action argument contract") from error
    return ActionFamily(action.plugin, action.capability, tuple(sorted(names)))


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
class ProjectedExample:
    """The exact projected sample consumed by a fit, with its source linkage."""

    attempt_id: str
    provider: str
    source_ids: tuple[str, str]
    action: Call
    action_family: ActionFamily
    facts: frozenset
    outcome: Any
    split: str


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
        return len(left) == len(right) and all(
            any(_same(key, other_key) and _same(value, other_value) for other_key, other_value in right.items())
            for key, value in left.items())
    if isinstance(left, (set, frozenset)):
        return len(left) == len(right) and all(any(_same(value, other) for other in right) for value in left)
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
    """Supplied features(before, action) and outcome(before, action, after).

    Context selects a measured target; it does not license copying predicted
    labels from the action. Both fit and evidence replay use detached inputs.
    """
    name: str
    features: Callable[[Any, Any], Mapping[str, Any]]
    outcome: Callable[[Any, Any, Any], Any]
    provenance: tuple[str, ...]
    kind: str = "authored"

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str) or not self.name.strip() or self.kind != "authored"
                or not isinstance(self.provenance, tuple) or not self.provenance
                or any(not isinstance(item, str) or not item for item in self.provenance)
                or not callable(self.features) or not callable(self.outcome)):
            raise ValueError("projection must name its supplied authored semantics and immutable provenance")


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
    action_family: ActionFamily | None = None

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
    model_id: str
    model_revision: int
    rule_id: str
    action_family: ActionFamily


@dataclass(frozen=True)
class RuleSuspension:
    revision: int
    rule_id: str
    source_ids: tuple[str, ...]
    reason: str
    predicted: Any
    observed: Any


@dataclass(frozen=True)
class ModelSnapshot:
    id: str
    revision: int
    rule_ids: tuple[str, ...]
    action_families: tuple[ActionFamily, ...]
    projection: str
    projection_provenance: tuple[str, ...]
    source_ids: tuple[str, ...]
    history: tuple[RuleSuspension, ...]


def _features(projection: Projection, before: Any, action: Any) -> frozenset:
    mapping = projection.features(deepcopy(before), deepcopy(action))
    if not isinstance(mapping, Mapping) or not mapping or any(not isinstance(k, str) for k in mapping):
        raise ValueError("projection must return a nonempty mapping with string feature names")
    if any(isinstance(value, Unknown) for value in mapping.values()):
        raise ValueError("unknown feature values cannot become observed facts")
    return frozenset(mapping.items())


@dataclass
class ResidualRule(Rule):
    """Pure observed residual conjunction, including its exact feature-name set.

    Unlike induced partial conditions, this conservative extension makes no
    generalization across feature combinations. Validation still gates use.
    """
    feature_names: tuple[str, ...] = ()

    def matches(self, facts: Any) -> bool:
        mapping = dict(facts.items() if callable(getattr(facts, "items", None)) else facts)
        return (set(mapping) == set(self.feature_names)
                and all(condition.predicate in mapping
                        and _same(mapping[condition.predicate], condition.value)
                        for condition in self.conditions))


def _retain_pure_residuals(artifact, training, min_support):
    groups = []
    for facts, outcome in training:
        if _matched(artifact, facts) is not None:
            continue
        group = next((group for group in groups if _same(dict(group[0]), dict(facts))), None)
        if group is None:
            groups.append([facts, [outcome]])
        else:
            group[1].append(outcome)
    for facts, outcomes in groups:
        if not facts or len(outcomes) < min_support or not all(_same(outcomes[0], value) for value in outcomes):
            continue
        ordered = sorted(facts, key=lambda pair: pair[0])
        artifact.rules.append(ResidualRule(tuple(Literal(name, value) for name, value in ordered),
            deepcopy(outcomes[0]), len(outcomes), len(outcomes), tuple(name for name, _ in ordered)))


def _matched(artifact: DecisionList, facts: frozenset) -> int | None:
    mapping = dict(facts)
    for index, rule in enumerate(artifact.rules):
        if isinstance(rule, ResidualRule) and set(mapping) != set(rule.feature_names):
            continue
        # Negated equality must not make an absent observation count as false.
        if any(condition.predicate not in mapping for condition in rule.conditions):
            return None
        matches = True
        for condition in rule.conditions:
            actual = mapping[condition.predicate]
            if condition.kind == "present":
                holds = True
            elif condition.kind == "at_least":
                if type(actual) is not type(condition.value) or type(actual) not in (int, float):
                    matches = False
                    break
                holds = actual >= condition.value
            else:
                # A different type cannot supply either positive or negative
                # evidence for a literal learned over another value domain.
                if type(actual) is not type(condition.value):
                    matches = False
                    break
                holds = _same(actual, condition.value)
            if condition.negated:
                holds = not holds
            if not holds:
                matches = False
                break
        if matches:
            return index
    return None


class LearnedTransitionModel:
    """Sample-validated rules, scoped action contracts, and monotonic suspension."""

    def __init__(self, artifact: DecisionList, projection: Projection, evidence: tuple[RuleEvidence, ...],
                 evaluation: Evaluation, policy: ValidationPolicy, provider: str,
                 feature_values: Mapping[str, tuple[Any, ...]],
                 family_evidence: Mapping[tuple[int, ActionFamily], RuleEvidence],
                 action_families: tuple[ActionFamily, ...], source_ids: tuple[str, ...],
                 examples: tuple[ProjectedExample, ...] | None = None) -> None:
        self._artifact = deepcopy(artifact)
        self._projection = projection
        self._evidence = deepcopy(evidence)
        self._evaluation, self._policy, self._provider = evaluation, policy, provider
        self._feature_values = deepcopy(dict(feature_values))
        self._family_evidence = deepcopy(dict(family_evidence))
        self._action_families = tuple(action_families)
        self._source_ids = tuple(source_ids)
        saved_examples = deepcopy(examples or ())
        self._examples = {example.attempt_id: example for example in saved_examples}
        if len(self._examples) != len(saved_examples):
            raise ValueError("fit snapshots require unique attempt IDs")
        self._id = f"transition-model:{uuid4().hex}"
        self._revision = 0
        self._history: tuple[RuleSuspension, ...] = ()

    @property
    def id(self) -> str:
        return self._id

    @property
    def model_id(self) -> str:
        return self._id

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def projection(self) -> Projection:
        return self._projection

    @property
    def evidence(self) -> tuple[RuleEvidence, ...]:
        return self._evidence

    @property
    def evaluation(self) -> Evaluation:
        return self._evaluation

    @property
    def policy(self) -> ValidationPolicy:
        return self._policy

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def action_families(self) -> tuple[ActionFamily, ...]:
        return self._action_families

    @property
    def examples(self) -> tuple[ProjectedExample, ...]:
        """Detached projected samples, including unassigned/default-only cases."""
        return deepcopy(tuple(self._examples.values()))

    def validate_transition(self, transition: Transition) -> bool | Unknown:
        """Compare retained execution evidence with the sample actually fitted.

        Source IDs alone cannot establish that a fitted label came from those
        sources. Reprojecting the supplied actual transition checks precisely the
        representation the learner consumed. Fields ignored by the projection
        intentionally do not participate, except full action/provenance identity.
        """
        if not isinstance(transition, Transition) or not isinstance(transition.attempt_id, str):
            return Unknown("fit_evidence_mismatch", "expected a retained transition with an attempt identity")
        example = self._examples.get(transition.attempt_id)
        if example is None:
            return Unknown("missing_fit_example", "no projected fit snapshot for this attempt")
        try:
            if (transition.provider != example.provider or transition.source_ids != example.source_ids
                    or action_family(transition.action) != example.action_family
                    or not _same(transition.action, example.action)
                    or not isinstance(transition.receipt, Receipt) or transition.receipt.status != "applied"
                    or not _action_matches(transition.action, transition.action, transition.receipt.action)):
                return Unknown("fit_evidence_mismatch", "action or source linkage differs from the fitted sample")
            facts = _features(self.projection, transition.before, transition.action)
            outcome = self.projection.outcome(deepcopy(transition.before), deepcopy(transition.action), deepcopy(transition.after))
            if (isinstance(outcome, Unknown) or not _same(dict(facts), dict(example.facts))
                    or not _same(outcome, example.outcome)):
                return Unknown("fit_evidence_mismatch", "projected observation differs from the fitted sample")
        except Exception as error:
            return Unknown("fit_evidence_unavailable", f"cannot reproduce projected sample: {type(error).__name__}: {error}")
        return True

    @property
    def history(self) -> tuple[RuleSuspension, ...]:
        return deepcopy(self._history)

    @property
    def artifact(self) -> DecisionList:
        return deepcopy(self._artifact)

    def rule_id(self, index: int) -> str:
        if type(index) is not int or not 0 <= index < len(self._artifact.rules):
            raise ValueError("unknown rule index")
        return f"{self.id}/rule:{index}"

    def snapshot(self) -> ModelSnapshot:
        return ModelSnapshot(self.id, self.revision, tuple(self.rule_id(i) for i in range(len(self._artifact.rules))),
                             self.action_families, self.projection.name, self.projection.provenance,
                             self._source_ids, self.history)

    def is_current(self, prediction: TransitionPrediction) -> bool:
        if (not self._examples or not isinstance(prediction, TransitionPrediction) or prediction.model_id != self.id
                or prediction.model_revision != self.revision or type(prediction.rule_index) is not int
                or prediction.rule_index not in range(len(self._artifact.rules))):
            return False
        try:
            return (prediction.rule_id == self.rule_id(prediction.rule_index)
                    and prediction.projection == self.projection.name
                    and prediction.projection_provenance == self.projection.provenance
                    and prediction.action_family in self.action_families
                    and isinstance(prediction.evidence, RuleEvidence) and prediction.evidence.verified
                    and prediction.evidence == self._family_evidence.get((prediction.rule_index, prediction.action_family))
                    and all(attempt in self._examples for attempt in prediction.evidence.training_attempt_ids + prediction.evidence.evaluation_attempt_ids)
                    and _same(prediction.outcome, self._artifact.rules[prediction.rule_index].label)
                    and not any(event.rule_id == prediction.rule_id for event in self._history))
        except (TypeError, ValueError, AttributeError):
            return False

    def observe_outcome(self, prediction: TransitionPrediction, actual_outcome: Any, *,
                        source_ids: Iterable[str], reason: str) -> RuleSuspension | None:
        """Suspend a contradicted rule; a new fit is required to authorize it again.

        The caller supplies an observed projected outcome and its source IDs, not
        another model's prediction. Stale revisions from this same fixed rule set
        may still report counterexamples; foreign or altered predictions cannot.
        """
        sources = tuple(source_ids)
        if not sources or any(not isinstance(s, str) or not s for s in sources) or not isinstance(reason, str) or not reason.strip():
            raise ValueError("counterexamples require source IDs and a reason")
        if (not isinstance(prediction, TransitionPrediction) or prediction.model_id != self.id
                or type(prediction.rule_index) is not int
                or prediction.rule_index not in range(len(self._artifact.rules))
                or prediction.rule_id != self.rule_id(prediction.rule_index)
                or type(prediction.model_revision) is not int or not 0 <= prediction.model_revision <= self.revision
                or prediction.action_family not in self.action_families
                or prediction.projection != self.projection.name
                or prediction.projection_provenance != self.projection.provenance
                or prediction.evidence != self._family_evidence.get((prediction.rule_index, prediction.action_family))
                or not _same(prediction.outcome, self._artifact.rules[prediction.rule_index].label)):
            raise ValueError("prediction does not identify this model's unchanged rule")
        if isinstance(actual_outcome, Unknown):
            raise ValueError("an unknown outcome is not contradictory evidence")
        if _same(prediction.outcome, actual_outcome):
            return None
        self._revision += 1
        event = RuleSuspension(self.revision, prediction.rule_id, tuple(dict.fromkeys(sources)), reason,
                               deepcopy(prediction.outcome), deepcopy(actual_outcome))
        self._history += (event,)
        return deepcopy(event)

    def predict(self, before: Any, action: Any) -> TransitionPrediction | Unknown:
        if not self._examples:
            return Unknown("missing_fit_examples", "supplied artifact has no retained projected training/evaluation samples")
        try:
            family = action_family(action)
        except ValueError as error:
            return Unknown("invalid_action_family", str(error))
        if family not in self._action_families:
            return Unknown("unseen_action_family", "plugin, capability, or argument names lack training evidence")
        try:
            facts = _features(self.projection, before, action)
        except (ValueError, TypeError, KeyError) as error:
            return Unknown("unprojectable_observation", str(error))
        if any(name not in self._feature_values for name, _ in facts):
            return Unknown("unseen_feature", "projection produced feature names absent from training")
        if any(not any(_same(value, seen) for seen in self._feature_values[name]) for name, value in facts):
            return Unknown("unseen_feature_value", "prediction lies outside observed training feature values")
        index = _matched(self._artifact, facts)
        if index is None:
            return Unknown("unsupported_transition", "no explicit induced rule covers the observed features")
        identity = self.rule_id(index)
        if any(event.rule_id == identity for event in self._history):
            return Unknown("suspended_rule", "observed counterexample suspended this rule; refit and revalidate")
        evidence = self._family_evidence.get((index, family))
        if evidence is None or not evidence.verified:
            return Unknown("unverified_transition", "; ".join(evidence.reasons) if evidence else "no action-family rule evidence")
        if any(attempt not in self._examples for attempt in evidence.training_attempt_ids + evidence.evaluation_attempt_ids):
            return Unknown("missing_fit_examples", "rule evidence lacks retained projected samples")
        return TransitionPrediction(deepcopy(self._artifact.rules[index].label), index, evidence,
                                    self.projection.name, self.projection.provenance, self.id, self.revision,
                                    identity, family)


def fit_transitions(transitions: Iterable[Transition], *, projection: Projection,
                    train_attempt_ids: Iterable[str], evaluation_attempt_ids: Iterable[str],
                    policy: ValidationPolicy = ValidationPolicy()) -> LearnedTransitionModel:
    """Fit on explicit training attempts; validate rules on disjoint heldout attempts.

    Every supplied transition must be assigned to exactly one split. Evaluation
    gates learned rules but never changes their conditions or labels. Its scores
    describe this validation set, not an untouched final test-set estimate.
    """
    rows = list(deepcopy(tuple(transitions)))
    if any(not isinstance(row, Transition) or not isinstance(row.attempt_id, str) or not row.attempt_id
           or not isinstance(row.provider, str) or not row.provider.startswith("plugin:")
           or len(row.source_ids) != 2 or any(not isinstance(sid, str) or not sid for sid in row.source_ids)
           for row in rows):
        raise ValueError("transitions require attempt, plugin provider, and paired source identities")
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
        outcome = projection.outcome(deepcopy(row.before), deepcopy(row.action), deepcopy(row.after))
        if isinstance(outcome, Unknown):
            raise ValueError("unknown projected outcomes cannot be training labels")
        hash(outcome)
        cases[row.attempt_id] = (_features(projection, row.before, row.action), outcome)
    families = {row.attempt_id: action_family(row.action) for row in rows}
    learned_families = tuple(sorted({families[i] for i in train_ids}))
    training = [cases[i] for i in train_ids]
    artifact = decision_list(training, candidate_literals(training), min_support=policy.min_training_support)
    _retain_pure_residuals(artifact, training, policy.min_training_support)
    domains: dict[str, list] = {}
    for facts, _ in training:
        for name, value in facts:
            observed = domains.setdefault(name, [])
            if not any(_same(value, seen) for seen in observed):
                observed.append(value)
    assignments = {i: (_matched(artifact, facts) if families[i] in learned_families and all(name in domains and any(_same(value, seen) for seen in domains[name])
                                                        for name, value in facts) else None)
                   for i, (facts, _) in cases.items()}
    evidence: list[RuleEvidence] = []
    for index, rule in enumerate(artifact.rules):
        tr = tuple(i for i in train_ids if assignments[i] == index)
        ev = tuple(i for i in eval_ids if assignments[i] == index)
        tc = sum(_same(cases[i][1], rule.label) for i in tr)
        ec = sum(_same(cases[i][1], rule.label) for i in ev)
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
    family_evidence = {}
    for index, rule in enumerate(artifact.rules):
        for family in learned_families:
            tr = tuple(i for i in train_ids if assignments[i] == index and families[i] == family)
            ev = tuple(i for i in eval_ids if assignments[i] == index and families[i] == family)
            tc = sum(_same(cases[i][1], rule.label) for i in tr)
            ec = sum(_same(cases[i][1], rule.label) for i in ev)
            reasons = []
            if len(tr) < policy.min_training_support:
                reasons.append("insufficient action-family training support")
            if len(ev) < policy.min_evaluation_support:
                reasons.append("insufficient action-family heldout support")
            if tr and tc / len(tr) < policy.min_accuracy:
                reasons.append("action-family training contradictions exceed policy")
            if ev and ec / len(ev) < policy.min_accuracy:
                reasons.append("action-family heldout contradictions exceed policy")
            family_evidence[index, family] = RuleEvidence(index, tr, ev,
                tuple(s for i in tr + ev for s in by_id[i].source_ids), tc, ec, tuple(reasons), family)
    predicted = [i for i in eval_ids if assignments[i] is not None and family_evidence[assignments[i], families[i]].verified]
    evaluation = Evaluation(len(eval_ids), len(predicted), sum(_same(cases[i][1], artifact.rules[assignments[i]].label) for i in predicted))
    return LearnedTransitionModel(artifact, projection, tuple(evidence), evaluation, policy, next(iter(providers)),
                                  {name: tuple(values) for name, values in domains.items()}, family_evidence, learned_families,
                                  tuple(source_ids), tuple(ProjectedExample(
                                      row.attempt_id, row.provider, row.source_ids, row.action, families[row.attempt_id],
                                      cases[row.attempt_id][0], cases[row.attempt_id][1],
                                      "training" if row.attempt_id in train else "evaluation") for row in rows))
