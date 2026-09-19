"""Finite empirical dynamics with source-bound support for every observed outcome.

State projection, split assignment and support thresholds are authored. Edges and
successors are aggregated from executed transitions. Empirical support is neither
exhaustive environment coverage nor a probability distribution. States currently
support primitive scalars and recursive tuples, using typed structural equality.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Iterable
from uuid import uuid4

from ..agent.plugin import Call
from ..outcomes import Receipt, Unknown
from .experience import Transition, _same, action_family


@dataclass(frozen=True)
class StateProjection:
    name: str
    state: Callable[[Any], Any]
    provenance: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or not callable(self.state):
            raise ValueError('state projection requires a name and callable')
        if (not isinstance(self.provenance, tuple) or not self.provenance
                or any(not isinstance(p, str) or not p.strip() for p in self.provenance)):
            raise ValueError('state projection requires explicit provenance')


@dataclass(frozen=True)
class DynamicsPolicy:
    min_training_support: int = 2
    min_evaluation_support: int = 1

    def __post_init__(self):
        for value in (self.min_training_support, self.min_evaluation_support):
            if type(value) is not int or value < 1:
                raise ValueError('dynamics support thresholds must be positive integers')


@dataclass(frozen=True)
class OutcomeEvidence:
    state: Any
    training_attempt_ids: tuple[str, ...]
    evaluation_attempt_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class DynamicsPrediction:
    state: Any
    call: Call
    outcomes: tuple[OutcomeEvidence, ...]
    model_id: str


@dataclass(frozen=True)
class DynamicsExample:
    attempt_id: str
    provider: str
    source_ids: tuple[str, str]
    state: Any
    call: Call
    successor: Any
    split: str


@dataclass(frozen=True)
class DynamicsEdge:
    state: Any
    call: Call
    outcomes: tuple[OutcomeEvidence, ...]
    eligible: bool


def _state(value):
    if type(value) in (type(None), bool, int, str, bytes):
        return value
    if type(value) is float and isfinite(value):
        return value
    if type(value) is tuple:
        return tuple(_state(item) for item in value)
    raise ValueError('states must be finite primitive scalars or recursive tuples')


def _distinct(values):
    result = []
    for value in values:
        if not any(_same(value, existing) for existing in result):
            result.append(deepcopy(value))
    return tuple(result)


def _prepare(transitions, projection, train_ids, eval_ids, policy):
    if not isinstance(projection, StateProjection) or not isinstance(policy, DynamicsPolicy):
        raise TypeError('supply a StateProjection and support-only DynamicsPolicy')
    # Revalidate frozen records too: object.__setattr__ is not evidence authority.
    projection.__post_init__()
    policy.__post_init__()
    rows = deepcopy(tuple(transitions))
    if any(not isinstance(row, Transition) or not isinstance(row.attempt_id, str) or not row.attempt_id
           or not isinstance(row.provider, str) or not row.provider.startswith('plugin:')
           or len(row.source_ids) != 2 or any(not isinstance(sid, str) or not sid for sid in row.source_ids)
           for row in rows):
        raise ValueError('transitions require attempts, plugin providers and paired source IDs')
    by_id = {row.attempt_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate attempt IDs')
    if any(not isinstance(value, str) or not value for value in (*train_ids, *eval_ids)):
        raise ValueError('split attempt IDs must be nonempty strings')
    train, held = set(train_ids), set(eval_ids)
    if not train or not held or len(train) != len(train_ids) or len(held) != len(eval_ids):
        raise ValueError('both splits require unique nonempty attempt IDs')
    if train & held or train | held != set(by_id):
        raise ValueError('splits must disjointly partition all supplied attempts')
    providers = {row.provider for row in rows}
    source_ids = tuple(sid for row in rows for sid in row.source_ids)
    if len(providers) != 1 or len(source_ids) != len(set(source_ids)):
        raise ValueError('require one provider and independent observation source IDs')
    examples = []
    for row in rows:
        action_family(row.action)
        if not isinstance(row.receipt, Receipt) or row.receipt.status != 'applied' or not _same(row.action, row.receipt.action):
            raise ValueError('dynamics fitting requires matching applied receipts')
        examples.append(DynamicsExample(row.attempt_id, row.provider, tuple(row.source_ids),
            _state(projection.state(deepcopy(row.before))), deepcopy(row.action),
            _state(projection.state(deepcopy(row.after))), 'training' if row.attempt_id in train else 'evaluation'))
    edges = []
    for example in examples:
        if any(_same(example.state, edge.state) and _same(example.call, edge.call) for edge in edges):
            continue
        matches = tuple(item for item in examples if _same(item.state, example.state) and _same(item.call, example.call))
        outcomes = []
        for successor in _distinct(item.successor for item in matches):
            samples = tuple(item for item in matches if _same(item.successor, successor))
            outcomes.append(OutcomeEvidence(successor,
                tuple(item.attempt_id for item in samples if item.split == 'training'),
                tuple(item.attempt_id for item in samples if item.split == 'evaluation'),
                tuple(sid for item in samples for sid in item.source_ids)))
        eligible = all(len(outcome.training_attempt_ids) >= policy.min_training_support
                       and len(outcome.evaluation_attempt_ids) >= policy.min_evaluation_support for outcome in outcomes)
        edges.append(DynamicsEdge(example.state, deepcopy(example.call), tuple(outcomes), eligible))
    return rows, tuple(examples), tuple(edges), next(iter(providers)), source_ids


@dataclass(frozen=True, init=False)
class EmpiricalDynamics:
    """Immutable fit; copies returned to callers never share mutable call payloads.

    The constructor accepts source transitions, never precomputed edges. Agent
    execution must still authenticate these supplied rows against retained raw
    observation sources with ``validate_transitions``. Fitting alone cannot prove
    that supplied transitions were executed.
    """
    _rows: tuple
    _examples: tuple
    _edges: tuple
    _projection: StateProjection
    _policy: DynamicsPolicy
    _train_ids: tuple
    _eval_ids: tuple
    _provider: str
    _source_ids: tuple
    _id: str

    def __init__(self, transitions, *, projection, train_attempt_ids, evaluation_attempt_ids,
                 policy=DynamicsPolicy()):
        train, held = tuple(train_attempt_ids), tuple(evaluation_attempt_ids)
        rows, examples, edges, provider, source_ids = _prepare(transitions, projection, train, held, policy)
        for name, value in (('_rows', rows), ('_examples', examples), ('_edges', edges),
                            ('_projection', projection), ('_policy', policy), ('_train_ids', train),
                            ('_eval_ids', held), ('_provider', provider), ('_source_ids', source_ids),
                            ('_id', 'empirical-dynamics:' + uuid4().hex)):
            object.__setattr__(self, name, value)

    @property
    def id(self): return self._id
    @property
    def revision(self): return 0
    @property
    def provider(self): return self._provider
    @property
    def projection(self): return self._projection
    @property
    def policy(self): return deepcopy(self._policy)
    @property
    def examples(self): return deepcopy(self._examples)
    @property
    def edges(self): return deepcopy(self._edges)
    @property
    def source_ids(self): return self._source_ids
    @property
    def states(self): return _distinct(state for item in self._examples for state in (item.state, item.successor))
    @property
    def calls(self): return _distinct(item.call for item in self._examples)

    def predict(self, state, call):
        try:
            state = _state(state)
            action_family(call)
        except (TypeError, ValueError):
            return Unknown('invalid_dynamics_query')
        edge = next((item for item in self._edges if _same(item.state, state) and _same(item.call, call)), None)
        if edge is None:
            return Unknown('unsupported_dynamics_transition')
        if not edge.eligible:
            return Unknown('insufficient_outcome_support', 'every observed successor requires separate training and validation support')
        return DynamicsPrediction(deepcopy(edge.state), deepcopy(edge.call), deepcopy(edge.outcomes), self.id)

    def validate_transitions(self, transitions: Iterable[Transition]):
        """Recompute the complete fit from exact externally authenticated rows.

        The caller supplies all fitted attempts, including both splits, and may
        include later executions. New outcomes on known state/action edges are
        counterexamples: they require a refit before this model can authorize more
        planning. This assumes the caller's explicit same-provider context scope.
        Missing rows, source aliases, altered actions or observations, drifted
        projections, and tampered cached edges invalidate the model boundary.
        """
        try:
            supplied = tuple(transitions)
            by_id = {row.attempt_id: row for row in supplied}
            fitted_ids = {row.attempt_id for row in self._rows}
            if len(by_id) != len(supplied) or not fitted_ids <= set(by_id):
                return Unknown('dynamics_evidence_mismatch', 'fitted attempts must be covered exactly once')
            ordered = tuple(by_id[row.attempt_id] for row in self._rows)
            if not _same(ordered, self._rows):
                return Unknown('dynamics_evidence_mismatch', 'raw execution evidence differs from fit')
            _, examples, edges, provider, source_ids = _prepare(ordered, self._projection, self._train_ids, self._eval_ids, self._policy)
            if not (_same(examples, self._examples) and _same(edges, self._edges)
                    and provider == self._provider and source_ids == self._source_ids):
                return Unknown('dynamics_evidence_mismatch', 'recomputed fit differs from retained model')
            for row in supplied:
                if row.attempt_id in fitted_ids or row.provider != self.provider:
                    continue
                action_family(row.action)
                if not isinstance(row.receipt, Receipt) or row.receipt.status != 'applied' or not _same(row.action, row.receipt.action):
                    return Unknown('dynamics_evidence_mismatch', 'later transitions require matching applied receipts')
                if not any(_same(row.action, edge.call) for edge in edges):
                    continue
                state = _state(self.projection.state(deepcopy(row.before)))
                edge = next((item for item in edges if _same(item.state, state) and _same(item.call, row.action)), None)
                if edge is None:
                    continue
                successor = _state(self.projection.state(deepcopy(row.after)))
                if not any(_same(successor, outcome.state) for outcome in edge.outcomes):
                    return Unknown('empirical_counterexample', row.attempt_id)
        except Exception as error:
            return Unknown('dynamics_evidence_mismatch', f'{type(error).__name__}: {error}')
        return True


def fit_dynamics(transitions: Iterable[Transition], *, projection: StateProjection,
                 train_attempt_ids: Iterable[str], evaluation_attempt_ids: Iterable[str],
                 policy: DynamicsPolicy = DynamicsPolicy()) -> EmpiricalDynamics:
    return EmpiricalDynamics(transitions, projection=projection, train_attempt_ids=train_attempt_ids,
                             evaluation_attempt_ids=evaluation_attempt_ids, policy=policy)
