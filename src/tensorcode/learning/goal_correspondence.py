"""Supervised Ref substitution across exact structured frame/goal examples.

No lexical role mapping is supplied. Only reference identity is abstracted;
all other syntax, qualifiers, literal values and goal invariants remain exact.
Examples are teaching data, not proof of intent or successful execution.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from itertools import combinations
import math
from uuid import uuid4

from ..goals import Condition, GoalSpec
from ..language import Entity, Frame
from ..records import Ref


@dataclass(frozen=True)
class GoalExample:
    id: str
    frame: Frame
    goal: GoalSpec
    basis: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrespondenceProposal:
    goal: GoalSpec
    template_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorrespondenceCandidates:
    proposals: tuple[CorrespondenceProposal, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Template:
    id: str
    frame: tuple
    goal: tuple
    constant_refs: tuple[Ref | None, ...]
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    conflicting_ids: tuple[str, ...]


_TYPES = {cls.__name__: cls for cls in (Frame, Entity, Condition)}


def _encode(value, refs, *, allow_new=True):
    if type(value) is Ref:
        if value not in refs:
            if not allow_new:
                raise ValueError('goal_reference_absent_from_frame')
            refs.append(value)
        return ('ref', refs.index(value))
    if value is None or type(value) in (bool, int, str, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError('nonfinite_value')
        return ('scalar', type(value).__name__, value)
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_encode(v, refs, allow_new=allow_new) for v in value))
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError('unsupported_mapping_key')
        return ('dict', tuple((k, _encode(value[k], refs, allow_new=allow_new)) for k in sorted(value)))
    if type(value) in _TYPES.values():
        return (type(value).__name__, tuple((f.name, _encode(getattr(value, f.name), refs, allow_new=allow_new)) for f in fields(value)))
    raise ValueError('unsupported_node_type:' + type(value).__name__)


def _decode(node, refs):
    kind = node[0]
    if kind == 'ref': return refs[node[1]]
    if kind == 'scalar': return node[2]
    if kind == 'tuple': return tuple(_decode(v, refs) for v in node[1])
    if kind == 'list': return [_decode(v, refs) for v in node[1]]
    values = {k: _decode(v, refs) for k, v in node[1]}
    if kind == 'dict': return values
    return _TYPES[kind](**values)


def _example(example):
    if (type(example) is not GoalExample or type(example.id) is not str or not example.id
            or type(example.frame) is not Frame or type(example.goal) is not GoalSpec
            or type(example.basis) is not tuple or any(type(x) is not str for x in example.basis)):
        raise ValueError('invalid_goal_example')
    refs = []
    frame = _encode(example.frame, refs)
    goal = _encode((example.goal.conditions, example.goal.invariants), refs, allow_new=False)
    return frame, goal, tuple(refs)


class GoalCorrespondenceModel:
    def __init__(self, training, validation, templates, complete, unresolved):
        self._id = 'goal-correspondence:' + uuid4().hex
        self._training = deepcopy(training)
        self._validation = deepcopy(validation)
        self._templates = tuple(templates)
        self._complete = complete
        self._unresolved = tuple(unresolved)

    @property
    def id(self): return self._id
    @property
    def complete(self): return self._complete
    @property
    def unresolved(self): return self._unresolved
    @property
    def training_examples(self): return deepcopy(self._training)
    @property
    def validation_examples(self): return deepcopy(self._validation)
    @property
    def templates(self): return deepcopy(self._templates)

    def propose(self, frame):
        refs = []
        try:
            if type(frame) is not Frame: raise ValueError('expected_frame')
            structure = _encode(frame, refs)
        except (ValueError, RecursionError) as error:
            return CorrespondenceCandidates((), False, (str(error),))
        proposals = []
        unsupported = []
        for template in self._templates:
            if (template.frame != structure
                    or any(constant is not None and refs[i] != constant
                           for i, constant in enumerate(template.constant_refs))):
                continue
            if not template.validation_ids:
                unsupported.append('unvalidated_correspondence:' + template.id)
                continue
            conditions, invariants = _decode(template.goal, refs)
            goal = GoalSpec(conditions, invariants=invariants,
                            basis=('learned-ref-correspondence:' + template.id,))
            proposals.append(CorrespondenceProposal(goal, (template.id,), template.training_ids,
                                                   template.validation_ids, template.conflicting_ids))
        unresolved = self._unresolved + tuple(unsupported) + (() if proposals else ('no_validated_correspondence',))
        return CorrespondenceCandidates(tuple(proposals), self.complete, unresolved)


def fit_correspondences(train_examples, validation_examples, *, max_pairs=256):
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError('max_pairs must be a positive integer')
    training, validation = tuple(deepcopy(tuple(train_examples))), tuple(deepcopy(tuple(validation_examples)))
    all_examples = (*training, *validation)
    if any(type(x) is not GoalExample or type(x.id) is not str or not x.id for x in all_examples):
        raise ValueError('invalid_goal_example')
    ids = [x.id for x in all_examples]
    if len(ids) != len(set(ids)): raise ValueError('example IDs must be unique and split-disjoint')
    encoded = {}
    unresolved = []
    for example in all_examples:
        try: encoded[example.id] = _example(example)
        except (ValueError, RecursionError) as error: unresolved.append(example.id + ':' + str(error))
    train_refs = {r for x in training if x.id in encoded for r in encoded[x.id][2]}
    eval_refs = {r for x in validation if x.id in encoded for r in encoded[x.id][2]}
    ref_sets = [set(encoded[x.id][2]) for x in all_examples if x.id in encoded]
    common_refs = set.intersection(*ref_sets) if ref_sets else set()
    if (train_refs & eval_refs) - common_refs:
        raise ValueError('validation variable entities must be disjoint from training')
    patterns = {}
    examined = 0
    complete = not unresolved
    for left, right in combinations(training, 2):
        if examined >= max_pairs:
            complete = False
            unresolved.append('pair_budget_exhausted')
            break
        examined += 1
        if left.id not in encoded or right.id not in encoded: continue
        lf, lg, lr = encoded[left.id]
        rf, rg, rr = encoded[right.id]
        # Equal references remain explicit constants. Only differing slots are
        # abstracted, preserving the full joint reference equality pattern.
        if lf != rf or lg != rg or not lr or lr == rr: continue
        constants = tuple(a if a == b else None for a, b in zip(lr, rr))
        patterns[(lf, lg, constants)] = True
    templates = []
    for (frame, goal, constants) in patterns:
        def matches(x):
            if x.id not in encoded or encoded[x.id][0] != frame: return False
            refs = encoded[x.id][2]
            return all(constant is None or refs[i] == constant for i, constant in enumerate(constants))
        train_ids = tuple(x.id for x in training if matches(x) and encoded[x.id][1] == goal)
        matching = tuple(x for x in validation if matches(x))
        for x in matching:
            if any(constant is None and encoded[x.id][2][i] in train_refs
                   for i, constant in enumerate(constants)):
                raise ValueError('validation variable entities must be disjoint from training')
        correct = tuple(x.id for x in matching if encoded[x.id][1] == goal)
        conflicting = tuple(x.id for x in matching if encoded[x.id][1] != goal)
        templates.append(_Template('template:' + uuid4().hex, frame, goal, constants, train_ids, correct, conflicting))
    return GoalCorrespondenceModel(training, validation, templates, complete, unresolved)
