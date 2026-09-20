"""Supervised full-question to informing-plan correspondences.

Only input reference identities are abstracted. Words, qualifiers, query roles,
capability names, polarity and metadata remain supplied literal teaching data.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import date, datetime
from itertools import combinations
from uuid import uuid4
import math

from ..language import Entity, Frame, Question
from ..records import Interval, Proposition, Ref, Var


@dataclass(frozen=True)
class InformingPlan:
    plugin: str
    capability: str
    args: tuple[tuple[str, object], ...]
    answer_query: Proposition
    answer_variable: str

    def __post_init__(self):
        if any(type(x) is not str or not x.strip() for x in (self.plugin, self.capability, self.answer_variable)):
            raise ValueError('informing requires exact plugin, capability and answer variable names')
        if (type(self.args) is not tuple or any(type(item) is not tuple or len(item) != 2
                or type(item[0]) is not str or not item[0].strip() for item in self.args)
                or len({item[0] for item in self.args}) != len(self.args)):
            raise ValueError('arguments require unique explicit names')
        _validate_answer_query(self.answer_query, self.answer_variable)
        if _query_variables(self.args):
            raise ValueError('informing arguments must be bound, not query variables')


def _query_variables(value):
    if type(value) is Var: return {value.name}
    if type(value) in (Proposition, Interval):
        return set().union(*(_query_variables(getattr(value, f.name)) for f in fields(value)))
    if type(value) is dict:
        return set().union(*(_query_variables(v) for v in value.values())) if value else set()
    if type(value) in (tuple, list):
        return set().union(*(_query_variables(v) for v in value)) if value else set()
    return set()


def _validate_answer_query(query, answer_variable):
    if type(query) is not Proposition:
        raise ValueError('answer query must be an explicit Proposition')
    if type(answer_variable) is not str or not answer_variable.strip():
        raise ValueError('answer variable must be an explicit nonempty name')
    if answer_variable not in _query_variables(query):
        raise ValueError('answer variable must occur in explicit query')



@dataclass(frozen=True)
class InformingExample:
    id: str
    source_id: str
    text: str
    question: Question
    plan: InformingPlan
    basis: tuple[str, ...] = ()


@dataclass(frozen=True)
class InformingProposal:
    plan: InformingPlan
    template_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...]
    conflicting_training_example_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class InformingCandidates:
    proposals: tuple[InformingProposal, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class InformingTemplate:
    id: str
    question: tuple
    plan: tuple
    constant_refs: tuple[Ref | None, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...]
    conflicting_training_example_ids: tuple[str, ...] = ()


_TYPES = {cls.__name__: cls for cls in (Entity, Frame, Question, Proposition, Interval, Var, InformingPlan)}


def _encode(value, refs, *, input_value=True, types=None):
    types = _TYPES if types is None else types
    if type(value) is Ref:
        if value not in refs:
            if not input_value: return ('constant_ref', value.id)
            refs.append(value)
        return ('ref', refs.index(value))
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value): raise ValueError('nonfinite informing value')
        return ('scalar', type(value).__name__, value)
    if type(value) in (date, datetime): return (type(value).__name__, value.isoformat())
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_encode(v, refs, input_value=input_value, types=types) for v in value))
    if type(value) is dict:
        if any(type(k) is not str for k in value): raise ValueError('unsupported mapping keys')
        return ('dict', tuple((k, _encode(value[k], refs, input_value=input_value, types=types)) for k in sorted(value)))
    if type(value) in types.values():
        return (type(value).__name__, tuple((f.name, _encode(getattr(value, f.name), refs, input_value=input_value, types=types)) for f in fields(value)))
    raise ValueError('unsupported informing value:' + type(value).__name__)


def _decode(node, refs, types=None):
    types = _TYPES if types is None else types
    kind = node[0]
    if kind == 'ref': return refs[node[1]]
    if kind == 'constant_ref': return Ref(node[1])
    if kind == 'scalar': return node[2]
    if kind == 'date': return date.fromisoformat(node[1])
    if kind == 'datetime': return datetime.fromisoformat(node[1])
    if kind == 'tuple': return tuple(_decode(v, refs, types) for v in node[1])
    if kind == 'list': return [_decode(v, refs, types) for v in node[1]]
    values = {k: _decode(v, refs, types) for k, v in node[1]}
    return values if kind == 'dict' else types[kind](**values)


def _question(question):
    if type(question) is not Question or type(question.frame) is not Frame or type(question.asked) is not str or not question.asked:
        raise ValueError('full explicit Question required')
    refs = []
    return _encode(question, refs), refs


class InformingModel:
    _kind = 'informing'
    _codec = _TYPES
    _proposal_type = InformingProposal
    _candidates_type = InformingCandidates

    @staticmethod
    def _coverage(plan):
        return plan.args, plan.answer_query

    def __init__(self, training, validation, templates, complete, unresolved):
        self._id = self._kind + ':' + uuid4().hex
        self._training, self._validation = deepcopy(training), deepcopy(validation)
        self._templates, self._complete, self._unresolved = tuple(templates), complete, tuple(unresolved)

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

    def propose(self, question):
        try: shape, refs = _question(question)
        except (ValueError, TypeError, RecursionError) as error:
            return self._candidates_type((), False, (str(error),))
        proposals, unresolved = [], list(self.unresolved)
        active = tuple(template for template in self._templates if shape == template.question
            and all(value is None or refs[i] == value for i, value in enumerate(template.constant_refs)))
        supported_plans = {template.plan for template in active if template.validation_example_ids}
        training = {example.id: example for example in self._training}
        for template in active:
            for ident in template.conflicting_training_example_ids:
                example = training[ident]
                _, example_refs = _question(example.question)
                rival = _encode(example.plan, example_refs, input_value=False, types=self._codec)
                if rival not in supported_plans:
                    unresolved.append('unvalidated_training_rival:' + ident)
            if not template.validation_example_ids:
                unresolved.append('unvalidated_' + self._kind + ':' + template.id)
                continue
            proposals.append(self._proposal_type(_decode(template.plan, refs, self._codec), (template.id,),
                template.training_example_ids, template.validation_example_ids,
                template.conflicting_validation_example_ids, template.conflicting_training_example_ids))
        if not proposals: unresolved.append('no_validated_' + self._kind)
        return self._candidates_type(tuple(proposals), self.complete, tuple(dict.fromkeys(unresolved)))


def fit_informing(training, validation, *, max_pairs=256):
    return _fit_question_plans(training, validation, max_pairs=max_pairs,
        example_type=InformingExample, plan_type=InformingPlan, model_type=InformingModel)


def _fit_question_plans(training, validation, *, max_pairs, example_type, plan_type, model_type):
    """Shared exact-question/Ref abstraction; concrete plans own their contracts."""
    if type(max_pairs) is not int or max_pairs < 1: raise ValueError('max_pairs must be positive')
    training, validation = deepcopy(tuple(training)), deepcopy(tuple(validation))
    examples = (*training, *validation)
    for x in examples:
        if (type(x) is not example_type or type(x.plan) is not plan_type
                or any(type(v) is not str or not v for v in (x.id, x.source_id, x.text))
                or type(x.basis) is not tuple or any(type(v) is not str for v in x.basis)):
            raise ValueError('invalid informing example')
        x.plan.__post_init__()
    if any(len({getattr(x, field) for x in examples}) != len(examples) for field in ('id', 'source_id')):
        raise ValueError('example and source IDs must be independent')
    texts = [' '.join(x.text.split()).casefold() for x in examples]
    if len(set(texts)) != len(texts): raise ValueError('teaching texts must be independent')
    encoded, unresolved = {}, []
    for x in examples:
        try:
            shape, refs = _question(x.question)
            projection = _encode(model_type._coverage(x.plan), refs, input_value=False, types=model_type._codec)
            def covered(node):
                if type(node) is not tuple: return set()
                if len(node) == 2 and node[0] == 'ref' and type(node[1]) is int: return {node[1]}
                return set().union(*(covered(item) for item in node)) if node else set()
            if covered(projection) != set(range(len(refs))):
                raise ValueError('unconsumed_question_reference: executable query plan omits an input reference')
            encoded[x.id] = (shape, _encode(x.plan, refs, input_value=False, types=model_type._codec), tuple(refs))
        except (ValueError, TypeError, RecursionError) as error: unresolved.append(x.id + ':' + str(error))
    train_refs = {r for x in training if x.id in encoded for r in encoded[x.id][2]}
    held_refs = {r for x in validation if x.id in encoded for r in encoded[x.id][2]}
    populations = [set(encoded[x.id][2]) for x in examples if x.id in encoded]
    common = set.intersection(*populations) if populations else set()
    if (train_refs & held_refs) - common: raise ValueError('heldout references must be disjoint')
    patterns, complete = {}, not unresolved
    for index, (left, right) in enumerate(combinations(training, 2)):
        if index >= max_pairs:
            complete = False
            unresolved.append('pair_budget_exhausted')
            break
        if left.id not in encoded or right.id not in encoded: continue
        lq, lp, lr = encoded[left.id]
        rq, rp, rr = encoded[right.id]
        if lq != rq or lp != rp or not lr or lr == rr: continue
        constants = tuple(a if a == b else None for a, b in zip(lr, rr))
        patterns[(lq, lp, constants)] = True
    templates = []
    for question, plan, constants in patterns:
        def matches(x):
            return (x.id in encoded and encoded[x.id][0] == question
                    and all(value is None or encoded[x.id][2][i] == value for i, value in enumerate(constants)))
        train = tuple(x.id for x in training if matches(x) and encoded[x.id][1] == plan)
        training_conflicts = tuple(x.id for x in training if matches(x) and encoded[x.id][1] != plan)
        held, conflicts = [], []
        for x in validation:
            if not matches(x): continue
            if any(value is None and encoded[x.id][2][i] in train_refs for i, value in enumerate(constants)):
                raise ValueError('heldout variable reference leaks training identity')
            (held if encoded[x.id][1] == plan else conflicts).append(x.id)
        templates.append(InformingTemplate(model_type._kind + '-template:' + uuid4().hex, question, plan, constants,
                                           train, tuple(held), tuple(conflicts), training_conflicts))
    return model_type(training, validation, templates, complete, unresolved)
