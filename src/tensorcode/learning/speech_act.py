"""Supervised communicative hypotheses over retained neutral syntax and frames.

The feature projection and bounded pair abstraction are supplied algorithms;
intent labels and question correspondences come exclusively from teaching.
No punctuation, prefix, verb, or missing-subject rule supplies an intent.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import combinations
from uuid import uuid4

from ..language import Entity, Frame, Request, Question
from ..language.deps_semantics import ProvisionalMeaning
from ..outcomes import Unknown
from .goal_correspondence import _encode


@dataclass(frozen=True)
class SpeechActLabel:
    kind: str
    asked: str | None = None
    query_path: tuple[str | int, ...] = ()
    token_indices: tuple[int, ...] = ()

    def __post_init__(self):
        if self.kind not in ('request', 'statement', 'question', 'unresolved'):
            raise ValueError('unsupported speech act label')
        if type(self.query_path) is not tuple or any(type(p) not in (str, int) for p in self.query_path):
            raise ValueError('query correspondence requires a structural path')
        if type(self.token_indices) is not tuple or any(type(i) is not int or i < 0 for i in self.token_indices):
            raise ValueError('query correspondence requires token indices')
        if self.kind == 'question':
            if type(self.asked) is not str or not self.asked or not self.token_indices:
                raise ValueError('question teaching requires queried role and source tokens')
        elif self.asked is not None or self.query_path or self.token_indices:
            raise ValueError('only questions carry query correspondences')


@dataclass(frozen=True)
class SpeechActExample:
    id: str
    source_id: str
    text: str
    meaning: ProvisionalMeaning
    label: SpeechActLabel
    basis: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpeechActProposal:
    label: SpeechActLabel
    meaning: object
    frame: Frame
    template_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...]
    conflicting_training_example_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SpeechActCandidates:
    proposals: tuple[SpeechActProposal, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class LexicalSlot:
    index: int


@dataclass(frozen=True)
class SpeechActTemplate:
    id: str
    syntax: tuple
    pattern: tuple
    label: SpeechActLabel
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...]
    conflicting_training_example_ids: tuple[str, ...] = ()


def _input(meaning):
    if type(meaning) is not ProvisionalMeaning or type(meaning.frame) is not Frame:
        raise ValueError('expected source-anchored ProvisionalMeaning')
    n = len(meaning.words)
    if (any(type(seq) is not tuple for seq in (meaning.words, meaning.tags, meaning.lemmas, meaning.heads, meaning.labels))
            or type(meaning.frame_index) is not int or meaning.frame_index < 0
            or not n or len(meaning.tags) != n or len(meaning.lemmas) != n
            or any(type(v) is not str for seq in (meaning.words, meaning.tags, meaning.lemmas) for v in seq)
            or type(meaning.root) is not int or not 1 <= meaning.root <= n):
        raise ValueError('incomplete source syntax')
    heads, labels = dict(meaning.heads), dict(meaning.labels)
    if (len(heads) != n or len(labels) != n or set(heads) != set(range(1, n + 1))
            or set(labels) != set(heads) or len(meaning.heads) != n or len(meaning.labels) != n
            or any(type(i) is not int for i in (*heads, *labels))
            or any(type(h) is not int or h < 0 or h > n for h in heads.values())
            or any(type(v) is not str for v in labels.values())
            or [i for i, h in heads.items() if h == 0] != [meaning.root]):
        raise ValueError('invalid source dependency tree')
    for start in heads:
        seen, node = set(), start
        while node:
            if node in seen: raise ValueError('cyclic source dependency tree')
            seen.add(node)
            node = heads[node]
    syntax = _encode((meaning.tags, tuple(sorted(heads.items())), tuple(sorted(labels.items())),
                      meaning.root, meaning.frame_index), [])
    payload = _encode((meaning.words, meaning.lemmas, meaning.frame), [])
    return syntax, payload


def _query_frame(label, meaning):
    """Consume only an explicitly taught, source-corresponding query slot."""
    indices = label.token_indices
    if indices != tuple(range(indices[0], indices[-1] + 1)) or indices[-1] >= len(meaning.words):
        raise ValueError('question requires one exact contiguous source token span')
    if not label.query_path:
        if indices != tuple(range(len(meaning.words))):
            raise ValueError('whole-frame question requires whole source correspondence')
        return deepcopy(meaning.frame)
    def selected(value, path):
        if not path: return value
        step, rest = path[0], path[1:]
        if type(value) in (Frame, Entity) and step in ('roles', 'features') and hasattr(value, step):
            return selected(getattr(value, step), rest)
        if type(value) is dict and step in value: return selected(value[step], rest)
        if type(value) is tuple and type(step) is int and 0 <= step < len(value): return selected(value[step], rest)
        raise ValueError('question correspondence is absent from retained frame')
    target = selected(meaning.frame, label.query_path)
    if type(target) is Entity:
        if target.ref is not None or target.candidates or target.features:
            raise ValueError('query span cannot erase grounded or qualified entity semantics')
        surface = target.text
    elif type(target) in (str, int, float): surface = str(target)
    elif type(target) is tuple and all(type(v) is str for v in target): surface = ' '.join(target)
    else: raise ValueError('query slot has no supported exact lexical correspondence')
    normalize = lambda value: ' '.join(value.split()).casefold()
    observed = (' '.join(meaning.words[i] for i in indices), ' '.join(meaning.lemmas[i] for i in indices))
    if normalize(surface) not in tuple(normalize(value) for value in observed):
        raise ValueError('query slot does not correspond to the taught source token span')
    def remove(value, path):
        step, rest = path[0], path[1:]
        if type(value) in (Frame, Entity) and step in ('roles', 'features') and hasattr(value, step) and rest:
            return replace(value, **{step: remove(getattr(value, step), rest)})
        if type(value) is dict and step in value:
            return {key: (remove(item, rest) if key == step and rest else deepcopy(item))
                    for key, item in value.items() if rest or key != step}
        if type(value) is tuple and type(step) is int and 0 <= step < len(value):
            return tuple(remove(item, rest) if i == step and rest else deepcopy(item)
                         for i, item in enumerate(value) if rest or i != step)
        raise ValueError('unsupported query consumption path')
    return remove(meaning.frame, label.query_path)


def _label_for(label, meaning):
    if type(label) is not SpeechActLabel: raise ValueError('explicit SpeechActLabel required')
    label.__post_init__()
    if label.kind == 'question':
        return _query_frame(label, meaning)
    return deepcopy(meaning.frame)


def _abstract(left, right, pairs, slots):
    if left == right: return left
    if (type(left) is tuple and type(right) is tuple and len(left) == len(right) == 3
            and left[:2] == right[:2] == ('scalar', 'str') and (left[2], right[2]) in pairs):
        pair = (left[2], right[2])
        if pair not in slots: slots.append(pair)
        return LexicalSlot(slots.index(pair))
    if type(left) is tuple and type(right) is tuple and len(left) == len(right):
        return tuple(_abstract(a, b, pairs, slots) for a, b in zip(left, right))
    raise ValueError('nonlexical structure differs')


def _match(pattern, value, bindings=None):
    bindings = {} if bindings is None else bindings
    if type(pattern) is LexicalSlot:
        if type(value) is not tuple or len(value) != 3 or value[:2] != ('scalar', 'str'): return False
        if pattern.index in bindings: return bindings[pattern.index] == value
        bindings[pattern.index] = value
        return True
    if type(pattern) is tuple:
        return type(value) is tuple and len(pattern) == len(value) and all(_match(a, b, bindings) for a, b in zip(pattern, value))
    return type(pattern) is type(value) and pattern == value


class SpeechActModel:
    def __init__(self, training, validation, templates, complete, unresolved):
        self._id = 'speech-act:' + uuid4().hex
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

    def propose(self, meaning):
        try: syntax, payload = _input(meaning)
        except (ValueError, TypeError, RecursionError) as error:
            return SpeechActCandidates((), False, (str(error),))
        proposals, unresolved = [], list(self.unresolved)
        applicable = tuple(t for t in self._templates
                           if syntax == t.syntax and _match(t.pattern, payload))
        represented = tuple(t.label for t in applicable if t.validation_example_ids)
        training = {x.id: x for x in self._training}
        for template in self._templates:
            if syntax != template.syntax or not _match(template.pattern, payload): continue
            if not template.validation_example_ids:
                unresolved.append('unvalidated_speech_act:' + template.id)
                continue
            frame = deepcopy(meaning.frame)
            label = template.label
            try: realized = _label_for(label, meaning)
            except ValueError as error:
                unresolved.append(str(error))
                continue
            interpreted = (Request(frame) if label.kind == 'request' else
                           Question(realized, label.asked) if label.kind == 'question' else
                           frame if label.kind == 'statement' else Unknown('learned_unresolved_speech_act'))
            proposals.append(SpeechActProposal(label, interpreted, frame, (template.id,),
                template.training_example_ids, template.validation_example_ids,
                template.conflicting_validation_example_ids, template.conflicting_training_example_ids))
            for identity in template.conflicting_training_example_ids:
                if training[identity].label not in represented:
                    unresolved.append('unrepresented_training_rival:' + identity)
        if not proposals: unresolved.append('no_validated_speech_act')
        return SpeechActCandidates(tuple(proposals), self.complete, tuple(dict.fromkeys(unresolved)))


def fit_speech_acts(training, validation, *, max_pairs=256):
    if type(max_pairs) is not int or max_pairs < 1: raise ValueError('max_pairs must be positive')
    training, validation = deepcopy(tuple(training)), deepcopy(tuple(validation))
    examples = (*training, *validation)
    for example in examples:
        if (type(example) is not SpeechActExample or any(type(v) is not str or not v for v in (example.id, example.source_id, example.text))
                or type(example.basis) is not tuple or any(type(v) is not str for v in example.basis)):
            raise ValueError('invalid speech act teaching example')
    for field in ('id', 'source_id'):
        if len({getattr(x, field) for x in examples}) != len(examples):
            raise ValueError('teaching example and source IDs must be independent')
    normalized = [' '.join(x.text.split()).casefold() for x in examples]
    if len(set(normalized)) != len(normalized): raise ValueError('teaching texts must be independent')
    encoded = {}
    for example in examples:
        encoded[example.id] = _input(example.meaning)
        _label_for(example.label, example.meaning)
        if ''.join(example.text.split()) != ''.join(''.join(example.meaning.words).split()):
            raise ValueError('teaching text does not cover retained source tokens')
    patterns, complete, unresolved = {}, True, []
    for index, (left, right) in enumerate(combinations(training, 2)):
        if index >= max_pairs:
            complete = False
            unresolved.append('pair_budget_exhausted')
            break
        ls, lp = encoded[left.id]
        rs, rp = encoded[right.id]
        if ls != rs or left.label != right.label: continue
        pairs = {(a, b) for seq1, seq2 in ((left.meaning.words, right.meaning.words), (left.meaning.lemmas, right.meaning.lemmas))
                 for a, b in zip(seq1, seq2) if a != b}
        slots = []
        try: pattern = _abstract(lp, rp, pairs, slots)
        except ValueError: continue
        if slots: patterns[(ls, pattern, left.label)] = True
    templates = []
    for syntax, pattern, label in patterns:
        def matches(example):
            shape, payload = encoded[example.id]
            return shape == syntax and _match(pattern, payload)
        train = tuple(x.id for x in training if matches(x) and x.label == label)
        held = tuple(x.id for x in validation if matches(x) and x.label == label)
        conflicts = tuple(x.id for x in validation if matches(x) and x.label != label)
        training_conflicts = tuple(x.id for x in training if matches(x) and x.label != label)
        templates.append(SpeechActTemplate('speech-template:' + uuid4().hex, syntax, pattern, label,
                                           train, held, conflicts, training_conflicts))
    return SpeechActModel(training, validation, templates, complete, unresolved)
