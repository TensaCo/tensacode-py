"""Exact typed structure and reference-correspondence template fitting.

This module does not assign semantic labels.  Adapters supply an exact input
structure and an exact output value from explicit teaching.  Fitting may only
abstract ``Ref`` identities; scalar types, collection order, dataclass fields,
reference equality, and constants remain part of the learned structure.
"""
from dataclasses import dataclass, fields
from itertools import combinations
import math
from uuid import uuid4

from ..goals import Condition
from ..language import Entity, Frame
from ..records import Ref


_TYPES = {cls.__name__: cls for cls in (Frame, Entity, Condition)}


def encode(value, refs, *, allow_new=True):
    """Encode supported values while assigning first-occurrence Ref slots."""
    if type(value) is Ref:
        if value not in refs:
            if not allow_new:
                raise ValueError('output_reference_absent_from_input')
            refs.append(value)
        return ('ref', refs.index(value))
    if value is None or type(value) in (bool, int, str, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError('nonfinite_value')
        return ('scalar', type(value).__name__, value)
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(encode(item, refs, allow_new=allow_new)
                                            for item in value))
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError('unsupported_mapping_key')
        return ('dict', tuple((key, encode(value[key], refs, allow_new=allow_new))
                              for key in sorted(value)))
    if type(value) in _TYPES.values():
        return (type(value).__name__, tuple(
            (field.name, encode(getattr(value, field.name), refs, allow_new=allow_new))
            for field in fields(value)))
    raise ValueError('unsupported_node_type:' + type(value).__name__)


def decode(node, refs):
    """Reconstruct a value encoded by :func:`encode` with supplied Ref slots."""
    kind = node[0]
    if kind == 'ref':
        return refs[node[1]]
    if kind == 'scalar':
        return node[2]
    if kind == 'tuple':
        return tuple(decode(item, refs) for item in node[1])
    if kind == 'list':
        return [decode(item, refs) for item in node[1]]
    values = {key: decode(value, refs) for key, value in node[1]}
    if kind == 'dict':
        return values
    return _TYPES[kind](**values)


@dataclass(frozen=True)
class StructuralObservation:
    """One adapter-provided encoded teaching observation."""
    id: str
    episode_id: str
    structure: tuple
    result: tuple
    refs: tuple[Ref, ...]


@dataclass(frozen=True)
class StructuralTemplate:
    """A fitted exact structure with only demonstrated Ref slots abstracted."""
    id: str
    structure: tuple
    result: tuple
    constant_refs: tuple[Ref | None, ...]
    training_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    conflicting_ids: tuple[str, ...]
    conflicting_training_example_ids: tuple[str, ...] = ()
    training_episode_ids: tuple[str, ...] = ()
    validation_episode_ids: tuple[str, ...] = ()
    conflicting_validation_episode_ids: tuple[str, ...] = ()
    conflicting_training_episode_ids: tuple[str, ...] = ()


def matches(template, structure, refs):
    """Match exact structure, constants, and its complete Ref equality pattern."""
    return (template.structure == structure
            and len(template.constant_refs) == len(refs)
            and all(constant is None or refs[index] == constant
                    for index, constant in enumerate(template.constant_refs)))


def _unique(values):
    return tuple(dict.fromkeys(values))


def fit_templates(training, validation, *, max_pairs=256, template_prefix='template',
                  unresolved=(), require_reference_variation=True):
    """Fit templates from pairs belonging to independent teaching episodes.

    Adapters that learn substitution keep ``require_reference_variation=True``.
    An adapter for a positively taught exact classification may explicitly opt
    out, allowing two independent reference-free episodes to support a template.
    """
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError('max_pairs must be a positive integer')
    if type(require_reference_variation) is not bool:
        raise ValueError('require_reference_variation must be boolean')
    training, validation = tuple(training), tuple(validation)
    observations = (*training, *validation)
    if any(type(row) is not StructuralObservation for row in observations):
        raise ValueError('expected structural observations')
    identities = [row.id for row in observations]
    if len(identities) != len(set(identities)):
        raise ValueError('example IDs must be unique and split-disjoint')
    if {row.episode_id for row in training} & {row.episode_id for row in validation}:
        raise ValueError('training and validation episode IDs must be disjoint')

    train_refs = {ref for row in training for ref in row.refs}
    validation_refs = {ref for row in validation for ref in row.refs}
    ref_sets = [set(row.refs) for row in observations]
    common_refs = set.intersection(*ref_sets) if ref_sets else set()
    if (train_refs & validation_refs) - common_refs:
        raise ValueError('validation variable entities must be disjoint from training')

    patterns = {}
    examined = 0
    reasons = list(unresolved)
    complete = not reasons
    for left, right in combinations(training, 2):
        if examined >= max_pairs:
            complete = False
            reasons.append('pair_budget_exhausted')
            break
        examined += 1
        if left.episode_id == right.episode_id:
            continue
        if (left.structure != right.structure or left.result != right.result
                or len(left.refs) != len(right.refs)):
            continue
        if require_reference_variation and (not left.refs or left.refs == right.refs):
            continue
        constants = tuple(first if first == second else None
                          for first, second in zip(left.refs, right.refs))
        patterns[(left.structure, left.result, constants)] = True

    templates = []
    for structure, result, constants in patterns:
        def applicable(row):
            return (row.structure == structure and len(row.refs) == len(constants)
                    and all(constant is None or row.refs[index] == constant
                            for index, constant in enumerate(constants)))

        matching_training = tuple(row for row in training if applicable(row))
        matching_validation = tuple(row for row in validation if applicable(row))
        for row in matching_validation:
            if any(constant is None and row.refs[index] in train_refs
                   for index, constant in enumerate(constants)):
                raise ValueError('validation variable entities must be disjoint from training')
        supported_training = tuple(row for row in matching_training if row.result == result)
        supported_validation = tuple(row for row in matching_validation if row.result == result)
        conflicting_validation = tuple(row for row in matching_validation if row.result != result)
        conflicting_training = tuple(row for row in matching_training if row.result != result)
        templates.append(StructuralTemplate(
            template_prefix + ':' + uuid4().hex, structure, result, constants,
            tuple(row.id for row in supported_training),
            tuple(row.id for row in supported_validation),
            tuple(row.id for row in conflicting_validation),
            tuple(row.id for row in conflicting_training),
            _unique(row.episode_id for row in supported_training),
            _unique(row.episode_id for row in supported_validation),
            _unique(row.episode_id for row in conflicting_validation),
            _unique(row.episode_id for row in conflicting_training),
        ))
    return tuple(templates), complete, tuple(dict.fromkeys(reasons))
