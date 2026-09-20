"""Learn rooted relational queries from supervised scene/mention alignments.

Descriptions are exact structured values, not interpreted words. Scene predicates
and roles are opaque data; queries are induced, never supplied by a teacher.
Unlabeled scene nodes are not negative examples. Negative-only annotations constrain
queries but never provide positive corroboration. validation_example_ids names
positive validation; all negative constraints remain in validation_examples.
A proposal remains a hypothesis.
"""
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from ..agent.scene import SceneGraph
from ..language import Entity, Frame
from ..records import Ref
from .goal_correspondence import _encode
from .experience import _same
from .graph_queries import enumerate_rooted_queries, match_query


@dataclass(frozen=True)
class GroundingExample:
    id: str
    description: object
    scene: SceneGraph
    positive_refs: tuple[Ref, ...]
    negative_refs: tuple[Ref, ...] = ()
    basis: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingMatch:
    reference: Ref
    query_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    matched_proposition_indices: tuple[int, ...]
    assignments: tuple[Ref, ...]


@dataclass(frozen=True)
class GroundingCandidates:
    matches: tuple[GroundingMatch, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class LearnedQuery:
    id: str
    description: tuple
    query: object
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...]


def _description(value):
    # Keep exact Ref IDs in descriptions, unlike scene-local query variables.
    refs = []
    structure = _encode(value, refs)
    return structure, tuple(ref.id for ref in refs)


def _check(example):
    if (type(example) is not GroundingExample or type(example.id) is not str or not example.id
            or type(example.scene) is not SceneGraph):
        raise ValueError('invalid_grounding_example')
    example.scene.validate()
    for values in (example.positive_refs, example.negative_refs):
        if type(values) is not tuple or any(type(ref) is not Ref or ref not in (example.scene.image, *example.scene.nodes) for ref in values):
            raise ValueError('alignment_requires_declared_scene_nodes')
        if len(set(values)) != len(values): raise ValueError('duplicate_alignment')
    if not example.positive_refs and not example.negative_refs:
        raise ValueError('at_least_one_alignment_required')
    if set(example.positive_refs) & set(example.negative_refs): raise ValueError('conflicting_alignment')
    if type(example.basis) is not tuple or any(type(x) is not str for x in example.basis):
        raise ValueError('invalid_example_basis')
    return _description(example.description)


def _supported(example, query, max_matches):
    result = match_query(query, example.scene, max_matches=max_matches, max_states=max_matches)
    targets = {m.bindings[0] for m in result.matches}
    supported = (set(example.positive_refs) <= targets and not set(example.negative_refs) & targets)
    return supported and result.complete, result


class SceneGroundingModel:
    def __init__(self, training, validation, queries, complete, unresolved, max_matches):
        self._id = 'scene-grounding:' + uuid4().hex
        self._training = deepcopy(training)
        self._validation = deepcopy(validation)
        self._queries = tuple(queries)
        self._complete = complete
        self._unresolved = tuple(unresolved)
        self._max_matches = max_matches

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
    def queries(self): return deepcopy(self._queries)
    @property
    def max_matches(self): return self._max_matches

    def propose(self, description, scene):
        try:
            key = _description(description)
            if type(scene) is not SceneGraph: raise ValueError('expected_scene_graph')
            scene.validate()
        except (ValueError, TypeError, RecursionError) as error:
            return GroundingCandidates((), False, (str(error),))
        rows, unresolved = [], list(self.unresolved)
        complete = self.complete
        known = False
        for learned in self._queries:
            if learned.description != key: continue
            known = True
            if not learned.validation_example_ids or learned.conflicting_validation_example_ids:
                unresolved.append('unvalidated_query:' + learned.id)
                continue
            result = match_query(learned.query, scene, max_matches=self.max_matches, max_states=self.max_matches)
            if not result.complete:
                complete = False
                unresolved.extend(learned.id + ':' + reason for reason in result.unresolved)
            if result.complete and not result.matches:
                unresolved.append('query_predicts_no_referent:' + learned.id)
            for match in result.matches:
                rows.append(GroundingMatch(match.bindings[0], (learned.id,), learned.training_example_ids,
                                          learned.validation_example_ids, match.fact_indices, match.bindings))
        if not known: unresolved.append('unknown_description_or_no_supported_query')
        elif not rows: unresolved.append('no_validated_scene_match')
        return GroundingCandidates(tuple(rows), complete, tuple(unresolved))


def fit_scene_grounding(training, validation, *, max_atoms=3, max_patterns=512, max_matches=4096):
    if any(type(n) is not int or n < 1 for n in (max_atoms, max_patterns, max_matches)):
        raise ValueError('search bounds must be positive integers')
    training, validation = deepcopy(tuple(training)), deepcopy(tuple(validation))
    examples = (*training, *validation)
    checked = [(_check(example), example) for example in examples]
    keys = {example.id: key for key, example in checked}
    if len(keys) != len(examples): raise ValueError('example IDs must be unique and split-disjoint')
    scene_snapshots = {}
    for example in examples:
        previous = scene_snapshots.get(example.scene.image)
        if previous is not None and not _same(previous, example.scene):
            raise ValueError('same scene identity has inconsistent retained graph content')
        scene_snapshots[example.scene.image] = example.scene
    train_entities = {r for x in training for r in (x.scene.image, *x.scene.nodes)}
    validation_entities = {r for x in validation for r in (x.scene.image, *x.scene.nodes)}
    if train_entities & validation_entities:
        raise ValueError('validation scene and entity IDs must be disjoint from training')
    for index, first in enumerate(training):
        for second in training[index + 1:]:
            if first.scene.image != second.scene.image and set(first.scene.nodes) & set(second.scene.nodes):
                raise ValueError('independent training scenes require distinct entity IDs')
    patterns = {}
    unresolved = []
    complete = True
    remaining = max_patterns
    for example in training:
        for target in example.positive_refs:
            if remaining == 0:
                complete = False
                unresolved.append('pattern_budget_exhausted')
                break
            search = enumerate_rooted_queries(example.scene, target, max_atoms=max_atoms,
                max_patterns=remaining, max_states=max_matches)
            remaining -= len(search.patterns)
            if not search.complete:
                complete = False
                unresolved.extend(example.id + ':' + reason for reason in search.unresolved)
            for query in search.patterns: patterns[(keys[example.id], query)] = True
        if remaining == 0:
            # Continuing examples would require further enumeration, even if
            # existing patterns happen already to explain them.
            if example is not training[-1]:
                complete = False
                unresolved.append('pattern_budget_exhausted')
            break
    learned = []
    for (description, query) in patterns:
        train_ids = []
        training_scenes = set()
        training_conflict = False
        for example in training:
            if keys[example.id] != description: continue
            supported, result = _supported(example, query, max_matches)
            if not result.complete:
                complete = False
                unresolved.extend(example.id + ':' + reason for reason in result.unresolved)
            if supported:
                train_ids.append(example.id)
                if example.positive_refs:
                    training_scenes.add(example.scene.image)
            else:
                training_conflict = True
        if training_conflict or len(training_scenes) < 2: continue
        valid_ids, conflicts = [], []
        for example in validation:
            if keys[example.id] != description: continue
            supported, result = _supported(example, query, max_matches)
            if not result.complete:
                complete = False
                unresolved.extend(example.id + ':' + reason for reason in result.unresolved)
            if not supported:
                conflicts.append(example.id)
            elif example.positive_refs:
                valid_ids.append(example.id)
        learned.append(LearnedQuery('query:' + uuid4().hex, description, query, tuple(train_ids),
                                    tuple(valid_ids), tuple(conflicts)))
    return SceneGroundingModel(training, validation, learned, complete, tuple(dict.fromkeys(unresolved)), max_matches)
