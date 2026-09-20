"""Whole-context goal revision from supplied structural supervision.

The internal frame is a structural envelope, not a language interpretation. All
previous conditions/invariants and every ordered correction frame participate
in exact matching. Only Ref identity can generalize. Goal labels and explanatory
basis are metadata and do not participate in learned conditions. No constraint
is copied into a proposal unless its output correspondence was taught and
validated; this component does not establish language correction capability.
"""
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from ..goals import GoalSpec
from ..language import Frame
from .goal_correspondence import (
    CorrespondenceCandidates, GoalExample, fit_correspondences,
)


@dataclass(frozen=True)
class GoalRevisionExample:
    id: str
    previous: GoalSpec
    corrections: tuple[Frame, ...]
    revised: GoalSpec
    basis: tuple[str, ...] = ()


def _context(previous, corrections):
    if type(previous) is not GoalSpec:
        raise ValueError('expected_previous_goal_spec')
    if (type(corrections) is not tuple or not corrections
            or any(type(frame) is not Frame for frame in corrections)):
        raise ValueError('expected_nonempty_correction_frame_tuple')
    return Frame('goal-revision-context', {
        'previous_conditions': previous.conditions,
        'previous_invariants': previous.invariants,
        'corrections': corrections,
    })


def _teaching(example):
    if (type(example) is not GoalRevisionExample
            or type(example.id) is not str or not example.id
            or type(example.revised) is not GoalSpec
            or type(example.basis) is not tuple
            or any(type(item) is not str for item in example.basis)):
        raise ValueError('invalid_goal_revision_example')
    return GoalExample(example.id, _context(example.previous, example.corrections),
                       example.revised, example.basis)


class GoalRevisionModel:
    """Detached teaching snapshots and delegated structural correspondences.

    Admission code must authenticate the underlying correspondence state as well
    as the public teaching snapshots. Public properties return detached copies;
    neither constructor arguments nor returned mutable mappings can retrain it.
    """

    def __init__(self, training, validation, correspondence):
        self._id = 'goal-revision:' + uuid4().hex
        self._training = deepcopy(tuple(training))
        self._validation = deepcopy(tuple(validation))
        self._correspondence = deepcopy(correspondence)

    @property
    def id(self): return self._id
    @property
    def complete(self): return self._correspondence.complete
    @property
    def unresolved(self): return self._correspondence.unresolved
    @property
    def training_examples(self): return deepcopy(self._training)
    @property
    def validation_examples(self): return deepcopy(self._validation)
    @property
    def templates(self): return self._correspondence.templates

    def propose(self, previous: GoalSpec, corrections: tuple[Frame, ...]) -> CorrespondenceCandidates:
        try:
            context = _context(previous, corrections)
        except (ValueError, RecursionError) as error:
            return CorrespondenceCandidates((), False, (str(error),))
        return self._correspondence.propose(context)


def fit_goal_revisions(training, validation, *, max_pairs=256) -> GoalRevisionModel:
    """Learn whole contextual correspondences; never infer a task association."""
    training, validation = deepcopy(tuple(training)), deepcopy(tuple(validation))
    correspondence = fit_correspondences(
        tuple(_teaching(example) for example in training),
        tuple(_teaching(example) for example in validation),
        max_pairs=max_pairs,
    )
    return GoalRevisionModel(training, validation, correspondence)
