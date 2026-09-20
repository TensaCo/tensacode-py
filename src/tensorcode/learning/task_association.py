"""Learn pairwise task relations from supplied complete-context examples.

The labels ``revise`` and ``exclude`` are teaching outcomes, never executable
goals or defaults.  Inputs retain every ordered incoming and originating frame
plus all current goal conditions and invariants.  Identifiers, timestamps, goal
labels, and explanatory basis are excluded from the learned semantic context.
"""
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from ..goals import GoalSpec
from ..language import Frame
from .structural_correspondence import (
    StructuralObservation,
    encode,
    fit_templates,
    matches,
)


@dataclass(frozen=True)
class TaskAssociationExample:
    id: str
    episode_id: str
    incoming: tuple[Frame, ...]
    originating: tuple[Frame, ...]
    previous: GoalSpec
    relation: str
    basis: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskAssociationProposal:
    relation: str
    template_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...] = ()
    conflicting_training_example_ids: tuple[str, ...] = ()
    training_episode_ids: tuple[str, ...] = ()
    validation_episode_ids: tuple[str, ...] = ()
    conflicting_validation_episode_ids: tuple[str, ...] = ()
    conflicting_training_episode_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskAssociationCandidates:
    proposals: tuple[TaskAssociationProposal, ...]
    complete: bool
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskAssociationTemplate:
    id: str
    relation: str
    constant_refs: tuple
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...]
    conflicting_training_example_ids: tuple[str, ...]
    training_episode_ids: tuple[str, ...]
    validation_episode_ids: tuple[str, ...]
    conflicting_validation_episode_ids: tuple[str, ...]
    conflicting_training_episode_ids: tuple[str, ...]


def _context(incoming, originating, previous):
    if (type(incoming) is not tuple or not incoming
            or any(type(frame) is not Frame for frame in incoming)):
        raise ValueError('expected_nonempty_incoming_frame_tuple')
    if (type(originating) is not tuple or not originating
            or any(type(frame) is not Frame for frame in originating)):
        raise ValueError('expected_nonempty_originating_frame_tuple')
    if type(previous) is not GoalSpec:
        raise ValueError('expected_previous_goal_spec')
    refs = []
    structure = encode((incoming, originating, previous.conditions,
                        previous.invariants), refs)
    return structure, tuple(refs)


def _observation(example):
    if (type(example) is not TaskAssociationExample
            or any(type(value) is not str or not value
                   for value in (example.id, example.episode_id))
            or example.relation not in ('revise', 'exclude')
            or type(example.basis) is not tuple
            or any(type(item) is not str for item in example.basis)):
        raise ValueError('invalid_task_association_example')
    structure, refs = _context(example.incoming, example.originating,
                               example.previous)
    return StructuralObservation(example.id, example.episode_id, structure,
                                 ('relation', example.relation), refs)


def _public_template(template):
    return TaskAssociationTemplate(
        template.id, template.result[1], template.constant_refs,
        template.training_ids, template.validation_ids, template.conflicting_ids,
        template.conflicting_training_example_ids, template.training_episode_ids,
        template.validation_episode_ids, template.conflicting_validation_episode_ids,
        template.conflicting_training_episode_ids,
    )


class TaskAssociationModel:
    def __init__(self, training, validation, templates, complete, unresolved):
        self._id = 'task-association:' + uuid4().hex
        self._training = deepcopy(tuple(training))
        self._validation = deepcopy(tuple(validation))
        self._templates = deepcopy(tuple(templates))
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
    def templates(self): return deepcopy(tuple(_public_template(row) for row in self._templates))

    def propose(self, incoming, originating, previous):
        try:
            structure, refs = _context(incoming, originating, previous)
        except (ValueError, TypeError, RecursionError) as error:
            return TaskAssociationCandidates((), False, (str(error),))
        proposals = []
        unresolved = list(self.unresolved)
        applicable = tuple(template for template in self._templates
                           if matches(template, structure, refs))
        represented = {template.result for template in applicable
                       if template.validation_ids}
        training = {example.id: example for example in self._training}
        for template in applicable:
            if not template.validation_ids:
                unresolved.append('unvalidated_task_association:' + template.id)
                continue
            proposals.append(TaskAssociationProposal(
                template.result[1], (template.id,), template.training_ids,
                template.validation_ids, template.conflicting_ids,
                template.conflicting_training_example_ids,
                template.training_episode_ids, template.validation_episode_ids,
                template.conflicting_validation_episode_ids,
                template.conflicting_training_episode_ids,
            ))
            for identity in template.conflicting_training_example_ids:
                if ('relation', training[identity].relation) not in represented:
                    unresolved.append('unrepresented_training_rival:' + identity)
        if not proposals:
            unresolved.append('no_validated_task_association')
        return TaskAssociationCandidates(tuple(proposals), self.complete,
                                         tuple(dict.fromkeys(unresolved)))


def fit_task_associations(training, validation, *, max_pairs=256):
    """Fit pair relations while keeping conversational episodes independent."""
    training, validation = deepcopy(tuple(training)), deepcopy(tuple(validation))
    examples = (*training, *validation)
    if any(type(row) is not TaskAssociationExample for row in examples):
        raise ValueError('invalid_task_association_example')
    identities = [row.id for row in examples]
    if len(identities) != len(set(identities)):
        raise ValueError('example IDs must be unique and split-disjoint')
    observations = tuple(_observation(row) for row in training)
    held = tuple(_observation(row) for row in validation)
    templates, complete, unresolved = fit_templates(
        observations, held, max_pairs=max_pairs,
        template_prefix='task-association-template')
    return TaskAssociationModel(training, validation, templates, complete, unresolved)
