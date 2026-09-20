"""Supplied pair labels teach task association without lexical or recency rules."""
from dataclasses import replace

import pytest

from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame
from tensorcode.records import Ref
from tensorcode.learning.task_association import (
    TaskAssociationExample,
    fit_task_associations,
)
from tensorcode.learning.structural_correspondence import (
    StructuralObservation,
    fit_templates,
)


def example(episode, candidate, relation, *, matching, incoming=None):
    resource = Ref(f'resource:{episode}:{candidate}')
    destination = Ref(f'destination:{episode}:{candidate}')
    replacement = Ref(f'replacement:{episode}:{candidate}')
    mentioned = resource if matching else Ref(f'rival:{episode}:{candidate}')
    previous = GoalSpec(
        (Condition('located', {'resource': resource, 'destination': destination}),),
        label=f'task:{episode}:{candidate}',
        invariants=(Condition('protected', {'resource': resource}),),
        basis=(f'task revision {episode}:{candidate}',),
    )
    clauses = incoming or (
        Frame('change', {'destination': replacement}),
        Frame('preserve', {'resource': mentioned}, {'qualifier': 'existing'}),
    )
    originating = (
        Frame('place', {'resource': resource, 'destination': destination}),
        Frame('protect', {'resource': resource}),
    )
    return TaskAssociationExample(
        f'{episode}:{candidate}:{relation}', episode, clauses, originating,
        previous, relation, ('explicit pair label',),
    )


def dataset():
    training = tuple(
        example(episode, relation, relation, matching=relation == 'revise')
        for episode in ('train-a', 'train-b')
        for relation in ('revise', 'exclude')
    )
    validation = tuple(
        example('held', relation, relation, matching=relation == 'revise')
        for relation in ('revise', 'exclude')
    )
    return training, validation


def propose(model, *, matching=True, clauses=None):
    fresh = example('fresh', 'candidate', 'revise' if matching else 'exclude',
                    matching=matching, incoming=clauses)
    return model.propose(fresh.incoming, fresh.originating, fresh.previous)


def test_independent_episodes_transfer_reference_equality_to_supplied_relations():
    training, validation = dataset()
    model = fit_task_associations(training, validation)

    revise = propose(model, matching=True)
    exclude = propose(model, matching=False)

    assert revise.complete and [proposal.relation for proposal in revise.proposals] == ['revise']
    assert exclude.complete and [proposal.relation for proposal in exclude.proposals] == ['exclude']
    assert revise.proposals[0].training_example_ids == (
        'train-a:revise:revise', 'train-b:revise:revise')
    assert revise.proposals[0].training_episode_ids == ('train-a', 'train-b')
    assert revise.proposals[0].validation_example_ids == ('held:revise:revise',)
    assert revise.proposals[0].validation_episode_ids == ('held',)


def test_missing_negated_or_reordered_incoming_clauses_do_not_match_silently():
    model = fit_task_associations(*dataset())
    fresh = example('fresh', 'candidate', 'revise', matching=True)
    changed = (
        fresh.incoming[0],
        Frame('preserve', fresh.incoming[1].roles,
              {'qualifier': 'existing', 'polarity': 'negative'}),
    )

    assert not model.propose(fresh.incoming[:1], fresh.originating, fresh.previous).proposals
    assert not model.propose(changed, fresh.originating, fresh.previous).proposals
    assert not model.propose(fresh.incoming[::-1], fresh.originating, fresh.previous).proposals


def test_conflicting_teaching_preserves_both_validated_relations_and_conflict_evidence():
    training = tuple(
        example(episode, relation, relation, matching=True)
        for episode in ('train-a', 'train-b')
        for relation in ('revise', 'exclude')
    )
    validation = (
        example('held', 'revise', 'revise', matching=True),
        example('held', 'exclude', 'exclude', matching=True),
    )

    result = propose(fit_task_associations(training, validation), matching=True)

    assert {proposal.relation for proposal in result.proposals} == {'revise', 'exclude'}
    assert all(proposal.conflicting_validation_example_ids for proposal in result.proposals)
    assert all(proposal.conflicting_training_example_ids for proposal in result.proposals)


def test_unvalidated_rival_remains_visible_beside_supported_relation():
    training = tuple(
        example(episode, relation, relation, matching=True)
        for episode in ('train-a', 'train-b')
        for relation in ('revise', 'exclude')
    )
    validation = (example('held', 'revise', 'revise', matching=True),)
    model = fit_task_associations(training, validation)

    result = propose(model, matching=True)

    assert [proposal.relation for proposal in result.proposals] == ['revise']
    rival = next(template for template in model.templates
                 if template.relation == 'exclude')
    assert rival.conflicting_validation_example_ids == ('held:revise:revise',)
    assert 'unvalidated_task_association:' + rival.id in result.unresolved


def test_examples_from_one_episode_cannot_form_independent_support():
    training = (
        example('shared', 'one', 'revise', matching=True),
        example('shared', 'two', 'revise', matching=True),
    )
    validation = (example('held', 'one', 'revise', matching=True),)

    model = fit_task_associations(training, validation)

    assert not model.templates
    assert not propose(model, matching=True).proposals


def test_episode_splits_are_disjoint_even_when_example_ids_differ():
    training = (
        example('train-a', 'one', 'revise', matching=True),
        example('train-b', 'one', 'revise', matching=True),
    )
    held = replace(example('held', 'one', 'revise', matching=True),
                   episode_id='train-a')

    with pytest.raises(ValueError, match='episode'):
        fit_task_associations(training, (held,))


def test_pair_budget_exhaustion_and_invalid_live_context_are_explicit():
    training = (
        example('shared', 'one', 'revise', matching=True),
        example('shared', 'two', 'revise', matching=True),
        example('independent', 'one', 'revise', matching=True),
    )
    validation = (example('held', 'one', 'revise', matching=True),)
    model = fit_task_associations(training, validation, max_pairs=1)
    fresh = example('fresh', 'candidate', 'revise', matching=True)

    result = model.propose(fresh.incoming, fresh.originating, fresh.previous)

    assert not model.complete and 'pair_budget_exhausted' in result.unresolved
    malformed = model.propose((fresh.incoming[0], object()), fresh.originating, fresh.previous)
    assert not malformed.complete and not malformed.proposals


def test_public_snapshots_are_detached_and_metadata_does_not_enter_context():
    training, validation = dataset()
    model = fit_task_associations(training, validation)
    renamed = tuple(replace(row, id='renamed:' + row.id,
                            episode_id='renamed:' + row.episode_id,
                            previous=replace(row.previous, label='renamed', basis=('renamed',)),
                            basis=('renamed teaching',)) for row in training)
    renamed_validation = tuple(replace(row, id='renamed:' + row.id,
                                       episode_id='renamed:' + row.episode_id,
                                       previous=replace(row.previous, label='renamed', basis=('renamed',)),
                                       basis=('renamed teaching',)) for row in validation)
    same_structure = fit_task_associations(renamed, renamed_validation)

    model.training_examples[0].incoming[0].roles.clear()

    assert model.training_examples[0].incoming[0].roles
    assert [p.relation for p in propose(model).proposals] == [
        p.relation for p in propose(same_structure).proposals]


def test_shared_fitter_requires_explicit_opt_in_for_exact_reference_free_teaching():
    structure = ('scalar', 'str', 'complete positively taught context')
    output = ('relation', 'new')
    training = (
        StructuralObservation('train-a', 'episode-a', structure, output, ()),
        StructuralObservation('train-b', 'episode-b', structure, output, ()),
    )
    validation = (
        StructuralObservation('held', 'episode-held', structure, output, ()),
    )

    default_templates, _, _ = fit_templates(training, validation)
    exact_templates, complete, unresolved = fit_templates(
        training, validation, require_reference_variation=False)

    assert not default_templates
    assert complete and not unresolved and len(exact_templates) == 1
    assert exact_templates[0].training_episode_ids == ('episode-a', 'episode-b')
    assert exact_templates[0].validation_episode_ids == ('episode-held',)
