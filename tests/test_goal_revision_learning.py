"""Authored structural supervision, not a natural-language correction capability."""
from dataclasses import replace
import pytest
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame
from tensorcode.records import Ref
from tensorcode.learning.goal_revision import GoalRevisionExample, fit_goal_revisions


def example(name, *, keep=True):
    item, old, new, protected = (Ref(name + ':' + role) for role in ('item', 'old', 'new', 'protected'))
    previous = GoalSpec((Condition('located', {'item': item, 'destination': old}),),
        invariants=(Condition('unchanged', {'item': protected}),), label=name,
        basis=('previous:' + name,))
    corrections = (Frame('supplied-change', {'destination': new}, {'mood': 'imperative'}),
                   Frame('supplied-preserve', {'item': protected}))
    revised = GoalSpec((Condition('located', {'item': item, 'destination': new}),),
        invariants=previous.invariants if keep else (), label='revised:' + name,
        basis=('teacher:' + name,))
    return GoalRevisionExample(name, previous, corrections, revised, ('supplied:' + name,))


def fitted():
    return fit_goal_revisions((example('a'), example('b')), (example('held'),))


def test_contextual_reference_transfer_preserves_learned_invariant_and_omits_metadata():
    model, fresh = fitted(), example('fresh')
    result = model.propose(fresh.previous, fresh.corrections)
    assert result.complete and not result.unresolved and len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.goal.conditions == (Condition('located', {'item': Ref('fresh:item'), 'destination': Ref('fresh:new')}),)
    assert proposal.goal.invariants == (Condition('unchanged', {'item': Ref('fresh:protected')}),)
    assert proposal.training_example_ids == ('a', 'b')
    assert proposal.validation_example_ids == ('held',)
    assert proposal.goal.label == ''
    assert proposal.goal.basis != fresh.revised.basis
    renamed = replace(fresh.previous, label='anything', basis=('unrelated',))
    assert model.propose(renamed, fresh.corrections).proposals[0].goal == proposal.goal


def test_invariants_are_learned_output_not_automatic_copies():
    model = fit_goal_revisions((example('a', keep=False), example('b', keep=False)),
                               (example('held', keep=False),))
    fresh = example('fresh')
    assert model.propose(fresh.previous, fresh.corrections).proposals[0].goal.invariants == ()


@pytest.mark.parametrize('change', ['missing', 'reordered', 'negated', 'qualifier', 'old-invariant', 'old-condition', 'extra'])
def test_unsupported_whole_context_abstains(change):
    model, fresh = fitted(), example('fresh')
    previous, corrections = fresh.previous, fresh.corrections
    if change == 'missing': corrections = corrections[:1]
    if change == 'reordered': corrections = corrections[::-1]
    if change == 'negated': corrections = (corrections[0], corrections[1].added(polarity='negative'))
    if change == 'qualifier': corrections = (corrections[0].added(quantity=2), corrections[1])
    if change == 'old-invariant': previous = replace(previous, invariants=())
    if change == 'old-condition': previous = replace(previous, conditions=(replace(previous.conditions[0], negated=True),))
    if change == 'extra': corrections = (*corrections, Frame('another-act'))
    result = model.propose(previous, corrections)
    assert not result.proposals and result.unresolved


def test_conflicting_teaching_retains_alternatives_and_unvalidated_rivals():
    train = (example('a'), example('b'), example('c', keep=False), example('d', keep=False))
    fresh = example('fresh')
    result = fit_goal_revisions(train, (example('held'), example('rival-held', keep=False))).propose(fresh.previous, fresh.corrections)
    assert len(result.proposals) == 2
    assert {len(p.goal.invariants) for p in result.proposals} == {0, 1}
    assert all(p.conflicting_training_example_ids and p.conflicting_validation_example_ids for p in result.proposals)
    unsupported = fit_goal_revisions(train, (example('held'),)).propose(fresh.previous, fresh.corrections)
    assert len(unsupported.proposals) == 1 and unsupported.unresolved


def test_pair_exhaustion_remains_explicit_on_proposals():
    model = fit_goal_revisions((example('a'), example('b'), example('c')), (example('held'),), max_pairs=1)
    fresh = example('fresh')
    result = model.propose(fresh.previous, fresh.corrections)
    assert not model.complete and not result.complete
    assert 'pair_budget_exhausted' in result.unresolved


def test_training_and_exposed_snapshots_are_detached_from_mutable_callers():
    train, held, fresh = [example('a'), example('b')], [example('held')], example('fresh')
    model = fit_goal_revisions(train, held)
    expected = model.propose(fresh.previous, fresh.corrections)
    train[0].previous.conditions[0].args.clear()
    train[1].corrections[0].roles.clear()
    held[0].revised.invariants[0].args.clear()
    model.training_examples[0].corrections[0].roles.clear()
    model.validation_examples[0].previous.conditions[0].args.clear()
    assert model.training_examples[0].previous.conditions[0].args
    assert model.training_examples[1].corrections[0].roles
    assert model.validation_examples[0].revised.invariants[0].args
    assert model.propose(fresh.previous, fresh.corrections) == expected
    expected.proposals[0].goal.conditions[0].args.clear()
    assert model.propose(fresh.previous, fresh.corrections).proposals[0].goal.conditions[0].args


@pytest.mark.parametrize('corrections', [(), [], (object(),)])
def test_invalid_correction_context_never_proposes(corrections):
    result = fitted().propose(example('fresh').previous, corrections)
    assert not result.complete and not result.proposals and result.unresolved


def test_no_validation_no_invented_references_and_no_split_leakage():
    a, b, fresh = example('a'), example('b'), example('fresh')
    assert not fit_goal_revisions((a, b), ()).propose(fresh.previous, fresh.corrections).proposals
    alien = GoalSpec((Condition('located', {'item': Ref('entity:absent')}),))
    model = fit_goal_revisions((replace(a, revised=alien), replace(b, revised=alien)), (example('held'),))
    assert not model.complete and not model.propose(fresh.previous, fresh.corrections).proposals
    with pytest.raises(ValueError, match='IDs'):
        fit_goal_revisions((a, b), (a,))
    with pytest.raises(ValueError, match='entities'):
        fit_goal_revisions((a, b), (replace(a, id='leak'),))
