"""Supplied lexical resources generate alternatives, never an implicit winner."""
from dataclasses import replace

import pytest

from tensorcode.language import Frame
from tensorcode.language import verbnet as vn
from tensorcode.outcomes import Unknown


def resource(class_id, *, predicate='exists', pp=(), extra=(), negated=False):
    syntax = (('NP', 'Agent'), ('VERB', ''), ('NP', 'Theme'))
    syntax += tuple(item for role in pp for item in (('PREP', ''), ('NP', role))) + extra
    semantics = (vn.Pred('do', (('Event', 'e1'),)),
                 vn.Pred(predicate, (('Event', 'e2'), ('ThemRole', 'Theme')) +
                         tuple(('ThemRole', role) for role in pp), negated))
    return vn.VerbClass(class_id, ('change',), (vn.VFrame('authored construction', syntax, semantics),))


def test_all_classes_survive_without_priors_or_file_order_authorizing_goal():
    assert not hasattr(vn, 'sense_counts')
    assert not hasattr(vn, 'class_prior')
    first, second = resource('a', predicate='exists'), resource('b', predicate='destroyed')
    frame = Frame('change', {'object': 'item'})
    result = vn.goal_candidates(frame, {'change': (first, second)})
    reversed_result = vn.goal_candidates(frame, {'change': (second, first)})
    assert result.complete and len(result.proposals) == 2
    assert {p.goal.conditions[0].pred for p in result.proposals} == {p.goal.conditions[0].pred for p in reversed_result.proposals}
    assert vn.goal_of(frame, {'change': (first, second)}).reason == 'ambiguous_goal'


def test_compatible_pp_bindings_are_enumerated_instead_of_first_match():
    vc = resource('ambiguous', pp=('Destination', 'Goal'))
    frame = Frame('change', {'object': 'item', 'destination': 'there'})
    result = vn.goal_candidates(frame, {'change': (vc,)})
    assert result.complete and len(result.proposals) == 2
    mappings = {proposal.derivations[0].bindings for proposal in result.proposals}
    assert (('destination', 'Destination'), ('object', 'Theme')) in mappings
    assert (('destination', 'Goal'), ('object', 'Theme')) in mappings
    assert all('construction_slots_unresolved' in p.derivations[0].obligations for p in result.proposals)


def test_competing_pp_inputs_keep_both_maximal_partial_assignments():
    vc = resource('conflict', pp=('Goal',))
    frame = Frame('change', {'object': 'item', 'destination': 'a', 'recipient': 'b'})
    result = vn.goal_candidates(frame, {'change': (vc,)})
    assert result.complete and len(result.proposals) == 2
    assert {p.goal.unmapped_roles for p in result.proposals} == {('recipient',), ('destination',)}
    for proposal in result.proposals:
        assigned = [target for _, target in proposal.derivations[0].bindings]
        assert len(assigned) == len(set(assigned))


def test_exact_duplicates_group_all_derivations_without_role_synonym_equivalence():
    first = resource('first')
    second = resource('second')
    frame = Frame('change', {'object': {'nested': ['item']}})
    result = vn.goal_candidates(frame, {'change': (first, second)})
    assert result.complete and len(result.proposals) == 1
    assert {d.verb_class for d in result.proposals[0].derivations} == {'first', 'second'}
    assert isinstance(vn.goal_of(frame, {'change': (first, second)}), vn.Goal)
    altered = replace(second, frames=(replace(second.frames[0], semantics=(
        vn.Pred('do', (('Event', 'e1'),)), vn.Pred('exists', (('Event', 'e2'), ('ThemRole', 'Patient'))))),))
    assert len(vn.goal_candidates(frame, {'change': (first, altered)}).proposals) == 2


def test_original_modifiers_and_quantities_survive_candidate_projection():
    frame = Frame('change', {'object': {'quantity': 3}}, {'negated': True, 'scope': ['request']})
    result = vn.goal_candidates(frame, {'change': (resource('fixture'),)})
    assert result.proposals[0].goal.frame == frame
    frame.features['scope'].clear()
    assert result.proposals[0].goal.frame.features['scope'] == ['request']


def test_syntax_mismatch_is_retained_as_obligation_not_silently_filtered():
    vc = resource('extra-object', extra=(('NP', 'Result'),))
    result = vn.goal_candidates(Frame('change', {'object': 'item'}), {'change': (vc,)})
    assert result.complete and len(result.proposals) == 1
    assert result.proposals[0].derivations[0].obligations == ('construction_slots_unresolved',)


def test_budget_exhaustion_is_explicit_and_does_not_create_unique_authority():
    lexicon = {'change': (resource('one'), resource('two', predicate='other'))}
    frame = Frame('change', {'object': 'item'})
    zero = vn.goal_candidates(frame, lexicon, max_derivations=0)
    assert not zero.complete and not zero.proposals and zero.unresolved
    limited = vn.goal_candidates(frame, lexicon, max_derivations=3)
    assert not limited.complete
    assert all(item.reason == 'derivation_budget_exhausted' for item in limited.unresolved)
    # A default-bound search also remains incomplete if the resource exceeds it.
    many = {'change': tuple(resource(str(i), predicate=str(i)) for i in range(300))}
    assert vn.goal_of(frame, many).reason == 'incomplete_goal_search'


@pytest.mark.parametrize('budget', [-1, True, 1.2])
def test_derivation_budget_requires_nonnegative_integer(budget):
    with pytest.raises(ValueError):
        vn.goal_candidates(Frame('change'), {}, max_derivations=budget)


def test_unknown_verb_and_absent_resource_result_are_explicit():
    assert vn.goal_of(Frame('missing'), {}).reason == 'unknown_verb'
    empty = vn.VerbClass('empty', ('change',), ())
    result = vn.goal_candidates(Frame('change'), {'change': (empty,)})
    assert result.complete and result.unresolved[0].reason == 'no_result_state'
    assert not result.proposals


def test_negation_is_not_deduplicated_away():
    frame = Frame('change', {'object': 'item'})
    result = vn.goal_candidates(frame, {'change': (resource('yes'), resource('no', negated=True))})
    assert len(result.proposals) == 2
    assert {proposal.goal.conditions[0].negated for proposal in result.proposals} == {False, True}


def test_unique_goal_with_unresolved_construction_does_not_gain_authority():
    frame = Frame('change', {'object': 'item'})
    lexicon = {'change': (resource('extra-object', extra=(('NP', 'Result'),)),)}
    batch = vn.goal_candidates(frame, lexicon)
    assert batch.complete and len(batch.proposals) == 1
    assert vn.goal_of(frame, lexicon).reason == 'unresolved_goal_projection'


def test_unique_goal_with_unmapped_input_retains_obligation_and_abstains():
    frame = Frame('change', {'object': 'item', 'unrepresented-role': 'important'})
    lexicon = {'change': (resource('fixture'),)}
    batch = vn.goal_candidates(frame, lexicon)
    assert batch.proposals[0].goal.unmapped_roles == ('unrepresented-role',)
    assert vn.goal_of(frame, lexicon).reason == 'unresolved_goal_projection'


def test_merged_class_label_preserves_ambiguity_independent_of_resource_order():
    first, second = resource('roll'), resource('slide')
    frame = Frame('change', {'object': 'item'})
    forward = vn.goal_candidates(frame, {'change': (first, second)}).proposals[0]
    reverse = vn.goal_candidates(frame, {'change': (second, first)}).proposals[0]
    assert forward.goal == reverse.goal
    assert forward.goal.verb_class == 'alternatives:roll|slide'
    assert {derivation.verb_class for derivation in forward.derivations} == {'roll', 'slide'}
    assert vn.goal_candidates(frame, {'change': (first,)}).proposals[0].goal.verb_class == 'roll'
