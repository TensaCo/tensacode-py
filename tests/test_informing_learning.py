"""Explicit teaching learns reference alignment, not lexical answer routing."""
from dataclasses import replace
from datetime import datetime, timezone
import pytest
from tensorcode.language import Entity, Frame, Question
from tensorcode.records import Proposition, Ref, Var, Interval
from tensorcode.learning.informing import InformingPlan, InformingExample, fit_informing


def example(name, *, route='observe', qualifiers=None, asked='quantity', swap=False, kind=None):
    owner, item = Ref('owner:' + name), Ref('kind:' + (kind or name))
    question = Question(Frame('opaque-question', {'owner': owner, 'kind': item}, qualifiers or {}), asked)
    args = (('who', item if swap else owner), ('which', owner if swap else item))
    query = Proposition('opaque-result', {'owner': args[0][1], 'kind': args[1][1], 'value': Var('reply')})
    plan = InformingPlan('supplied-provider', route, args, query, 'reply')
    return InformingExample(name, 'source:' + name, 'supplied question ' + name, question, plan, ('explicit teacher',))


def test_full_question_generalizes_two_ref_arguments_and_answer_correspondence():
    model = fit_informing([example('a'), example('b')], [example('held')])
    fresh = example('new')
    result = model.propose(fresh.question)
    assert result.complete and len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.plan == fresh.plan
    assert proposal.training_example_ids == ('a', 'b') and proposal.validation_example_ids == ('held',)
    assert proposal.plan.answer_query.roles['value'] == Var('reply')


def test_role_order_does_not_choose_bindings_and_competing_plans_remain_visible():
    model = fit_informing([example('a'), example('b'), example('c', swap=True), example('d', swap=True)],
                         [example('e'), example('f', swap=True)])
    fresh = example('new')
    reordered = replace(fresh.question, frame=Frame('opaque-question', dict(reversed(tuple(fresh.question.frame.roles.items())))))
    result = model.propose(reordered)
    assert len(result.proposals) == 2
    assert {p.plan.args[0][1] for p in result.proposals} == {Ref('owner:new'), Ref('kind:new')}
    assert all(p.conflicting_validation_example_ids for p in result.proposals)


def test_all_qualifiers_and_asked_slot_remain_literal_constraints():
    model = fit_informing([example('a', qualifiers={'polarity': 'negative', 'count': 1}),
                           example('b', qualifiers={'polarity': 'negative', 'count': 1})],
                          [example('held', qualifiers={'polarity': 'negative', 'count': 1})])
    assert model.propose(example('new', qualifiers={'polarity': 'negative', 'count': 1}).question).proposals
    for question in (example('new').question, example('new', qualifiers={'polarity': 'negative', 'count': True}).question,
                     example('new', qualifiers={'polarity': 'negative', 'count': 1}, asked='location').question):
        assert not model.propose(question).proposals


def test_query_polarity_modality_scope_and_validity_survive_transfer():
    stamp = Interval.at(datetime(2026, 1, 2, tzinfo=timezone.utc))
    def taught(name):
        row = example(name)
        return replace(row, plan=replace(row.plan, answer_query=replace(row.plan.answer_query,
            polarity=False, modality='believed', scope=Ref('scope:explicit'), valid=stamp)))
    model = fit_informing([taught('a'), taught('b')], [taught('held')])
    assert model.propose(taught('fresh').question).proposals[0].plan == taught('fresh').plan


def test_constant_kind_context_and_disjoint_variable_owner():
    model = fit_informing([example('a', kind='shared'), example('b', kind='shared')], [example('held', kind='shared')])
    assert model.propose(example('fresh', kind='shared').question).proposals
    assert not model.propose(example('fresh', kind='different').question).proposals
    with pytest.raises(ValueError, match='disjoint'):
        fit_informing([example('a'), example('b')], [replace(example('a'), id='held', source_id='source:held', text='independent text')])


def test_source_text_splits_and_unvalidated_rivals_are_explicit():
    with pytest.raises(ValueError, match='IDs'):
        fit_informing([example('a'), example('b')], [replace(example('held'), source_id='source:a')])
    with pytest.raises(ValueError, match='texts'):
        fit_informing([example('a'), example('b')], [replace(example('held'), text='SUPPLIED  question a')])
    model = fit_informing([example('a'), example('b'), example('c', route='other'), example('d', route='other')], [example('held')])
    result = model.propose(example('fresh').question)
    assert len(result.proposals) == 1 and any(x.startswith('unvalidated_informing:') for x in result.unresolved)


def test_no_lexical_description_erasure_or_unbounded_search_claim():
    def described(name):
        row = example(name)
        question = replace(row.question, frame=Frame('opaque-question', {'owner': Entity('name', name, ref=Ref('owner:' + name)),
                                                                       'kind': Ref('kind:' + name)}))
        return replace(row, question=question)
    assert not fit_informing([described('a'), described('b')], [described('held')]).propose(described('new').question).proposals
    bounded = fit_informing([example('a'), example('b'), example('c')], [example('held')], max_pairs=1)
    assert not bounded.complete and 'pair_budget_exhausted' in bounded.unresolved
    model = fit_informing([example('a'), example('b')], [example('held')])
    bad = Question(Frame('opaque-question', {'owner': object()}), 'quantity')
    assert not model.propose(bad).complete


def test_plan_requires_bound_explicit_names_and_answer_variable():
    row = example('a')
    with pytest.raises(ValueError, match='answer variable'):
        replace(row.plan, answer_variable='missing')
    with pytest.raises(ValueError, match='unique'):
        replace(row.plan, args=(('who', Ref('owner:a')), ('who', Ref('owner:b'))))


def test_singleton_training_rival_is_not_erased_by_pair_threshold():
    model = fit_informing([example('a'), example('b'), example('single', route='other')], [example('held')])
    result = model.propose(example('fresh').question)
    assert len(result.proposals) == 1
    assert result.proposals[0].conflicting_training_example_ids == ('single',)
    assert 'unvalidated_training_rival:single' in result.unresolved


def test_supported_alternative_covers_training_rival_without_selecting_a_winner():
    model = fit_informing([example('a'), example('b'), example('c', route='other'), example('d', route='other')],
                         [example('held'), example('other-held', route='other')])
    result = model.propose(example('fresh').question)
    assert {p.plan.capability for p in result.proposals} == {'observe', 'other'}
    assert not any(reason.startswith('unvalidated_training_rival:') for reason in result.unresolved)
    assert all(p.conflicting_training_example_ids for p in result.proposals)


def test_owner_only_plan_cannot_drop_question_kind_reference():
    def lossy(name):
        row = example(name)
        return replace(row, plan=InformingPlan('supplied-provider', 'observe', row.plan.args[:1],
            Proposition('opaque-result', {'owner': row.question.frame.roles['owner'], 'value': Var('reply')}), 'reply'))
    model = fit_informing([lossy('a'), lossy('b')], [lossy('held')])
    assert not model.complete
    assert any('unconsumed_question_reference' in reason for reason in model.unresolved)
    assert not model.propose(example('fresh').question).proposals


def test_question_reference_coverage_may_be_in_answer_query_even_when_not_action_argument():
    rows = [replace(example(name), plan=replace(example(name).plan, args=example(name).plan.args[:1]))
            for name in ('a', 'b', 'held')]
    model = fit_informing(rows[:2], rows[2:])
    result = model.propose(example('fresh').question)
    assert result.complete and result.proposals
    assert result.proposals[0].plan.answer_query.roles['kind'] == Ref('kind:fresh')
