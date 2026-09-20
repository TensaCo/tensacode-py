"""Explicit memory-query teaching transfers references without query heuristics."""
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tensorcode.language import Frame, Question
from tensorcode.records import Interval, Proposition, Ref, Var
from tensorcode.learning.store_query import StoreQueryExample, StoreQueryPlan, fit_store_queries


def example(name, *, predicate='opaque-answer', scope=None, features=None):
    owner, kind = Ref('owner:' + name), Ref('kind:' + name)
    question = Question(Frame('opaque-question', {'owner': owner, 'kind': kind}, features or {}), 'answer')
    query = Proposition(predicate, {'owner': owner, 'kind': kind, 'answer': Var('reply')}, scope=scope)
    return StoreQueryExample(name, 'source:' + name, 'explicit independent text ' + name,
                             question, StoreQueryPlan(query, 'reply', (scope,)), ('supplied teaching',))


def test_reference_transfer_retains_full_query_and_provenance():
    model = fit_store_queries([example('a'), example('b')], [example('held')])
    result = model.propose(example('fresh').question)
    assert result.complete and not result.unresolved
    assert len(result.proposals) == 1
    assert result.proposals[0].plan == example('fresh').plan
    assert result.proposals[0].training_example_ids == ('a', 'b')
    assert result.proposals[0].validation_example_ids == ('held',)
    assert model.training_examples[0].basis == ('supplied teaching',)


def test_literal_scope_none_is_not_wildcard_and_query_metadata_survives():
    stamp = Interval.at(datetime(2026, 1, 2, tzinfo=timezone.utc))
    def taught(name):
        row = example(name, scope=Ref('scope:fixed'))
        return replace(row, plan=replace(row.plan, query=replace(row.plan.query,
            polarity=False, modality='believed', valid=stamp), allowed_scopes=(None, Ref('scope:fixed'))))
    model = fit_store_queries([taught('a'), taught('b')], [taught('held')])
    plan = model.propose(taught('fresh').question).proposals[0].plan
    assert plan == taught('fresh').plan and plan.query.scope == Ref('scope:fixed')
    assert example('unscoped').plan.query.scope is None
    with pytest.raises(ValueError, match='not explicitly allowed'):
        StoreQueryPlan(taught('a').plan.query, 'reply', (None,))
    for scopes in ((), (None, None), ('not-a-ref',)):
        with pytest.raises(ValueError, match='exact set'):
            StoreQueryPlan(example('a').plan.query, 'reply', scopes)
    with pytest.raises(ValueError, match='answer variable'):
        replace(example('a').plan, answer_variable='missing')


def test_question_qualifiers_are_exact_typed_constraints():
    rows = [example(n, features={'qualifier': 1}) for n in ('a', 'b', 'held')]
    model = fit_store_queries(rows[:2], rows[2:])
    assert model.propose(example('fresh', features={'qualifier': 1}).question).proposals
    assert not model.propose(example('fresh', features={'qualifier': True}).question).proposals
    assert not model.propose(example('fresh').question).proposals
    assert not model.propose(replace(rows[0].question, asked='different')).proposals


def test_competing_queries_retained_and_singleton_rival_not_erased():
    model = fit_store_queries([example('a'), example('b'), example('c', predicate='rival'),
                              example('d', predicate='rival')],
                             [example('held'), example('other-held', predicate='rival')])
    result = model.propose(example('fresh').question)
    assert len(result.proposals) == 2 and not result.unresolved
    assert all(p.conflicting_training_example_ids for p in result.proposals)
    assert all(p.conflicting_validation_example_ids for p in result.proposals)
    incomplete_support = fit_store_queries([example('a'), example('b'), example('single', predicate='rival')],
                                          [example('held')]).propose(example('fresh').question)
    assert len(incomplete_support.proposals) == 1
    assert 'unvalidated_training_rival:single' in incomplete_support.unresolved


def test_unvalidated_pair_remains_unresolved_alongside_validated_plan():
    result = fit_store_queries([example('a'), example('b'), example('c', predicate='rival'),
                               example('d', predicate='rival')], [example('held')]).propose(example('fresh').question)
    assert any(reason.startswith('unvalidated_store_query:') for reason in result.unresolved)


def test_allowed_scope_declaration_cannot_hide_dropped_question_participant():
    def lossy(name):
        row = example(name)
        query = replace(row.plan.query, roles={'owner': row.question.frame.roles['owner'], 'answer': Var('reply')})
        return replace(row, plan=StoreQueryPlan(query, 'reply', (None, row.question.frame.roles['kind'])))
    model = fit_store_queries([lossy('a'), lossy('b')], [lossy('held')])
    assert not model.complete and not model.propose(example('fresh').question).proposals
    assert any('unconsumed_question_reference' in value for value in model.unresolved)


def test_independent_source_text_and_reference_splits():
    for field, value, message in (('source_id', 'source:a', 'IDs'),
                                  ('text', 'EXPLICIT independent text a', 'texts')):
        with pytest.raises(ValueError, match=message):
            fit_store_queries([example('a'), example('b')], [replace(example('held'), **{field: value})])
    with pytest.raises(ValueError, match='disjoint'):
        fit_store_queries([example('a'), example('b')],
                          [replace(example('a'), id='held', source_id='source:held', text='new heldout text')])


def test_budget_and_unsupported_shape_remain_explicit():
    bounded = fit_store_queries([example('a'), example('b'), example('c')], [example('held')], max_pairs=1)
    assert not bounded.complete and 'pair_budget_exhausted' in bounded.unresolved
    model = fit_store_queries([example('a'), example('b')], [example('held')])
    malformed = Question(Frame('opaque-question', {'unsupported': object()}), 'answer')
    assert not model.propose(malformed).complete
