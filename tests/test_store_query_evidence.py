"""Learned full-question memory queries retain and revalidate supporting evidence."""
from dataclasses import replace
from datetime import datetime, timezone
import pytest
from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.agent.store_query_learning import (retain_store_query_example, fit_store_query_model,
    admit_store_query_model, propose_store_queries, select_store_query, evaluate_store_query,
    validate_store_answer, answer_store_question)
from tensorcode.language import Question, Frame
from tensorcode.learning.store_query import StoreQueryPlan
from tensorcode.records import Proposition, Ref, Var, Evidence, Interval
from tensorcode.outcomes import Unknown


def case(name, *, scope=None, valid=None):
    ref = Ref('owner:' + name)
    question = Question(Frame('opaque', {'subject': ref}, {'qualifier': 'explicit'}), 'answer')
    plan = StoreQueryPlan(Proposition('opaque-result', {'subject': ref, 'result': Var('reply')},
        scope=scope, valid=Interval() if valid is None else valid), 'reply', (scope,))
    return question, plan


def language(agent, name, question):
    source = agent.interpretations.add_source('Full retained context ' + name, provider='test supplied question')
    group = agent.interpretations.create_group(source.id)
    act = Act('question', question, question.frame)
    candidate = agent.interpretations.propose(group.id, SentenceAlternative(None, (act,)))
    return group.id, candidate.id, act


def setup(*, scope=None, valid=None):
    agent = Agent()
    records = []
    for name in ('a', 'b', 'held'):
        q, plan = case(name, scope=scope, valid=Interval() if valid is None else valid)
        gid, cid, _ = language(agent, name, q)
        record = retain_store_query_example(agent, gid, cid, 0, plan, basis=('explicit teacher query',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    fit = fit_store_query_model(agent, records[:2], records[2:])
    assert not isinstance(fit, Unknown), fit
    admitted = admit_store_query_model(agent, fit, reason='explicit model admission')
    assert not isinstance(admitted, Unknown), admitted
    agent.store_query_model = admitted
    agent.store_query_selector = lambda g: InterpretationDecision(g.candidates[0].id, 'explicit fixture choice')
    return agent, admitted, records


def selected(agent, *, scope=None, valid=None):
    q, plan = case('fresh', scope=scope, valid=Interval() if valid is None else valid)
    gid, cid, act = language(agent, 'fresh', q)
    agent.interpretations.select(gid, cid, reason='explicit question choice')
    dep = agent.capture_task_dependency(gid, basis=('selected question',))
    report = propose_store_queries(agent, agent.store_query_model, q, parent_dependency=dep)
    assert not isinstance(report, Unknown), report
    choice = select_store_query(agent, report.group_id)
    assert not isinstance(choice, Unknown), choice
    return choice, act, dep


def assert_fact(agent, proposition):
    return agent.store.assert_(proposition, Evidence(Ref('source:fixture'), datetime.now(timezone.utc)))


def fact(*, scope=None, valid=None, result=7):
    return Proposition('opaque-result', {'subject': Ref('owner:fresh'), 'result': result}, scope=scope, valid=Interval() if valid is None else valid)


def test_learned_store_query_retains_exact_support_and_requires_explicit_choice():
    agent, _, records = setup()
    assert records[0].example.text == 'Full retained context a'
    stored = assert_fact(agent, fact())
    choice, _, _ = selected(agent)
    answer = evaluate_store_query(agent, choice)
    assert not isinstance(answer, Unknown), answer
    assert answer.answers == (7,) and answer.record_ids == (stored.id,)
    assert validate_store_answer(agent, answer) is True
    agent.store_query_selector = None
    q, _ = case('fresh')
    gid, cid, act = language(agent, 'fresh2', q)
    agent.interpretations.select(gid, cid, reason='choice')
    dep = agent.capture_task_dependency(gid, basis=('choice',))
    assert answer_store_question(agent, q, act, [], parent_dependency=dep).status == 'unknown'


def test_populated_store_does_not_supply_an_implicit_question_mapping():
    agent = Agent()
    assert_fact(agent, fact())
    assert isinstance(agent.lookup(case('fresh')[0]), Unknown)


@pytest.mark.parametrize('change', ['withdraw', 'retract', 'tamper', 'answer', 'source', 'contradiction'])
def test_cached_answer_loses_authority_when_evidence_changes(change):
    agent, model, _ = setup()
    record = assert_fact(agent, fact())
    choice, _, _ = selected(agent)
    answer = evaluate_store_query(agent, choice)
    assert not isinstance(answer, Unknown), answer
    if change == 'withdraw': agent.interpretations.unset(model.group_id, reason='withdraw')
    if change == 'retract': agent.store.supersede(record.proposition, why='withdraw')
    if change == 'tamper': record.evidence.clear()
    if change == 'answer': answer = replace(answer, answers=(999,))
    if change == 'source':
        agent.interpretations._sources[answer.evidence_source_id].payload['answers'] = (999,)
    if change == 'contradiction': assert_fact(agent, replace(fact(), polarity=False))
    assert isinstance(validate_store_answer(agent, answer), Unknown)


def test_conflicting_polarity_defers_and_empty_store_is_not_closed_world():
    agent, _, _ = setup()
    choice, _, _ = selected(agent)
    assert isinstance(evaluate_store_query(agent, choice), Unknown)
    assert_fact(agent, fact())
    assert_fact(agent, replace(fact(), polarity=False))
    assert isinstance(evaluate_store_query(agent, choice), Unknown)


def test_scope_interval_and_typed_values_are_exact_constraints():
    scope = Ref('scope:explicit')
    valid = Interval.at(datetime(2026, 1, 1, tzinfo=timezone.utc))
    agent, _, _ = setup(scope=scope, valid=Interval() if valid is None else valid)
    choice, _, _ = selected(agent, scope=scope, valid=Interval() if valid is None else valid)
    assert_fact(agent, fact())
    assert_fact(agent, fact(scope=scope))
    assert isinstance(evaluate_store_query(agent, choice), Unknown)
    assert_fact(agent, fact(scope=scope, valid=Interval() if valid is None else valid))
    answer = evaluate_store_query(agent, choice)
    assert not isinstance(answer, Unknown), answer
    assert answer.answers == (7,)


def test_publication_tampering_does_not_create_query_authority(monkeypatch):
    agent, _, _ = setup()
    original = agent.interpretations.propose
    def changed(gid, payload, **kwargs):
        if hasattr(payload, 'plan'):
            payload = replace(payload, plan=replace(payload.plan, answer_variable='forged'))
        return original(gid, payload, **kwargs)
    monkeypatch.setattr(agent.interpretations, 'propose', changed)
    q, _ = case('fresh')
    gid, cid, _ = language(agent, 'fresh', q)
    agent.interpretations.select(gid, cid, reason='question choice')
    dep = agent.capture_task_dependency(gid, basis=('choice',))
    assert isinstance(propose_store_queries(agent, agent.store_query_model, q, parent_dependency=dep), Unknown)


def test_store_callback_cannot_withdraw_model_while_returning_old_records(monkeypatch):
    agent, model, _ = setup()
    assert_fact(agent, fact())
    choice, _, _ = selected(agent)
    original = agent.store.propositions
    def changed(*args, **kwargs):
        records = original(*args, **kwargs)
        agent.interpretations.unset(model.group_id, reason='withdraw during store read')
        return records
    monkeypatch.setattr(agent.store, 'propositions', changed)
    assert isinstance(evaluate_store_query(agent, choice), Unknown)


def test_informing_failure_does_not_fall_through_to_memory():
    from tensorcode.agent.understand import Sentence
    agent, _, _ = setup()
    assert_fact(agent, fact())
    _, act, dep = selected(agent)
    agent.informing_model = object()  # Invalid configured authority cannot authorize a read.
    outcome = agent.ask(None, act, [], interpretation_dependency=dep)
    assert outcome.status == 'unknown'


@pytest.mark.parametrize('kind', ['unbounded', 'touching', 'disjoint'])
def test_opposite_polarity_overlapping_intervals_block_but_disjoint_do_not(kind):
    from datetime import timedelta
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    valid = Interval(start, end)
    other = {'unbounded': Interval(), 'touching': Interval(end, end + timedelta(days=1)),
             'disjoint': Interval(end + timedelta(seconds=1), None)}[kind]
    agent, _, _ = setup(valid=valid)
    choice, _, _ = selected(agent, valid=valid)
    assert_fact(agent, fact(valid=valid))
    assert_fact(agent, replace(fact(valid=other), polarity=False))
    answer = evaluate_store_query(agent, choice)
    assert isinstance(answer, Unknown) is (kind != 'disjoint')


@pytest.mark.parametrize('kind', ['empty', 'object', 'bad_source', 'bad_date', 'bad_premises', 'derived'])
def test_absent_malformed_or_unauthenticated_derived_evidence_cannot_support_answer(kind):
    agent, _, _ = setup()
    choice, _, _ = selected(agent)
    record = assert_fact(agent, fact())
    evidence = record.evidence[0]
    record.evidence[:] = {
        'empty': [], 'object': [object()],
        'bad_source': [replace(evidence, source='source:fake')],
        'bad_date': [replace(evidence, observed_at='today')],
        'bad_premises': [replace(evidence, derived_from=['claim:missing'])],
        'derived': [replace(evidence, derived_from=('claim:missing',))],
    }[kind]
    assert isinstance(evaluate_store_query(agent, choice), Unknown)
