"""Learned retrieval revalidates transitive evidence across provider stores."""
from dataclasses import replace
from datetime import datetime, timezone
import pytest
from tensorcode.derivations import (admit_operator, derive, export_derivation, import_derivation,
    withdraw_operator)
from tensorcode.records import Store, Proposition, Evidence, Ref
from tensorcode.outcomes import Unknown
from tensorcode.agent.store_query_learning import evaluate_store_query, validate_store_answer
from test_store_query_evidence import setup, selected, fact


@pytest.mark.parametrize('change', ['withdraw', 'retract', 'evidence', 'contradiction', 'population'])
def test_learned_answer_tracks_provider_derivation_not_just_local_record(change):
    agent, _, _ = setup()
    choice, _, _ = selected(agent)
    provider = Store()
    premise = Proposition('measured-input', {'amount': 7})
    evidence = Evidence(Ref('source:measurement'), datetime.now(timezone.utc))
    record = provider.assert_(premise, evidence)
    handle = admit_operator(provider, 'explicit test projection',
        lambda premises, params: fact(result=premises[0].role('amount')),
        reason='authored projection for mechanism test')
    receipt = derive(provider, handle, (record.id,), basis=('explicit premise choice',),
                     population_predicates=('measured-input',))
    assert not isinstance(receipt, Unknown), receipt
    reference = export_derivation(provider, receipt)
    imported = import_derivation(agent.store, reference)
    assert not isinstance(imported, Unknown), imported
    answer = evaluate_store_query(agent, choice)
    assert not isinstance(answer, Unknown), answer
    assert answer.answers == (7,)
    assert validate_store_answer(agent, answer) is True
    unchanged_local = list(agent.store.propositions())
    if change == 'withdraw':
        withdraw_operator(provider, handle, reason='withdraw supplied projection')
    elif change == 'retract':
        provider.supersede(premise)
    elif change == 'evidence':
        record.evidence.clear()
    elif change == 'contradiction':
        provider.assert_(replace(premise, polarity=False), evidence)
    else:
        provider.assert_(Proposition('measured-input', {'amount': 8}), evidence)
    assert agent.store.propositions() == unchanged_local
    assert isinstance(validate_store_answer(agent, answer), Unknown)
    assert isinstance(evaluate_store_query(agent, choice), Unknown)


@pytest.mark.parametrize('withdraw_during_reveal', [False, True])
def test_information_goal_preserves_derived_support(withdraw_during_reveal, monkeypatch):
    from tensorcode.agent import Agent
    from tensorcode.agent.plugin import Plugin, Capability, Informs
    from tensorcode.agent.understand import Act
    from tensorcode.goals import GoalSpec, Condition
    from tensorcode.language import Frame
    from tensorcode.outcomes import Receipt
    from tensorcode.runtime import use

    source = Store()
    premise = source.assert_(Proposition('input', {'value': 7}),
        Evidence(Ref('source:test'), datetime.now(timezone.utc)))
    handle = admit_operator(source, 'test projection', lambda operands, params: fact(),
                            reason='authored test information projection')
    proof = derive(source, handle, (premise.id,), basis=('explicit test evidence',))
    reference = export_derivation(source, proof)
    cap = Capability('report', (), informs=(Informs('opaque-result', 'result', 'subject'),), effect_kind='read')

    class Reporter(Plugin):
        def capabilities(self):
            return (cap,)
        def execute(self, action, *, key=None):
            return Receipt(action, 'applied', idempotency_key=key)
        def holds(self, capability, args):
            return True
        def reveal(self, capability, args, receipt):
            yield reference
            if withdraw_during_reveal:
                source.supersede(premise.proposition)

    reporter = Reporter('reporter')
    agent = Agent([reporter])
    monkeypatch.setattr(agent, 'choose_plan', lambda goal: (reporter, cap, {}))
    goal = GoalSpec((Condition('reported', {}),), basis=('authored test goal',))
    frame = Frame('report', {})
    with use(agent.runtime):
        result = agent._execute_goal(goal, Act('request', None, frame), [])
    if withdraw_during_reveal:
        assert result.status == 'unverified'
        assert result.answer is None
    else:
        assert result.status == 'done'
        assert result.answer == [reference.proposition]
        stored, = agent.store.propositions('opaque-result')
        assert all(e.derived_from for e in stored.evidence)


def test_cached_multi_answer_uses_one_cross_store_validation_read_set():
    agent, _, _ = setup()
    choice, _, _ = selected(agent)
    sources = [Store(), Store()]
    evidence = Evidence(Ref('source:test'), datetime.now(timezone.utc))
    premises = [s.assert_(Proposition('input', {'amount': n}), evidence)
                for s, n in zip(sources, (7, 8))]
    armed = [False]
    def second_operator(operands, params):
        if armed[0]:
            sources[0].supersede(premises[0].proposition)
        return fact(result=8)
    for source, premise, operator in zip(sources, premises,
            (lambda operands, params: fact(result=7), second_operator)):
        handle = admit_operator(source, 'test projection', operator, reason='explicit test operator')
        proof = derive(source, handle, (premise.id,), basis=('explicit operand',))
        assert not isinstance(import_derivation(agent.store, export_derivation(source, proof)), Unknown)
    answer = evaluate_store_query(agent, choice)
    assert not isinstance(answer, Unknown), answer
    assert set(answer.answers) == {7, 8}
    armed[0] = True
    assert isinstance(validate_store_answer(agent, answer), Unknown)
