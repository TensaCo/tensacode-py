"""Supplied full-question teaching learns guarded read-only answer calls."""
from dataclasses import replace
import pytest
from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.plugin import Plugin, Capability, Informs, Param
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.agent.informing_learning import (retain_informing_example, fit_informing_model,
    admit_informing_model, get_informing_model, propose_informing, select_informing,
    answer_informing_question)
from tensorcode.actions import Receipt
from tensorcode.outcomes import Unknown
from tensorcode.records import Proposition, Ref, Var
from test_informing_learning import example


class Provider(Plugin):
    def __init__(self):
        super().__init__('supplied-provider')
        self.calls = []
        self.effect = 'read'
        self.wrong = False
        self.hook = None
        self.reveal_hook = None
    def capabilities(self):
        return (Capability('observe', (Param('who', 'ref'), Param('which', 'ref')),
            informs=(Informs('opaque-result', 'unused', 'who', Proposition('opaque-result',
                {'owner': Var('who'), 'kind': Var('which'), 'value': Var('reply')}), 'reply'),),
            effect_kind=self.effect),)
    def observe_evidence(self):
        if self.hook:
            self.hook()
        return {'raw': 'observed'}
    def execute(self, action, *, key=None):
        self.calls.append(action)
        return Receipt(action, 'applied')
    def reveal(self, cap, args, receipt):
        if self.reveal_hook:
            self.reveal_hook()
        return (Proposition('opaque-result', {'owner': Ref('wrong') if self.wrong else args['who'],
            'kind': args['which'], 'value': 7}),)


def language(agent, row):
    source = agent.interpretations.add_source('Context ' + row.id + '. Full question preserved.', provider='explicit test syntax')
    group = agent.interpretations.create_group(source.id)
    act = Act('question', row.question, row.question.frame)
    child = agent.interpretations.propose(group.id, SentenceAlternative(None, (act,)))
    return group.id, child.id, act


def setup(*, selector=True):
    provider = Provider()
    agent = Agent([provider])
    records = []
    for name in ('a', 'b', 'held'):
        row = example(name)
        gid, cid, _ = language(agent, row)
        record = retain_informing_example(agent, gid, cid, 0, row.plan, basis=('explicit teacher',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    handle = fit_informing_model(agent, records[:2], records[2:])
    assert not isinstance(handle, Unknown), handle
    handle = admit_informing_model(agent, handle, reason='explicit admission')
    assert not isinstance(handle, Unknown), handle
    agent.informing_model = handle
    if selector:
        agent.informing_selector = lambda group: InterpretationDecision(group.candidates[0].id, 'explicit fixture plan policy')
    return agent, provider, handle, records


def request(agent):
    gid, cid, act = language(agent, example('fresh'))
    agent.interpretations.select(gid, cid, reason='explicit question selection')
    dep = agent.capture_task_dependency(gid, basis=('selected full question',))
    return act, dep


def answer(agent):
    act, dep = request(agent)
    return answer_informing_question(agent, act.meaning, act, [], parent_dependency=dep)


def test_full_source_teaching_explicit_admission_and_verified_answer():
    agent, provider, handle, records = setup()
    assert records[0].example.text == 'Context a. Full question preserved.'
    assert isinstance(get_informing_model(agent, replace(handle, dependency=None)), Unknown)
    outcome = answer(agent)
    assert outcome.status == 'answered', outcome
    assert outcome.answer == [7] and len(provider.calls) == 1
    assert dict(provider.calls[0].args) == dict(example('fresh').plan.args)


def test_no_model_or_no_explicit_plan_selection_does_not_observe():
    agent, provider, _, _ = setup(selector=False)
    assert answer(agent).status == 'unknown' and not provider.calls
    agent.informing_model = None
    assert answer(agent) is None and not provider.calls


@pytest.mark.parametrize('change', ['write', 'wrong_answer', 'before_withdraw', 'after_withdraw'])
def test_bad_contract_or_changed_authority_does_not_assert_answer(change):
    agent, provider, handle, _ = setup()
    if change == 'write': provider.effect = 'write'
    if change == 'wrong_answer': provider.wrong = True
    withdraw = lambda: agent.interpretations.unset(handle.group_id, reason='withdraw support')
    if change == 'before_withdraw': provider.hook = withdraw
    if change == 'after_withdraw': provider.reveal_hook = withdraw
    outcome = answer(agent)
    assert outcome.status == 'unknown', outcome
    assert not tuple(agent.store.propositions())
    assert len(provider.calls) == (1 if change in ('wrong_answer', 'after_withdraw') else 0)
    if provider.calls: assert outcome.receipt.status == 'applied'


def test_full_question_mutation_is_not_a_supported_projection():
    agent, provider, handle, _ = setup()
    act, dep = request(agent)
    changed = replace(act.meaning, asked='other')
    assert isinstance(propose_informing(agent, handle, changed, parent_dependency=dep), Unknown)
    assert not provider.calls


def test_publication_payload_mutation_is_not_authenticated(monkeypatch):
    agent, _, handle, _ = setup()
    act, dep = request(agent)
    original = agent.interpretations.propose
    def tamper(group_id, payload, **kwargs):
        if hasattr(payload, 'plan'):
            payload = replace(payload, plan=replace(payload.plan, capability='other'))
        return original(group_id, payload, **kwargs)
    monkeypatch.setattr(agent.interpretations, 'propose', tamper)
    assert isinstance(propose_informing(agent, handle, act.meaning, parent_dependency=dep), Unknown)


def test_retained_teaching_and_admission_are_authenticated():
    agent, _, handle, records = setup()
    bad = replace(records[0], example=replace(records[0].example, text='forged'))
    assert isinstance(fit_informing_model(agent, [bad, records[1]], records[2:]), Unknown)
    agent.interpretations.unset(handle.group_id, reason='withdraw')
    assert isinstance(get_informing_model(agent, handle), Unknown)


def test_declared_observation_query_must_match_the_learned_query(monkeypatch):
    agent, provider, _, _ = setup()
    cap = provider.capabilities()[0]
    bad = replace(cap.informs[0], query=Proposition('other-result',
        {'owner': Var('who'), 'kind': Var('which'), 'value': Var('reply')}))
    monkeypatch.setattr(provider, 'capabilities', lambda: (replace(cap, informs=(bad,)),))
    outcome = answer(agent)
    assert outcome.status == 'unknown' and not provider.calls


def test_capability_callback_cannot_withdraw_model_after_authority_check(monkeypatch):
    agent, provider, handle, _ = setup()
    original = provider.capabilities
    count = 0
    def capabilities():
        nonlocal count
        count += 1
        if count == 2:
            agent.interpretations.unset(handle.group_id, reason='late provider withdrawal')
        return original()
    monkeypatch.setattr(provider, 'capabilities', capabilities)
    assert answer(agent).status == 'unknown'
    assert not provider.calls


def test_answer_evidence_preserves_provider_and_retained_observation_location():
    agent, provider, _, _ = setup()
    assert answer(agent).status == 'answered'
    record, = agent.store.propositions()
    evidence, = record.evidence
    assert evidence.source == Ref('plugin:' + provider.name)
    assert evidence.derived_from == ()  # Workspace source IDs are not store premises.
    retained = agent.interpretations.get_source(evidence.locator)
    assert retained.provider == provider.name
    assert retained.payload['observations'] == [record.proposition]
    assert retained.payload['receipt'].status == 'applied'


@pytest.mark.parametrize('field,value', [
    ('scope', Ref('scope:unrequested')), ('valid', 'interval'),
    ('polarity', 1), ('modality', 'possible'),
])
def test_returned_metadata_is_not_wildcarded_or_type_coerced(monkeypatch, field, value):
    from datetime import datetime, timezone
    from tensorcode.records import Interval
    agent, provider, _, _ = setup()
    if value == 'interval':
        value = Interval.at(datetime(2026, 1, 1, tzinfo=timezone.utc))
    original = provider.reveal
    monkeypatch.setattr(provider, 'reveal', lambda *args: tuple(
        replace(item, **{field: value}) for item in original(*args)))
    outcome = answer(agent)
    assert outcome.status == 'unknown' and outcome.receipt.status == 'applied'
    assert not agent.store.propositions()


def test_answer_match_preserves_exact_intervals_scope_and_typed_constraints():
    from datetime import datetime, timezone
    from tensorcode.records import Interval
    from tensorcode.agent.informing_learning import _match_answer
    first = Interval.at(datetime(2026, 1, 1, tzinfo=timezone.utc))
    later = Interval.at(datetime(2026, 1, 2, tzinfo=timezone.utc))
    query = Proposition('measured', {'participant': True, 'answer': Var('a'), 'echo': Var('a')},
        scope=Ref('scope:one'), valid=first)
    fact = replace(query, roles={'participant': True, 'answer': True, 'echo': True})
    assert _match_answer(query, fact) == {'a': True}
    assert _match_answer(query, replace(fact, valid=later)) is None
    assert _match_answer(query, replace(fact, scope=None)) is None
    assert _match_answer(query, replace(fact, roles={**fact.roles, 'participant': 1})) is None
    assert _match_answer(query, replace(fact, roles={**fact.roles, 'echo': 1})) is None
    assert _match_answer(query, replace(fact, roles={**fact.roles, 'additional': True})) is None


@pytest.mark.parametrize('explicit_empty', [False, True])
def test_missing_observation_is_not_an_observed_empty_collection(monkeypatch, explicit_empty):
    agent, provider, _, _ = setup()
    original = provider.reveal
    def reveal(*args):
        if not explicit_empty:
            return ()
        return tuple(replace(item, roles={**item.roles, 'value': ()}) for item in original(*args))
    monkeypatch.setattr(provider, 'reveal', reveal)
    outcome = answer(agent)
    assert outcome.receipt.status == 'applied'
    assert outcome.status == ('answered' if explicit_empty else 'unknown')
    if explicit_empty:
        assert outcome.answer == [()]
    else:
        assert not agent.store.propositions()
