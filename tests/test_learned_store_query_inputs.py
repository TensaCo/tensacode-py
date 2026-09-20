"""Real parsed questions retrieve an answer backed by a prior read receipt.

Teachers supply labels, occurrence identities, query/measurement correspondences,
selection policies, and inventory records. The parsing is actually performed by
trained models; the measurement call and subsequent memory retrieval really run.
"""
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.language import Question
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.records import Evidence, Proposition, Ref, Var

from test_learned_informing_inputs import (
    actual_reader, teach_actual_speech, question_text, proper_question, ground_question,
    retain_test_measurement, select_test_sum,
)


@pytest.mark.parametrize('invalidated', ['imported_support', 'source_premise', 'new_measurement', 'operator'])
def test_real_question_retrieves_retained_derivation_without_another_call(actual_reader, invalidated):
    from tensorcode.agent.informing_learning import (
        retain_informing_example, fit_informing_model, admit_informing_model,
    )
    from tensorcode.agent.store_query_learning import (
        retain_store_query_example, fit_store_query_model, admit_store_query_model,
        validate_store_answer,
    )
    from tensorcode.learning.informing import InformingPlan
    from tensorcode.learning.store_query import StoreQueryPlan

    class RecordedQuantity(QuantityPlugin):
        def __init__(self):
            super().__init__()
            self.calls = []

        def execute(self, call, *, key=None):
            self.calls.append(call)
            return super().execute(call, key=key)

    plugin = RecordedQuantity()
    agent = Agent([plugin], reader=actual_reader)
    teach_actual_speech(agent)
    plant, coin = Ref('kind:plant'), Ref('kind:coin')
    informing_records, memory_records = [], []
    for context in ('Alpha', 'Beta', 'Gamma'):
        owner = Ref('owner:' + context)
        amount = retain_test_measurement(plugin, owner, Quantity(3, Unit.of('plant')), plant,
            'measurement:' + context)
        select_test_sum(plugin, owner, plant, (amount,))
        message = agent.interpret(context + '. ' + question_text())
        groups = [agent.interpretations.get(gid) for gid in message.group_ids]
        matching = [g for g in groups if any(
            isinstance(a.meaning, Question) and proper_question(a.frame, 'Shondra')
            for c in g.candidates for a in c.payload.acts)]
        assert len(matching) == 1
        group = matching[0]
        child = ground_question(agent, group, owner, plant)
        query = Proposition('calculated:sum:have', {'subject': owner, 'kind': plant, 'object': Var('answer')})
        observation = InformingPlan('quantity', 'calculate_sum_kind_have',
            (('owner', owner), ('kind', plant)), query, 'answer')
        read_record = retain_informing_example(agent, group.id, child.id, 0, observation,
            basis=('explicit full-question measurement correspondence',))
        memory_record = retain_store_query_example(agent, group.id, child.id, 0,
            StoreQueryPlan(query, 'answer', (None,)),
            basis=('explicit full-question retained-evidence query correspondence',))
        assert not isinstance(read_record, Unknown), read_record
        assert not isinstance(memory_record, Unknown), memory_record
        informing_records.append(read_record)
        memory_records.append(memory_record)

    fitted = fit_informing_model(agent, informing_records[:2], informing_records[2:])
    assert not isinstance(fitted, Unknown), fitted
    observation_model = admit_informing_model(agent, fitted, reason='explicit measurement-model admission')
    assert not isinstance(observation_model, Unknown), observation_model
    agent.informing_model = observation_model
    fresh_owner = Ref('owner:fresh-memory-execution')
    expected = Quantity(7, Unit.of('plant'))
    first = retain_test_measurement(plugin, fresh_owner, Quantity(3, Unit.of('plant')), plant, 'measurement:fresh-first')
    second = retain_test_measurement(plugin, fresh_owner, Quantity(4, Unit.of('plant')), plant, 'measurement:fresh-second')
    retain_test_measurement(plugin, fresh_owner, Quantity(19, Unit.of('coin')), coin, 'measurement:fresh-coins')
    calculation = select_test_sum(plugin, fresh_owner, plant, (first, second))

    def select_reading(group):
        child = ground_question(agent, group, fresh_owner, plant)
        current = agent.interpretations.get(group.id)
        return InterpretationDecision(child.id, 'explicit selected grounded question',
            compared_revision=current.revision,
            compared_candidate_ids=tuple(c.id for c in current.candidates))

    def select_plan(group):
        assert len(group.candidates) == 1
        return InterpretationDecision(group.candidates[0].id, 'explicit supported plan selection',
            compared_revision=group.revision,
            compared_candidate_ids=tuple(c.id for c in group.candidates))

    agent.interpretation_selector = select_reading
    agent.informing_selector = select_plan
    observed = agent.turn(question_text()).outcomes[0]
    assert observed.status == 'answered', (observed.reason, observed.verified)
    assert observed.answer == [expected] and observed.receipt.status == 'applied'
    assert len(plugin.calls) == 1
    support, = agent.store.propositions('calculated:sum:have')
    assert support.proposition.roles == {'subject': fresh_owner, 'kind': plant, 'object': expected}
    from tensorcode.derivations import DerivationReference, validate_record_support
    evidence, = support.evidence
    assert evidence.method == 'authenticated-derivation-import'
    assert evidence.derived_from == (support.id,)
    retained_observation, = [source for source in agent.interpretations.sources()
        if source.modality == 'informing-answer' and source.payload['receipt'] == observed.receipt]
    reference, = retained_observation.payload['observations']
    assert type(reference) is DerivationReference
    assert evidence.source == Ref(reference.source_store_id)
    assert reference.proposition == support.proposition
    assert evidence.locator != retained_observation.id
    assert validate_record_support(agent.store, support.id) is True
    assert not agent.store.propositions('have')  # Arithmetic did not mint a fresh measurement.

    # The caller explicitly switches to memory. Missing query authority is not
    # permission to restore the old automatic passive conversion.
    agent.informing_model = None
    no_model = agent.turn(question_text()).outcomes[0]
    assert no_model.status == 'unknown' and no_model.receipt is None
    fitted = fit_store_query_model(agent, memory_records[:2], memory_records[2:])
    assert not isinstance(fitted, Unknown), fitted
    memory_model = admit_store_query_model(agent, fitted, reason='explicit memory-query-model admission')
    assert not isinstance(memory_model, Unknown), memory_model
    agent.store_query_model = memory_model
    no_choice = agent.turn(question_text()).outcomes[0]
    assert no_choice.status == 'unknown' and no_choice.receipt is None
    agent.store_query_selector = select_plan
    remembered = agent.turn(question_text()).outcomes[0]
    assert remembered.status == 'answered', (remembered.reason, remembered.verified)
    assert remembered.answer == [expected] and remembered.receipt is None
    assert remembered.act.frame.roles['object'].text == 'how many plants'
    assert remembered.act.frame.roles['object'].features['noun'] == 'plant'
    assert len(plugin.calls) == 1
    assert remembered.verified.record_ids == (support.id,)
    assert validate_store_answer(agent, remembered.verified) is True

    counter = replace(support.proposition, polarity=False)
    agent.store.assert_(counter, Evidence(Ref('fixture:contrary-observation'), datetime.now(timezone.utc)))
    assert isinstance(validate_store_answer(agent, remembered.verified), Unknown)
    contradicted = agent.turn(question_text()).outcomes[0]
    assert contradicted.status == 'unknown' and contradicted.receipt is None
    agent.store.supersede(counter, why='explicit withdrawal of contrary fixture evidence')
    restored = agent.turn(question_text()).outcomes[0]
    assert restored.status == 'answered' and restored.answer == [expected]
    if invalidated == 'imported_support':
        agent.store.supersede(support.proposition, why='explicit withdrawal of imported derivation')
    elif invalidated == 'source_premise':
        plugin.mind.supersede(first, why='explicit withdrawal of source measurement')
    elif invalidated == 'new_measurement':
        retain_test_measurement(plugin, fresh_owner, Quantity(2, Unit.of('plant')), plant, 'measurement:additional')
    else:
        from tensorcode.derivations import withdraw_operator
        assert withdraw_operator(plugin.mind, plugin.calculate(calculation).operator,
            reason='explicit withdrawal of supplied arithmetic authority') is True
    assert isinstance(validate_store_answer(agent, restored.verified), Unknown)
    retracted = agent.turn(question_text()).outcomes[0]
    assert retracted.status == 'unknown' and retracted.receipt is None
    assert len(plugin.calls) == 1

    if invalidated == 'new_measurement':
        # The new source record changes the validation population, but cannot
        # silently become an additional operand in the separately selected sum.
        agent.informing_model = observation_model
        fresh = agent.turn(question_text()).outcomes[0]
        assert fresh.status == 'answered', (fresh.reason, fresh.verified)
        assert fresh.answer == [expected]
        assert len(plugin.calls) == 2
