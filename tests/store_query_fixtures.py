"""Explicit retained question-to-store-query teaching for mechanism tests.

Each caller supplies the complete query, answer variable, and exact scope. This
fixture does not derive a query from language, role order, or predicate names.
"""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace

from tensorcode.agent import InterpretationDecision
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from informing_fixtures import selected_question_dependency


def teach_store_query(agent, question, plan):
    from tensorcode.agent.store_query_learning import (
        retain_store_query_example, fit_store_query_model, admit_store_query_model,
    )
    records = []
    for index in range(3):
        references = {}
        def variant(value, *, input_value=False):
            if type(value) is Ref:
                if input_value and value not in references:
                    references[value] = Ref(f'fixture-store-query:{index}:{len(references)}')
                return references.get(value, value)
            if type(value) is dict:
                return {key: variant(item, input_value=input_value) for key, item in value.items()}
            if type(value) in (tuple, list):
                return type(value)(variant(item, input_value=input_value) for item in value)
            if is_dataclass(value):
                return replace(value, **{field.name: variant(getattr(value, field.name), input_value=input_value)
                                         for field in fields(value) if field.init})
            return deepcopy(value)
        taught_question = variant(question, input_value=True)
        taught_plan = variant(plan)
        source = agent.interpretations.add_source(f'Explicit store-query teaching context {index}',
                                                  provider='authored-store-query-fixture')
        group = agent.interpretations.create_group(source.id)
        candidate = agent.interpretations.propose(group.id, SentenceAlternative(None,
            (Act('question', taught_question, taught_question.frame),), provenance='authored-store-query-fixture'))
        record = retain_store_query_example(agent, group.id, candidate.id, 0, taught_plan,
            basis=('explicit full-question to exact scoped query fixture',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    model = fit_store_query_model(agent, records[:2], records[2:])
    assert not isinstance(model, Unknown), model
    handle = admit_store_query_model(agent, model, reason='explicit fixture store-query model admission')
    assert not isinstance(handle, Unknown), handle
    agent.store_query_model = handle
    def choose(group):
        choices = [candidate for candidate in group.candidates if hasattr(candidate.payload, 'plan')]
        return InterpretationDecision(choices[0].id if len(choices) == 1 else None,
            'fixture explicitly authorizes a sole complete taught store-query plan',
            compared_revision=group.revision,
            compared_candidate_ids=tuple(candidate.id for candidate in group.candidates))
    agent.store_query_selector = choose
    return handle


def taught_lookup(agent, question, query, *, answer_variable='answer'):
    from tensorcode.learning.store_query import StoreQueryPlan
    teach_store_query(agent, question, StoreQueryPlan(query, answer_variable, (query.scope,)))
    return agent.lookup(question, interpretation_dependency=selected_question_dependency(agent, question))
