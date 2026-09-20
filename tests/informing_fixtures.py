"""Explicit question/measurement teaching for downstream mechanism tests.

These authored structured labels do not claim natural-language understanding.
Reference identities vary across independent retained teaching contexts; every
non-reference part of the supplied question and plan is preserved.
"""
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace

from tensorcode.agent import InterpretationDecision
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref


def teach_informing(agent, question, plan, *, select=True):
    from tensorcode.agent.informing_learning import (
        retain_informing_example, fit_informing_model, admit_informing_model,
    )
    records = []
    for index in range(3):
        references = {}
        def variant(value):
            if type(value) is Ref:
                if value not in references:
                    references[value] = Ref(f'fixture-informing:{index}:{len(references)}')
                return references[value]
            if type(value) is dict:
                return {key: variant(item) for key, item in value.items()}
            if type(value) in (tuple, list):
                return type(value)(variant(item) for item in value)
            if is_dataclass(value):
                return replace(value, **{field.name: variant(getattr(value, field.name))
                                         for field in fields(value) if field.init})
            return deepcopy(value)
        taught_question, taught_plan = variant(question), variant(plan)
        source = agent.interpretations.add_source(f'Explicit informing teaching context {index}',
                                                  provider='authored-informing-fixture')
        group = agent.interpretations.create_group(source.id)
        candidate = agent.interpretations.propose(group.id, SentenceAlternative(None,
            (Act('question', taught_question, taught_question.frame),), provenance='authored-informing-fixture'))
        record = retain_informing_example(agent, group.id, candidate.id, 0, taught_plan,
            basis=('explicit full-question to exact measurement plan fixture',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    model = fit_informing_model(agent, records[:2], records[2:])
    assert not isinstance(model, Unknown), model
    handle = admit_informing_model(agent, model, reason='explicit fixture measurement model admission')
    assert not isinstance(handle, Unknown), handle
    agent.informing_model = handle
    if select:
        def choose(group):
            choices = [candidate for candidate in group.candidates if hasattr(candidate.payload, 'plan')]
            return InterpretationDecision(choices[0].id if len(choices) == 1 else None,
                'fixture explicitly authorizes a sole complete taught informing plan',
                compared_revision=group.revision,
                compared_candidate_ids=tuple(candidate.id for candidate in group.candidates))
        agent.informing_selector = choose
    return handle


def selected_question_dependency(agent, question):
    """Retain the explicit structured question selection used by a mechanism test."""
    from tensorcode.agent.task_dependencies import capture_dependency
    source = agent.interpretations.add_source('Explicit runtime question', provider='authored-informing-fixture')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, SentenceAlternative(None,
        (Act('question', question, question.frame),)))
    agent.interpretations.select(group.id, candidate.id, reason='explicit fixture question selection')
    return capture_dependency(agent.interpretations, group.id, basis=('explicit fixture selection',))
