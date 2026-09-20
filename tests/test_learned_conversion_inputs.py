"""Real parsed input learns an informing correspondence to an explicit conversion.

Identity bindings, measurement meanings, rates, operation/operand choices and
selection policies are supplied. The learned correspondence transfers full
questions across independently supplied owner references; it does not discover
conversion rates or infer that bundles should be converted to plants.
"""
from datetime import datetime, timezone

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.agent.informing_learning import (
    retain_informing_example, fit_informing_model, admit_informing_model,
)
from tensorcode.derivations import validate_record_support
from tensorcode.language import Question
from tensorcode.learning.informing import InformingPlan
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.records import Evidence, Interval, Proposition, Ref, Var

from test_learned_informing_inputs import (
    actual_reader, teach_actual_speech, ground_question, question_text, proper_question,
)


def supplied_conversion(plugin, owner, kind, identity):
    evidence = Evidence(Ref('fixture:explicit-conversion-teaching'),
        datetime(2026, 9, 20, tzinfo=timezone.utc), locator=identity)
    measurement = plugin.remember(owner, 'have', Quantity(2, Unit.of('plant_bundle')),
        kind=kind, measurement=Ref('measurement:' + identity), evidence=evidence)
    definition = plugin.remember_conversion(Ref('definition:' + identity),
        Unit.of('plant_bundle'), Unit.of('plant'), 4,
        scope=None, valid=Interval(), evidence=evidence)
    calculation = plugin.register_calculation('convert', (measurement.id, definition.id),
        context=CalculationContext(owner, 'have', kind), params={'scope': None, 'valid': Interval()},
        basis=('Fixture explicitly chooses measurement, directed rate and conversion operation',))
    assert plugin.select_calculation(calculation, reason='Explicit conversion selection, not inferred from language') is True
    return definition


def test_real_question_learns_explicit_conversion_and_cannot_reuse_invalidated_rate(actual_reader):
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
    plant = Ref('kind:plant')
    teaching = []
    for context in ('Alpha', 'Beta', 'Gamma'):
        owner = Ref('owner:conversion-' + context)
        supplied_conversion(plugin, owner, plant, 'conversion-' + context)
        message = agent.interpret(context + '. ' + question_text())
        groups = [agent.interpretations.get(gid) for gid in message.group_ids]
        matching = [group for group in groups if any(
            isinstance(act.meaning, Question) and proper_question(act.frame, 'Shondra')
            for candidate in group.candidates for act in candidate.payload.acts)]
        assert len(matching) == 1
        group = matching[0]
        grounded = ground_question(agent, group, owner, plant)
        plan = InformingPlan('quantity', 'calculate_convert_kind_have',
            (('owner', owner), ('kind', plant)),
            Proposition('calculated:convert:have',
                {'subject': owner, 'kind': plant, 'object': Var('answer')}), 'answer')
        example = retain_informing_example(agent, group.id, grounded.id, 0, plan,
            basis=('Explicit full-question correspondence to selected evidence-backed conversion',))
        assert not isinstance(example, Unknown), example
        teaching.append(example)
    model = fit_informing_model(agent, teaching[:2], teaching[2:])
    assert not isinstance(model, Unknown), model
    model = admit_informing_model(agent, model, reason='Explicit admission after independent Gamma validation')
    assert not isinstance(model, Unknown), model
    agent.informing_model = model

    fresh_owner = Ref('owner:fresh-conversion-execution')
    definition = supplied_conversion(plugin, fresh_owner, plant, 'fresh-conversion')

    def select_reading(group):
        grounded = ground_question(agent, group, fresh_owner, plant)
        current = agent.interpretations.get(group.id)
        return InterpretationDecision(grounded.id, 'Explicit fresh-context owner/kind grounding',
            compared_revision=current.revision,
            compared_candidate_ids=tuple(candidate.id for candidate in current.candidates))

    def select_plan(group):
        assert len(group.candidates) == 1
        return InterpretationDecision(group.candidates[0].id, 'Explicit learned informing-plan selection',
            compared_revision=group.revision,
            compared_candidate_ids=tuple(candidate.id for candidate in group.candidates))

    agent.interpretation_selector = select_reading
    agent.informing_selector = select_plan
    turn = agent.turn(question_text())
    outcome, = turn.outcomes
    assert outcome.status == 'answered', (outcome.reason, outcome.verified, outcome.receipt)
    assert outcome.answer == [Quantity(8, Unit.of('plant'))]
    assert outcome.receipt.status == 'applied' and len(plugin.calls) == 1
    assert outcome.act.frame.roles['subject'].ref == fresh_owner
    assert outcome.act.frame.roles['object'].ref == plant
    assert outcome.act.frame.roles['object'].text == 'how many plants'
    assert outcome.act.frame.roles['object'].features['noun'] == 'plant'
    assert model.dependency in outcome.plan.dependencies
    retained, = agent.store.propositions('calculated:convert:have')
    assert retained.evidence[0].method == 'authenticated-derivation-import'
    assert validate_record_support(agent.store, retained.id) is True
    assert not agent.store.propositions('have')

    rival = plugin.remember_conversion(Ref('definition:contradictory-fresh-rate'),
        Unit.of('plant_bundle'), Unit.of('plant'), 5,
        scope=None, valid=Interval(), evidence=Evidence(Ref('fixture:contradictory-rate'),
            datetime(2026, 9, 20, tzinfo=timezone.utc)))
    assert isinstance(validate_record_support(agent.store, retained.id), Unknown)
    conflicted = agent.turn(question_text())
    assert len(conflicted.outcomes) == 1
    assert conflicted.outcomes[0].status == 'unknown'
    assert conflicted.outcomes[0].answer is None
    assert conflicted.outcomes[0].receipt.status == 'rejected'
    assert len(plugin.calls) == 2

    plugin.mind.supersede(rival, 'withdraw conflicting rate')
    plugin.mind.supersede(definition, 'withdraw selected definition')
    assert isinstance(validate_record_support(agent.store, retained.id), Unknown)
    withdrawn = agent.turn(question_text())
    assert len(withdrawn.outcomes) == 1
    assert withdrawn.outcomes[0].status == 'unknown'
    assert withdrawn.outcomes[0].answer is None
    assert withdrawn.outcomes[0].receipt.status == 'rejected'
    assert len(plugin.calls) == 3
