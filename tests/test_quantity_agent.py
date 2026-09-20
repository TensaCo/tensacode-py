"""Explicit evidence, arithmetic selection, and independently taught language wiring.

Arithmetic tests supply occurrence identities, source evidence, ordered operands,
operation and output context. English tests additionally teach and select speech
and informing correspondences; none of those labels are production defaults.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_test_support import selected_agent as Agent
from tensorcode.agent.operations import MODEL
from tensorcode.agent.quantity_plugin import QuantityPlugin
from tensorcode.agent.understand import Act, Sentence
from tensorcode.language import Entity, Frame, Question, verbnet, wordnet
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit, convert
from tensorcode.records import Claim, Evidence, Ref, Store

pytestmark = pytest.mark.skipif(wordnet.find_wordnet() is None or verbnet.find_verbnet() is None,
                                reason="needs WordNet and VerbNet data on disk")

needs_parser = pytest.mark.skipif(not MODEL.exists(), reason="needs the treebank parser (eval/parsing/train_ud.py)")

SHONDRA = Ref("entity:Shondra")
TONI = Ref("entity:Toni")
THEM = Ref("entity:they")
REPORT = Ref("fixture:report")


def grounded_subject_turn(agent, text, reference, *, expected_question=None, informing_capability=None):
    """The fixture supplies identity and optional meaning, not inferred intent."""
    from tensorcode.agent.core import InterpretationDecision
    from tensorcode.agent.grounding import MentionBinding, propose_grounding

    evidence = agent.interpretations.add_source(
        "Quantity test supplies this subject identity", provider="test-fixture")

    def select(group):
        for _ in range(32):
            if not agent.interpretations.continuation_status(group.id).pending:
                break
            agent.expand_interpretation(group.id, max_expansions=1000000, max_candidates=256)
        assert not agent.interpretations.continuation_status(group.id).pending
        group = agent.interpretations.get(group.id)
        parent = group.candidates[0]
        if expected_question is not None:
            predicate, asked, subject_text = expected_question
            parents = [candidate for candidate in group.candidates
                       if len(candidate.payload.acts) == 1
                       and isinstance(candidate.payload.acts[0].meaning, Question)
                       and candidate.payload.acts[0].meaning.asked == asked
                       and candidate.payload.acts[0].frame.predicate == predicate
                       and isinstance(candidate.payload.acts[0].frame.roles.get("subject"), Entity)
                       and candidate.payload.acts[0].frame.roles["subject"].text == subject_text]
            assert parents, "learned alternatives must include the fixture's explicitly supplied meaning"
            parent = parents[0]
        candidate = propose_grounding(agent.interpretations, group.id, parent.id, [
            MentionBinding(("acts", 0, "frame", "roles", "subject"), reference,
                           (evidence.id,), "Authored quantity fixture subject binding")
        ])
        if informing_capability is not None:
            from informing_fixtures import teach_informing
            from tensorcode.learning.informing import InformingPlan
            from tensorcode.records import Proposition, Var, Interval
            provider, = agent.plugins
            cap, = [cap for cap in provider.capabilities() if cap.name == informing_capability]
            param, = cap.params
            question = candidate.payload.acts[0].meaning
            teach_informing(agent, question, InformingPlan(provider.name, cap.name,
                ((param.name, reference),), Proposition(cap.informs[0].query.predicate,
                    {'subject': reference, 'object': Var('answer')}), 'answer'))
        compared = agent.interpretations.get(group.id)
        return InterpretationDecision(candidate.id, "Fixture supplies grounded reading", (evidence.id,),
            compared_revision=compared.revision,
            compared_candidate_ids=tuple(item.id for item in compared.candidates))

    agent.interpretation_selector = select
    return agent.turn(text)


def plants(n: float) -> Quantity:
    return Quantity(n, Unit.of("plant"))


from quantity_fixtures import measurement, calculation, conversion_definition
from tensorcode.quantity_calculations import CalculationContext
from tensorcode.derivations import DerivationReference, import_derivation, validate_record_support
from tensorcode.records import Proposition, Var, Interval


def test_an_amount_can_be_what_a_claim_is_about():
    """The agent's retrieval puts a claim's object in a set, and a dict is not hashable.

    Before ``Unit.__hash__``, ``Agent._from_claims`` raised ``TypeError: unhashable type:
    'dict'`` the moment any claim in the store carried a quantity — so an amount could be
    recorded and never read back.
    """
    store = Store()
    store.tell(Claim(SHONDRA, "fixture:possession", plants(7)), Evidence(source=Ref("test:quantity"), observed_at=datetime.now(timezone.utc)))
    (record,) = store.claims(subject=SHONDRA, predicate="fixture:possession")
    assert {record.claim.subject, record.claim.object} == {SHONDRA, plants(7)}
    assert hash(plants(7)) == hash(Quantity(7, Unit.of("plant")))
    assert Unit.of("plant") != Unit.of("plants")  # Literal identifiers have no plural alias.


@pytest.mark.parametrize('amount,source,target,factor,expected', [
    (45, 'minute', 'second', 60, 2700), (3, 'foot', 'inch', 12, 36),
])
def test_an_amount_is_converted_only_with_selected_supported_definition(amount, source, target, factor, expected):
    plugin = QuantityPlugin()
    unit, target_unit = Unit.of(source), Unit.of(target)
    item = measurement(plugin, Ref('measurement:conversion-input'), SHONDRA, 'distance-or-time', Quantity(amount, unit))
    definition = conversion_definition(plugin, Ref('definition:explicit-unit-rate'), unit, target_unit, factor,
        scope=None, valid=Interval())
    assert isinstance(convert(Quantity(amount, unit), target_unit), Unknown)
    assert convert(Quantity(amount, unit), unit) == Quantity(amount, unit)
    chosen = calculation(plugin, 'convert', (item, definition), CalculationContext(SHONDRA, 'converted'),
        params={'scope': None, 'valid': Interval()})
    result = plugin.calculate(chosen)
    assert not isinstance(result, Unknown), result
    assert result.proposition.role('object') == Quantity(expected, target_unit)
    assert definition.id in result.premise_ids


def test_converting_across_dimensions_refuses_rather_than_scaling():
    got = convert(Quantity(3, Unit.of("foot")), Unit.of("coin"))
    assert isinstance(got, Unknown) and got.reason == "dimension_mismatch"


@needs_parser
def test_a_question_in_english_reaches_the_plugin():
    from tensorcode.agent.understand import LearnedReader

    plugin = QuantityPlugin()
    item = measurement(plugin, Ref('measurement:english-amount'), SHONDRA, 'have', plants(7))
    calculation(plugin, 'sum', (item,), CalculationContext(SHONDRA, 'have'))
    agent = Agent([plugin], reader=LearnedReader())
    from dependency_meaning_fixtures import teach_speech_family
    from tensorcode.learning.speech_act import SpeechActLabel
    teach_speech_family(agent, 'how much does Shondra have?',
        SpeechActLabel('question', 'quantity', ('roles', 'manner'), (1,)),
        matches=lambda neutral: neutral.frame.predicate == 'have'
        and neutral.frame.roles.get('manner') == 'much'
        and isinstance(neutral.frame.roles.get('subject'), Entity)
        and neutral.frame.roles['subject'].text == 'Shondra')
    turn = grounded_subject_turn(agent,
                                 "how much does Shondra have?", SHONDRA,
                                 expected_question=("have", "quantity", "Shondra"),
                                 informing_capability='calculate_sum_owner_have')
    assert "7" in turn.reply
    assert [o.status for o in turn.outcomes] == ["answered"]


@needs_parser
def test_taught_count_question_preserves_unbound_counted_noun_instead_of_broadening():
    from tensorcode.agent.understand import LearnedReader
    from dependency_meaning_fixtures import teach_speech_family
    from tensorcode.learning.speech_act import SpeechActLabel
    plugin = QuantityPlugin()
    measurement(plugin, Ref('measurement:unbound-count'), SHONDRA, 'have', plants(7))
    agent = Agent([plugin], reader=LearnedReader())
    teach_speech_family(agent, 'how many plants does Shondra have?',
        SpeechActLabel('question', 'quantity', (), (0, 1, 2, 3, 4, 5, 6)),
        matches=lambda neutral: neutral.frame.predicate == 'have'
        and isinstance(neutral.frame.roles.get('subject'), Entity)
        and neutral.frame.roles['subject'].text == 'Shondra')
    turn = grounded_subject_turn(agent, 'how many plants does Shondra have?', SHONDRA,
                                 expected_question=('have', 'quantity', 'Shondra'))
    assert [outcome.status for outcome in turn.outcomes] == ['unknown']
    assert turn.outcomes[0].reason == 'stated question roles require explicit grounding'
    counted = turn.outcomes[0].act.frame.roles['object']
    assert counted.text == 'how many plants' and counted.features['noun'] == 'plant'
    assert counted.ref is None and turn.outcomes[0].receipt is None


@needs_parser
def test_a_word_problem_is_abstained_on_rather_than_guessed_at():
    """Raw prose cannot authorize measurements, operands, or arithmetic choices."""
    from tensorcode.agent.understand import LearnedReader

    turn = Agent([QuantityPlugin()], reader=LearnedReader()).turn(
        "Shondra has 7 fewer plants than Toni. Toni has 60% more plants than Frederick. "
        "If Frederick has 10 plants, how many plants does Shondra have?")
    assert not any(o.status in ("answered", "done") for o in turn.outcomes)




def test_measurements_do_not_select_operations_or_publish_capabilities():
    plugin = QuantityPlugin()
    item = measurement(plugin, Ref('measurement:spend'), SHONDRA, 'spend', Quantity(8, Unit.of('dollar')))
    assert plugin.capabilities() == ()
    chosen = calculation(plugin, 'sum', (item,), CalculationContext(SHONDRA, 'spend'), select=False)
    assert plugin.capabilities() == ()
    assert plugin.select_calculation(chosen, reason='Explicit sum authorization') is True
    cap, = plugin.capabilities()
    assert cap.name == 'calculate_sum_owner_spend'
    assert cap.effect_kind == 'read' and not cap.effects
    assert cap.informs[0].query == Proposition('calculated:sum:spend',
        {'subject': Var('owner'), 'object': Var('answer')})
    assert plugin.calculate(chosen).proposition.predicate == 'calculated:sum:spend'
    for retired in ('observe', 'total', 'total_of_kind', '_total_claim', 'count_properties'):
        assert not hasattr(plugin, retired)


def test_two_independent_equal_measurements_count_twice_but_repeated_identity_does_not():
    plugin = QuantityPlugin()
    first = measurement(plugin, Ref('measurement:first-three'), SHONDRA, 'have', plants(3))
    second = measurement(plugin, Ref('measurement:second-three'), SHONDRA, 'have', plants(3))
    chosen = calculation(plugin, 'sum', (first, second), CalculationContext(SHONDRA, 'have'))
    result = plugin.calculate(chosen)
    assert not isinstance(result, Unknown), result
    assert result.proposition.role('object') == plants(6)
    assert set(result.premise_ids) >= {first.id, second.id}
    assert validate_record_support(plugin.mind, result.record_id) is True
    repeated = measurement(plugin, Ref('measurement:first-three'), SHONDRA, 'have', plants(3))
    assert repeated.id == first.id
    with pytest.raises(ValueError):
        calculation(plugin, 'sum', (first, repeated), CalculationContext(SHONDRA, 'have'))


def test_same_measurement_identity_with_conflicting_values_invalidates_calculation():
    plugin = QuantityPlugin()
    first = measurement(plugin, Ref('measurement:one-occurrence'), SHONDRA, 'have', plants(3))
    chosen = calculation(plugin, 'sum', (first,), CalculationContext(SHONDRA, 'have'))
    before = plugin.calculate(chosen)
    assert not isinstance(before, Unknown), before
    measurement(plugin, Ref('measurement:one-occurrence'), SHONDRA, 'have', plants(4))
    assert isinstance(plugin.calculate(chosen), Unknown)
    assert isinstance(validate_record_support(plugin.mind, before.record_id), Unknown)


def test_explicit_sum_excludes_unselected_owners_and_kinds():
    plugin = QuantityPlugin()
    plant, coin = Ref('kind:plant'), Ref('kind:coin')
    first = measurement(plugin, Ref('measurement:plants-a'), SHONDRA, 'have', Quantity(3, Unit.of('item')), kind=plant)
    second = measurement(plugin, Ref('measurement:plants-b'), SHONDRA, 'have', Quantity(4, Unit.of('item')), kind=plant)
    measurement(plugin, Ref('measurement:coins'), SHONDRA, 'have', Quantity(20, Unit.of('item')), kind=coin)
    measurement(plugin, Ref('measurement:toni'), TONI, 'have', Quantity(100, Unit.of('item')), kind=plant)
    chosen = calculation(plugin, 'sum', (first, second), CalculationContext(SHONDRA, 'have', plant))
    result = plugin.calculate(chosen)
    assert not isinstance(result, Unknown), result
    assert result.proposition == Proposition('calculated:sum:have',
        {'subject': SHONDRA, 'kind': plant, 'object': Quantity(7, Unit.of('item'))})


def test_units_do_not_choose_rate_multiplication():
    plugin = QuantityPlugin()
    rate = measurement(plugin, Ref('measurement:ride-rate'), THEM, 'cost', Quantity(6, Unit.of('ticket') / Unit.of('ride')))
    count = measurement(plugin, Ref('measurement:ride-count'), THEM, 'rode', Quantity(10, Unit.of('ride')))
    assert plugin.capabilities() == ()
    summed = calculation(plugin, 'sum', (rate, count), CalculationContext(THEM, 'tickets'))
    assert isinstance(plugin.calculate(summed), Unknown)
    product = calculation(plugin, 'mul', (rate, count), CalculationContext(THEM, 'tickets'))
    result = plugin.calculate(product)
    assert not isinstance(result, Unknown), result
    assert result.proposition.role('object') == Quantity(60, Unit.of('ticket'))
    incomplete = calculation(plugin, 'mul', (rate,), CalculationContext(THEM, 'tickets'))
    assert isinstance(plugin.calculate(incomplete), Unknown)


def test_ordered_subtraction_and_comparison_use_explicit_supported_conversion():
    plugin = QuantityPlugin()
    feet = measurement(plugin, Ref('measurement:feet'), SHONDRA, 'walk', Quantity(3, Unit.of('foot')))
    metres = measurement(plugin, Ref('measurement:metres'), TONI, 'walk', Quantity(2, Unit.of('metre')))
    definition = conversion_definition(plugin, Ref('definition:foot-to-metre'), Unit.of('foot'), Unit.of('metre'),
        0.3048, scope=None, valid=Interval())
    conversion = calculation(plugin, 'convert', (feet, definition), CalculationContext(SHONDRA, 'metres'),
        params={'scope': None, 'valid': Interval()})
    converted = plugin.calculate(conversion)
    assert not isinstance(converted, Unknown), converted
    context = CalculationContext(TONI, 'difference')
    subtraction = calculation(plugin, 'sub', (metres, converted.proposition), context)
    result = plugin.calculate(subtraction)
    assert not isinstance(result, Unknown), result
    assert result.proposition.role('object').value == pytest.approx(2 - 3 * 0.3048)
    comparison = calculation(plugin, 'compare', (metres, converted.proposition), context)
    assert plugin.calculate(comparison).proposition.role('object') == 'greater'
    reversed_order = calculation(plugin, 'sub', (converted.proposition, metres), context)
    assert plugin.calculate(reversed_order).proposition.role('object').value == pytest.approx(3 * 0.3048 - 2)
    plugin.mind.supersede(definition, why='withdraw supplied conversion evidence')
    assert isinstance(plugin.calculate(reversed_order), Unknown)


def test_explicit_conversion_can_use_an_authenticated_sum_as_operand():
    plugin = QuantityPlugin()
    hour = measurement(plugin, Ref('measurement:hour'), SHONDRA, 'run', Quantity(1, Unit.of('hour')))
    half_hour = measurement(plugin, Ref('measurement:half-hour'), SHONDRA, 'run', Quantity(0.5, Unit.of('hour')))
    total = plugin.calculate(calculation(plugin, 'sum', (hour, half_hour), CalculationContext(SHONDRA, 'duration')))
    assert not isinstance(total, Unknown), total
    definition = conversion_definition(plugin, Ref('definition:hour-to-minute'), Unit.of('hour'), Unit.of('minute'),
        60, scope=None, valid=Interval())
    chosen = calculation(plugin, 'convert', (total.proposition, definition), CalculationContext(SHONDRA, 'minutes'),
        params={'scope': None, 'valid': Interval()})
    converted = plugin.calculate(chosen)
    assert not isinstance(converted, Unknown), converted
    assert converted.proposition.role('object') == Quantity(90, Unit.of('minute'))
    plugin.mind.supersede(hour, why='withdraw original measurement')
    assert isinstance(validate_record_support(plugin.mind, converted.record_id), Unknown)


@pytest.mark.parametrize('operation', ['sum', 'sub', 'compare'])
def test_incompatible_dimensions_refuse_even_with_explicit_operation(operation):
    plugin = QuantityPlugin()
    apple = measurement(plugin, Ref('measurement:apple'), SHONDRA, 'have', Quantity(3, Unit.of('apple')))
    coin = measurement(plugin, Ref('measurement:coin'), SHONDRA, 'have', Quantity(4, Unit.of('coin')))
    chosen = calculation(plugin, operation, (apple, coin), CalculationContext(SHONDRA, 'have'))
    assert isinstance(plugin.calculate(chosen), Unknown)


@pytest.mark.parametrize('qualification', ['extra_role', 'negative', 'scope', 'valid', 'modality'])
def test_explicit_numeric_operands_reject_unsupported_qualifications(qualification):
    from dataclasses import replace
    from tensorcode.records import Interval
    plugin = QuantityPlugin()
    original = measurement(plugin, Ref('measurement:qualified'), SHONDRA, 'have', plants(3))
    changes = {'extra_role': {'roles': {**original.roles, 'condition': 'only sometimes'}},
        'negative': {'polarity': False}, 'scope': {'scope': Ref('scope:hypothesis')},
        'valid': {'valid': Interval(datetime.now(timezone.utc), None)}, 'modality': {'modality': 'possible'}}
    qualified = replace(original, **changes[qualification])
    plugin.mind.supersede(original, why='fixture replaces measurement with qualified evidence')
    plugin.mind.assert_(qualified, Evidence(Ref('fixture:qualified-observer'), datetime.now(timezone.utc)))
    chosen = calculation(plugin, 'sum', (qualified,), CalculationContext(SHONDRA, 'have'))
    assert isinstance(plugin.calculate(chosen), Unknown)


def test_count_selected_records_is_explicit_not_a_property_intent_classifier():
    plugin = QuantityPlugin()
    records = (measurement(plugin, Ref('measurement:selected-first'), REPORT, 'have', plants(3)),
               measurement(plugin, Ref('measurement:selected-second'), REPORT, 'have', plants(4)))
    assert plugin.capabilities() == ()
    chosen = calculation(plugin, 'count_selected', records, CalculationContext(REPORT, 'selected-records'))
    result = plugin.calculate(chosen)
    assert not isinstance(result, Unknown), result
    assert result.proposition.role('object') == Quantity(2, Unit.of('record'))
    empty = calculation(plugin, 'count_selected', (), CalculationContext(REPORT, 'selected-records'))
    assert plugin.calculate(empty).proposition.role('object') == Quantity(0, Unit.of('record'))


def test_capability_call_requires_exact_selected_context_and_authenticated_reveal():
    from tensorcode.agent.plugin import Call
    plugin = QuantityPlugin()
    kind = Ref('kind:plant')
    item = measurement(plugin, Ref('measurement:capability'), SHONDRA, 'have', plants(7), kind=kind)
    calculation(plugin, 'sum', (item,), CalculationContext(SHONDRA, 'have', kind))
    cap, = plugin.capabilities()
    assert cap.name == 'calculate_sum_kind_have'
    assert cap.informs[0].query == Proposition('calculated:sum:have',
        {'subject': Var('owner'), 'kind': Var('kind'), 'object': Var('answer')})
    action = Call(plugin.name, cap.name, (('owner', SHONDRA), ('kind', kind)))
    receipt = plugin.execute(action)
    assert receipt.status == 'applied'
    reference, = plugin.reveal(cap, dict(action.args), receipt)
    assert type(reference) is DerivationReference
    target = Store()
    imported = import_derivation(target, reference)
    assert not isinstance(imported, Unknown), imported
    assert validate_record_support(target, imported.id) is True
    assert list(plugin.reveal(cap, {'owner': TONI, 'kind': kind}, receipt)) == []
    for invalid in (Call('foreign', cap.name, action.args),
                    Call(plugin.name, cap.name, (('owner', SHONDRA),)),
                    Call(plugin.name, cap.name, (('owner', SHONDRA), ('kind', 'plant'))),
                    Call(plugin.name, cap.name, (*action.args, ('kind', kind))),
                    Call(plugin.name, cap.name, (*action.args, ('extra', kind)))):
        assert plugin.execute(invalid).status == 'rejected'
