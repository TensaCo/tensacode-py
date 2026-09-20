"""Quantity, reachable from the agent: what it answers and — mostly — what it refuses.

Every test here pins one of two things: a number the agent can now say, or a number it
must not say. The second kind outnumbers the first on purpose. The plugin exists because
"how many"/"how much" reached nothing at all, but the way it could fail is worse than the
way it was failing: a total summed across two kinds of thing, or a property count returned
to a question about plants, looks exactly like an answer.

They need WordNet and VerbNet on disk (the agent's taxonomy and goals) and skip without
them, like ``test_general_agent.py``. The ones that read English also need the treebank
parser, because the hand grammar does not produce ``asked == "quantity"`` at all; the rest
build the question the parser would have built, so the wiring is tested either way.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_test_support import selected_agent as Agent
from tensorcode.agent.operations import MODEL
from tensorcode.agent.quantity_plugin import POSSESSION, QuantityPlugin, world_predicate
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
            from tensorcode.records import Proposition, Var
            provider, = agent.plugins
            cap, = [cap for cap in provider.capabilities() if cap.name == informing_capability]
            param, = cap.params
            question = candidate.payload.acts[0].meaning
            teach_informing(agent, question, InformingPlan(provider.name, cap.name,
                ((param.name, reference),), Proposition(question.frame.predicate,
                    {'subject': reference, 'object': Var('answer')}), 'answer'))
        compared = agent.interpretations.get(group.id)
        return InterpretationDecision(candidate.id, "Fixture supplies grounded reading", (evidence.id,),
            compared_revision=compared.revision,
            compared_candidate_ids=tuple(item.id for item in compared.candidates))

    agent.interpretation_selector = select
    return agent.turn(text)


def plants(n: float) -> Quantity:
    return Quantity(n, Unit.of("plant"))


class AmountOnlyFixture(QuantityPlugin):
    """Authored choice isolates arithmetic; production never prefers this implicitly."""

    def capabilities(self):
        return tuple(cap for cap in super().capabilities() if cap.name != "count_properties")


def asking(predicate: str, subject: Entity) -> tuple[Sentence, Act]:
    """An explicitly supplied owner-only quantity question, not a parser claim."""
    question = Question(Frame(predicate, {"subject": subject}, {"mood": "interrogative"}), "quantity")
    act = Act("question", question, question.frame)
    return Sentence("how many …?", ("how", "many"), None, (act,)), act


def answer(agent: Agent, predicate: str, subject: Entity, *, capability=None):
    from tensorcode.agent.understand import SentenceAlternative
    from tensorcode.agent.task_dependencies import capture_dependency
    from tensorcode.learning.informing import InformingPlan
    from tensorcode.records import Proposition, Var
    from informing_fixtures import teach_informing
    sentence, act = asking(predicate, subject)
    if capability is not None:
        provider, = agent.plugins
        cap, = [cap for cap in provider.capabilities() if cap.name == capability]
        param, = cap.params
        plan = InformingPlan(provider.name, cap.name, ((param.name, subject.ref),),
            Proposition(predicate, {'subject': subject.ref, 'object': Var('answer')}), 'answer')
        teach_informing(agent, act.meaning, plan)
    source = agent.interpretations.add_source(sentence.text, provider='authored quantity question fixture')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, SentenceAlternative(None, (act,)))
    agent.interpretations.select(group.id, candidate.id, reason='explicit supplied structured question')
    dependency = capture_dependency(agent.interpretations, group.id, basis=('authored question choice',))
    return agent.handle(sentence, act, [], requests_in_message=0, interpretation_dependency=dependency)


# ------------------------------------------------------- the quantity module itself


def test_an_amount_can_be_what_a_claim_is_about():
    """The agent's retrieval puts a claim's object in a set, and a dict is not hashable.

    Before ``Unit.__hash__``, ``Agent._from_claims`` raised ``TypeError: unhashable type:
    'dict'`` the moment any claim in the store carried a quantity — so an amount could be
    recorded and never read back.
    """
    store = Store()
    store.tell(Claim(SHONDRA, POSSESSION, plants(7)), Evidence(source=Ref("test:quantity"), observed_at=datetime.now(timezone.utc)))
    (record,) = store.claims(subject=SHONDRA, predicate=POSSESSION)
    assert {record.claim.subject, record.claim.object} == {SHONDRA, plants(7)}
    assert hash(plants(7)) == hash(Quantity(7, Unit.of("plants")))  # the plural is the same unit


def test_an_amount_is_said_in_the_unit_that_was_asked_for():
    assert convert(Quantity(45, Unit.of("minute")), Unit.of("second")) == Quantity(2700, Unit.of("second"))
    assert convert(Quantity(3, Unit.of("foot")), Unit.of("inch")).value == pytest.approx(36)


def test_converting_across_dimensions_refuses_rather_than_scaling():
    got = convert(Quantity(3, Unit.of("foot")), Unit.of("coin"))
    assert isinstance(got, Unknown) and got.reason == "dimension_mismatch"


# ------------------------------------------------------------------ the vocabulary


def test_the_capabilities_are_read_off_what_it_holds():
    """Nothing here lists a verb: a host that records spending gets a capability that
    answers questions about spending."""
    plugin = QuantityPlugin()
    assert [c.name for c in plugin.capabilities()] == ["count_properties"]

    plugin.remember(SHONDRA, "spend", Quantity(8, Unit.of("dollar")))
    names = [c.name for c in plugin.capabilities()]
    assert names == ["amount_of_spend", "count_properties"]
    (informs,) = plugin.capabilities()[0].informs
    assert (informs.pred, informs.role) == ("spend", "undergoer")
    assert all(c.effect_kind == "read" and not c.effects for c in plugin.capabilities())


def test_predicates_are_preserved_without_a_lexical_alias():
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", plants(7))
    assert world_predicate("have") == "have"
    assert [c.name for c in plugin.capabilities()] == ["amount_of_have", "count_properties"]
    assert isinstance(plugin.total(SHONDRA, POSSESSION), Unknown)


# ------------------------------------------------------------------ arithmetic


def test_two_amounts_of_one_kind_are_added():
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, POSSESSION, plants(3))
    plugin.remember(SHONDRA, POSSESSION, plants(4))
    assert plugin.total(SHONDRA, POSSESSION) == plants(7)


def test_amounts_of_two_kinds_are_refused_because_the_question_lost_which_one():
    """"how many apples do I have?" and "how many pears do I have?" are the same question
    by the time a plugin sees it. Adding them would answer both wrongly."""
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, POSSESSION, Quantity(3, Unit.of("apple")))
    plugin.remember(SHONDRA, POSSESSION, Quantity(4, Unit.of("pear")))
    got = plugin.total(SHONDRA, POSSESSION)
    assert isinstance(got, Unknown) and got.reason == "several_dimensions"


def test_a_rate_times_a_count_is_a_product_because_the_units_cancel():
    """"each ride cost 6 tickets" and "they rode 10 times" is 60 tickets — a multiplication
    nothing here decided on, because ``ride`` appears once above and once below the line."""
    plugin = QuantityPlugin()
    plugin.remember(THEM, "use", Quantity(6, Unit.of("ticket") / Unit.of("ride")))
    plugin.remember(THEM, "use", Quantity(10, Unit.of("ride")))
    assert plugin.total(THEM, "use") == Quantity(60, Unit.of("ticket"))


def test_a_rate_with_nothing_to_multiply_is_refused():
    plugin = QuantityPlugin()
    plugin.remember(THEM, "use", Quantity(6, Unit.of("ticket") / Unit.of("ride")))
    got = plugin.total(THEM, "use")
    assert isinstance(got, Unknown) and got.reason == "dangling_rate"


def test_a_difference_is_taken_across_two_spellings_of_one_dimension():
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "walk", Quantity(3, Unit.of("foot")))
    plugin.remember(TONI, "walk", Quantity(2, Unit.of("metre")))
    got = plugin.difference(TONI, SHONDRA, "walk")
    assert isinstance(got, Quantity) and got.base() == pytest.approx(2 - 3 * 0.3048)
    assert plugin.compare(TONI, SHONDRA, "walk") == "greater"


def test_comparing_across_dimensions_refuses():
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, POSSESSION, plants(3))
    plugin.remember(TONI, POSSESSION, Quantity(3, Unit.of("coin")))
    got = plugin.compare(TONI, SHONDRA, POSSESSION)
    assert isinstance(got, Unknown) and got.reason == "dimension_mismatch"


def test_a_total_is_reported_in_the_unit_that_was_asked_for():
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "run", Quantity(1, Unit.of("hour")))
    plugin.remember(SHONDRA, "run", Quantity(30, Unit.of("minute")))
    assert plugin.in_unit(SHONDRA, "run", Unit.of("minute")) == Quantity(90, Unit.of("minute"))


def test_the_working_is_on_the_record_and_names_its_premises():
    """``quantity.derive`` writes the result with the premise claim ids, so a retracted
    premise withdraws the conclusion and ``explain`` can show the arithmetic."""
    plugin = QuantityPlugin()
    first = plugin.remember(SHONDRA, POSSESSION, plants(3))
    second = plugin.remember(SHONDRA, POSSESSION, plants(4))
    plugin.total(SHONDRA, POSSESSION)
    (derived,) = plugin.mind.claims(subject=SHONDRA, predicate=f"total:{POSSESSION}")
    assert derived.claim.object == plants(7)
    (evidence,) = derived.evidence
    assert evidence.method == "arithmetic:sum"
    assert set(evidence.derived_from) == {first.id, second.id}


# ------------------------------------------------------------------ reading statements


def test_a_statement_s_numbers_are_kept_with_the_thing_they_were_said_of():
    frame = Frame(POSSESSION, {"subject": Entity("name", "Shondra", ref=SHONDRA),
                           "object": Entity("description", "7 plants",
                                            {"count": "7", "noun": "plant", "number": "plural"})})
    plugin = QuantityPlugin()
    (claim,) = plugin.observe(frame)
    assert (claim.subject, claim.predicate, claim.object) == (SHONDRA, POSSESSION, plants(7))


def test_both_readers_spellings_of_a_numeral_are_read():
    """The hand grammar puts an ``Entity`` under ``count``, the treebank reader a string.

    ``semantics_bridge._number_of`` reads only the first, which is why the bridge recovered
    zero of the numbers in the twelve ``reasoning.gsm8k`` dev problems: the agent reads them
    with the treebank parser.
    """
    plugin = QuantityPlugin()
    as_entity = Frame(POSSESSION, {"subject": Entity("name", "Toni", ref=TONI),
                               "object": Entity("description", "7 plants",
                                                {"count": Entity("number", "7"), "noun": "plant"})})
    (claim,) = plugin.observe(as_entity)
    assert claim.object == plants(7)


def test_an_owner_the_discourse_resolved_is_filed_under_its_reference_not_its_wording():
    """The fixture explicitly binds the owner; surface spelling must not replace it."""
    subject = Entity("pronoun", "I", {"person": 1}, Ref("agent:user"))
    frame = Frame(POSSESSION, {"subject": subject,
                           "object": Entity("description", "3 apples", {"count": "3", "noun": "apple"})})
    (claim,) = QuantityPlugin().observe(frame)
    assert claim.subject == Ref("agent:user")


# ------------------------------------------------------------------ through the agent


def test_the_agent_answers_a_quantity_question_from_what_the_plugin_holds():
    plugin = AmountOnlyFixture()
    plugin.remember(SHONDRA, POSSESSION, plants(3))
    plugin.remember(SHONDRA, POSSESSION, plants(4))
    agent = Agent([plugin])
    outcome = answer(agent, POSSESSION, Entity("name", "Shondra", ref=SHONDRA), capability="amount_of_" + POSSESSION)
    assert outcome.status == "answered" and outcome.answer == [plants(7)]
    assert plugin.display(plants(7)) == "7 plant"


def test_the_agent_says_it_does_not_know_rather_than_adding_apples_to_pears():
    plugin = AmountOnlyFixture()
    plugin.remember(SHONDRA, POSSESSION, Quantity(3, Unit.of("apple")))
    plugin.remember(SHONDRA, POSSESSION, Quantity(4, Unit.of("pear")))
    outcome = answer(Agent([plugin]), POSSESSION, Entity("name", "Shondra", ref=SHONDRA), capability="amount_of_" + POSSESSION)
    assert outcome.status == "unknown" and "apple" in outcome.reason


def test_a_capability_with_nothing_to_say_never_reports_an_empty_answer():
    """An informing capability that reports ``applied`` and reveals nothing makes
    ``Agent._look`` answer with an empty list, which the reply renders as the confident
    "There is nothing there." and every scorer counts as a commitment. So it rejects."""
    outcome = answer(Agent([QuantityPlugin()]), POSSESSION, Entity("name", "Shondra", ref=SHONDRA))
    assert outcome.status == "unknown"
    assert outcome.answer is None


def test_properties_are_counted_over_the_store():
    agent = Agent([QuantityPlugin()])
    grounded_subject_turn(agent, "the report is red.", REPORT)
    grounded_subject_turn(agent, "the report is big.", REPORT)
    outcome = answer(agent, POSSESSION, Entity("description", "report", {"noun": "report", "definite": True}, ref=REPORT), capability="count_properties")
    assert outcome.status == "answered"
    assert outcome.answer == [Quantity(2, Unit.of("property"))]


def test_an_unlearned_question_never_selects_property_measurement_from_store_shape():
    """Stored activity cannot substitute for an admitted question-to-measurement plan."""
    agent = Agent([QuantityPlugin()])
    grounded_subject_turn(agent, "they raised 2100 dollars.", THEM)
    outcome = answer(agent, POSSESSION, Entity("pronoun", "they", {"person": 3}, ref=THEM))
    assert outcome.status == "unknown"
    assert agent.informing_model is None and outcome.receipt is None


def test_nothing_known_about_a_thing_is_not_zero_properties():
    outcome = answer(Agent([QuantityPlugin()]), POSSESSION, Entity("name", "Nobody", ref=Ref("fixture:nobody")), capability="count_properties")
    assert outcome.status == "unknown"


def test_available_amount_and_property_capabilities_do_not_authorize_informing_without_a_model():
    plugin = QuantityPlugin()
    plugin.remember(REPORT, POSSESSION, Quantity(12, Unit.of("page")))
    agent = Agent([plugin])
    grounded_subject_turn(agent, "the report is red.", REPORT)
    grounded_subject_turn(agent, "the report is big.", REPORT)
    outcome = answer(agent, POSSESSION, Entity("description", "report", {"noun": "report", "definite": True}, ref=REPORT))
    assert outcome.status == "unknown"
    assert agent.informing_model is None
    assert {cap.name for cap in plugin.capabilities()} == {"amount_of_" + POSSESSION, "count_properties"}
    assert outcome.receipt is None


def test_the_plugin_offers_no_way_to_change_the_world():
    """It only ever looks. A capability with effects could be chosen for a request."""
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, POSSESSION, plants(3))
    assert all(cap.effects == () and cap.effect_kind == "read" for cap in plugin.capabilities())


# ------------------------------------------------------------------ read from English


@needs_parser
def test_a_question_in_english_reaches_the_plugin():
    from tensorcode.agent.understand import LearnedReader

    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", plants(7))
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
                                 informing_capability='amount_of_have')
    assert "7" in turn.reply
    assert [o.status for o in turn.outcomes] == ["answered"]


@needs_parser
def test_taught_count_question_preserves_unbound_counted_noun_instead_of_broadening():
    from tensorcode.agent.understand import LearnedReader
    from dependency_meaning_fixtures import teach_speech_family
    from tensorcode.learning.speech_act import SpeechActLabel
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, 'have', plants(7))
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
    """Measured over the twelve ``reasoning.gsm8k`` dev items: 0 right, 0 wrong, 12
    abstained. The numbers are in the sentences and the plugin can hold them, but no
    statement is ever shown to a plugin (see ``needed_from_the_agent``), so nothing
    recorded reaches the question — and the agent says so instead of inventing a total."""
    from tensorcode.agent.understand import LearnedReader

    turn = Agent([QuantityPlugin()], reader=LearnedReader()).turn(
        "Shondra has 7 fewer plants than Toni. Toni has 60% more plants than Frederick. "
        "If Frederick has 10 plants, how many plants does Shondra have?")
    assert not any(o.status in ("answered", "done") for o in turn.outcomes)


def test_explicit_kind_measurements_separate_equal_units_and_preserve_derivation():
    plant_kind, coin_kind = Ref('kind:plant'), Ref('kind:coin')
    plugin = QuantityPlugin()
    first = plugin.remember(SHONDRA, 'have', Quantity(3, Unit.of('item')), kind=plant_kind)
    second = plugin.remember(SHONDRA, 'have', Quantity(4, Unit.of('item')), kind=plant_kind)
    plugin.remember(SHONDRA, 'have', Quantity(20, Unit.of('item')), kind=coin_kind)
    plugin.remember(TONI, 'have', Quantity(100, Unit.of('item')), kind=plant_kind)
    assert plugin.total_of_kind(SHONDRA, 'have', plant_kind) == Quantity(7, Unit.of('item'))
    assert plugin.total_of_kind(SHONDRA, 'have', coin_kind) == Quantity(20, Unit.of('item'))
    assert isinstance(plugin.total(SHONDRA, 'have'), Unknown)
    missing = plugin.total_of_kind(SHONDRA, 'have', Ref('kind:unrecorded'))
    assert isinstance(missing, Unknown) and missing.reason == 'nothing_recorded_for_kind'
    derived = [record for record in plugin.mind.propositions()
               if record.proposition.predicate == 'total_kind:have'
               and record.proposition.role('kind') == plant_kind]
    assert len(derived) == 1
    assert set(derived[0].evidence[0].derived_from) == {first.id, second.id}


def test_kind_measurement_capability_declares_both_parameters_and_exact_answer_roles():
    from tensorcode.agent.plugin import Call
    from tensorcode.records import Proposition, Var
    kind = Ref('kind:plant')
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, 'have', plants(7), kind=kind)
    cap = next(cap for cap in plugin.capabilities() if cap.name == 'amount_of_kind_have')
    assert tuple(param.name for param in cap.params) == ('owner', 'kind')
    assert cap.informs[0].query == Proposition('total_kind:have', {'subject': Var('owner'), 'kind': Var('kind'), 'object': Var('answer')})
    action = Call(plugin.name, cap.name, (('owner', SHONDRA), ('kind', kind)))
    receipt = plugin.execute(action)
    assert receipt.status == 'applied'
    from tensorcode.derivations import DerivationReference, import_derivation, validate_record_support
    reference, = plugin.reveal(cap, dict(action.args), receipt)
    assert type(reference) is DerivationReference
    assert reference.proposition == Proposition('total_kind:have',
        {'subject': SHONDRA, 'kind': kind, 'object': plants(7)})
    target = Store()
    imported = import_derivation(target, reference)
    assert not isinstance(imported, Unknown), imported
    assert validate_record_support(target, imported.id) is True
    assert imported.evidence[0].derived_from == (reference.proposition.id,)
    assert list(plugin.reveal(cap, {'owner': TONI, 'kind': kind}, receipt)) == []
    for invalid in (Call('foreign', cap.name, action.args),
                    Call(plugin.name, cap.name, (('owner', SHONDRA),)),
                    Call(plugin.name, cap.name, (('owner', SHONDRA), ('kind', 'plant'))),
                    Call(plugin.name, cap.name, (*action.args, ('kind', kind))),
                    Call(plugin.name, cap.name, (*action.args, ('extra', kind)))):
        assert plugin.execute(invalid).status == 'rejected'


def test_kind_measurement_never_uses_unit_spelling_as_a_kind_or_mixes_dimensions():
    kind = Ref('kind:declared')
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, 'have', plants(7))
    assert isinstance(plugin.total_of_kind(SHONDRA, 'have', Ref('kind:plant')), Unknown)
    with pytest.raises(TypeError, match='explicit Ref'):
        plugin.remember(SHONDRA, 'have', plants(2), kind='plant')
    plugin.remember(SHONDRA, 'have', plants(2), kind=kind)
    plugin.remember(SHONDRA, 'have', Quantity(5, Unit.of('coin')), kind=kind)
    assert isinstance(plugin.total_of_kind(SHONDRA, 'have', kind), Unknown)
    assert isinstance(plugin.total_of_kind(SHONDRA, 'have', 'declared'), Unknown)


@pytest.mark.parametrize('qualification', ['extra_role', 'negative', 'scope', 'valid', 'modality'])
def test_kind_total_refuses_qualified_overlapping_measurements(qualification):
    from dataclasses import replace
    from tensorcode.records import Proposition, Interval
    kind = Ref('kind:plant')
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, 'have', plants(7), kind=kind)
    unsupported = Proposition('have', {'subject': SHONDRA, 'kind': kind, 'object': plants(4)})
    changes = {'extra_role': {'roles': {**unsupported.roles, 'location': Ref('room:other')}},
               'negative': {'polarity': False}, 'scope': {'scope': Ref('scope:hypothetical')},
               'valid': {'valid': Interval.at(datetime(2026, 1, 1, tzinfo=timezone.utc))},
               'modality': {'modality': 'possible'}}
    unsupported = replace(unsupported, **changes[qualification])
    plugin.mind.assert_(unsupported, Evidence(source=Ref('test:qualified-measurement'),
                                             observed_at=datetime.now(timezone.utc)))
    result = plugin.total_of_kind(SHONDRA, 'have', kind)
    assert isinstance(result, Unknown) and result.reason == 'qualified_kind_measurement'
    assert not [r for r in plugin.mind.propositions() if r.proposition.predicate == 'total_kind:have']


def test_explicit_property_measurement_does_not_infer_intent_from_amounts_or_relations():
    from tensorcode.records import Proposition
    plugin = QuantityPlugin()
    agent = Agent([plugin])
    evidence = Evidence(source=Ref('test:record'), observed_at=datetime.now(timezone.utc))
    agent.store.assert_(Proposition('colour', {'subject': REPORT, 'value': 'red'}), evidence)
    agent.store.assert_(Proposition('owned_by', {'subject': REPORT, 'owner': SHONDRA}), evidence)
    plugin.remember(REPORT, POSSESSION, plants(7))
    assert plugin.count_properties(REPORT) == Quantity(1, Unit.of('property'))
