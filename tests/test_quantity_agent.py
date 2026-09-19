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

from tensorcode.agent.core import Agent
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


def plants(n: float) -> Quantity:
    return Quantity(n, Unit.of("plant"))


def asking(predicate: str, subject: Entity) -> tuple[Sentence, Act]:
    """The question "how many <something> does <subject> <predicate>?", as the reader makes it.

    Built here rather than parsed so the wiring — informs, refer, execute, reveal, lookup —
    is testable without a trained model. It is the shape the treebank reader really
    produces, counted noun and all: ``Reader.speech_act`` deletes the whole wh-phrase's
    role, so *plants* is not in it.
    """
    question = Question(Frame(predicate, {"subject": subject}, {"mood": "interrogative"}), "quantity")
    act = Act("question", question, question.frame)
    return Sentence("how many …?", ("how", "many"), None, (act,)), act


def answer(agent: Agent, predicate: str, subject: Entity):
    sentence, act = asking(predicate, subject)
    return agent.handle(sentence, act, [], requests_in_message=0)


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


def test_have_is_filed_under_the_predicate_a_question_is_matched_by():
    """``Agent.sought_predicate`` turns "have" into ``has_possession``; an amount recorded
    under the verb would never be found by a question about it."""
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", plants(7))
    assert world_predicate("have") == POSSESSION
    assert [c.name for c in plugin.capabilities()] == [f"amount_of_{POSSESSION}", "count_properties"]


# ------------------------------------------------------------------ arithmetic


def test_two_amounts_of_one_kind_are_added():
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", plants(3))
    plugin.remember(SHONDRA, "have", plants(4))
    assert plugin.total(SHONDRA, "have") == plants(7)


def test_amounts_of_two_kinds_are_refused_because_the_question_lost_which_one():
    """"how many apples do I have?" and "how many pears do I have?" are the same question
    by the time a plugin sees it. Adding them would answer both wrongly."""
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", Quantity(3, Unit.of("apple")))
    plugin.remember(SHONDRA, "have", Quantity(4, Unit.of("pear")))
    got = plugin.total(SHONDRA, "have")
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
    plugin.remember(SHONDRA, "have", plants(3))
    plugin.remember(TONI, "have", Quantity(3, Unit.of("coin")))
    got = plugin.compare(TONI, SHONDRA, "have")
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
    first = plugin.remember(SHONDRA, "have", plants(3))
    second = plugin.remember(SHONDRA, "have", plants(4))
    plugin.total(SHONDRA, "have")
    (derived,) = plugin.mind.claims(subject=SHONDRA, predicate=f"total:{POSSESSION}")
    assert derived.claim.object == plants(7)
    (evidence,) = derived.evidence
    assert evidence.method == "arithmetic:sum"
    assert set(evidence.derived_from) == {first.id, second.id}


# ------------------------------------------------------------------ reading statements


def test_a_statement_s_numbers_are_kept_with_the_thing_they_were_said_of():
    frame = Frame("have", {"subject": Entity("name", "Shondra"),
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
    as_entity = Frame("have", {"subject": Entity("name", "Toni"),
                               "object": Entity("description", "7 plants",
                                                {"count": Entity("number", "7"), "noun": "plant"})})
    (claim,) = plugin.observe(as_entity)
    assert claim.object == plants(7)


def test_an_owner_the_discourse_resolved_is_filed_under_its_reference_not_its_wording():
    """"I have 3 apples" is about the user, and ``Agent.deixis`` has already said so; filing
    it under the word "I" would put it where no question could find it."""
    subject = Entity("pronoun", "I", {"person": 1}, Ref("agent:user"))
    frame = Frame("have", {"subject": subject,
                           "object": Entity("description", "3 apples", {"count": "3", "noun": "apple"})})
    (claim,) = QuantityPlugin().observe(frame)
    assert claim.subject == Ref("agent:user")


# ------------------------------------------------------------------ through the agent


def test_the_agent_answers_a_quantity_question_from_what_the_plugin_holds():
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", plants(3))
    plugin.remember(SHONDRA, "have", plants(4))
    agent = Agent([plugin])
    outcome = answer(agent, "have", Entity("name", "Shondra"))
    assert outcome.status == "answered" and outcome.answer == [plants(7)]
    assert plugin.display(plants(7)) == "7 plant"


def test_the_agent_says_it_does_not_know_rather_than_adding_apples_to_pears():
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", Quantity(3, Unit.of("apple")))
    plugin.remember(SHONDRA, "have", Quantity(4, Unit.of("pear")))
    outcome = answer(Agent([plugin]), "have", Entity("name", "Shondra"))
    assert outcome.status == "unknown" and "apple" in outcome.reason


def test_a_capability_with_nothing_to_say_never_reports_an_empty_answer():
    """An informing capability that reports ``applied`` and reveals nothing makes
    ``Agent._look`` answer with an empty list, which the reply renders as the confident
    "There is nothing there." and every scorer counts as a commitment. So it rejects."""
    outcome = answer(Agent([QuantityPlugin()]), "have", Entity("name", "Shondra"))
    assert outcome.status == "unknown"
    assert outcome.answer is None


def test_properties_are_counted_over_the_store():
    agent = Agent([QuantityPlugin()])
    agent.turn("the report is red. the report is big.")
    outcome = answer(agent, "have", Entity("description", "report", {"noun": "report", "definite": True}))
    assert outcome.status == "answered"
    assert outcome.answer == [Quantity(2, Unit.of("property"))]


def test_a_thing_the_store_only_knows_through_what_it_did_is_not_property_counted():
    """This is a wrong answer that happened. "For how many hours do they have to fundraise?"
    arrived as ``?quantity in have(subject=they)`` and the store knew two things about
    *they*, so the count came back "1 property." to a question whose answer was 9. A
    property relates a thing to a value; a fact relating it to another entity is something
    it took part in, and the question is far more likely about that."""
    agent = Agent([QuantityPlugin()])
    agent.turn("they raised 2100 dollars.")
    outcome = answer(agent, "have", Entity("pronoun", "they", {"person": 3}))
    assert outcome.status == "unknown"


def test_nothing_known_about_a_thing_is_not_zero_properties():
    outcome = answer(Agent([QuantityPlugin()]), "have", Entity("name", "Nobody"))
    assert outcome.status == "unknown"


def test_an_amount_is_preferred_to_a_property_count():
    plugin = QuantityPlugin()
    plugin.remember(Ref("entity:report"), "have", Quantity(12, Unit.of("page")))
    agent = Agent([plugin])
    agent.turn("the report is red. the report is big.")
    outcome = answer(agent, "have", Entity("description", "report", {"noun": "report", "definite": True}))
    assert outcome.answer == [Quantity(12, Unit.of("page"))]


def test_the_plugin_offers_no_way_to_change_the_world():
    """It only ever looks. A capability with effects could be chosen for a request."""
    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", plants(3))
    assert all(cap.effects == () and cap.effect_kind == "read" for cap in plugin.capabilities())


# ------------------------------------------------------------------ read from English


@needs_parser
def test_a_question_in_english_reaches_the_plugin():
    from tensorcode.agent.understand import LearnedReader

    plugin = QuantityPlugin()
    plugin.remember(SHONDRA, "have", plants(7))
    turn = Agent([plugin], reader=LearnedReader()).turn("how many plants does Shondra have?")
    assert "7" in turn.reply
    assert [o.status for o in turn.outcomes] == ["answered"]


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
