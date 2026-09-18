"""The projection: what a parsed sentence gives up when it becomes claims."""

import pytest

from tensorcode.language import ENGLISH, to_claims, understand
from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.records import Ref, Store
from tensorcode.semantics_bridge import (Link, link_in, needed_from_grammar, probability_in, quantities_in,
                                        quantities_in_text, tell_link, tell_mentions)

SOURCE = Ref("obs:utterance")


def read(text: str):
    return understand(ENGLISH, text)


def test_the_number_the_default_projection_drops_is_recovered():
    """`to_claims` turns "Anem has 12 sheep" into `Anem have sheep`; the 12 is in the parse."""
    got = read("Anem has 12 sheep")
    claims = to_claims(got.meanings[0], source=SOURCE)
    assert [str(c.object) for c, _ in claims] == ["entity:sheep"]  # the count is nowhere in it

    (mention,) = quantities_in(got.meanings[0])
    assert mention.quantity == Quantity(12, Unit.of("sheep"))
    assert mention.owner == "Anem" and mention.predicate == "have"


def test_a_rate_keeps_both_of_its_units():
    (mention,) = quantities_in(read("wood costs 5 coins per bushel").meanings[0])
    assert mention.quantity == Quantity(5, Unit.of("coin") / Unit.of("bushel"))
    assert mention.per == "bushel"


def test_written_numbers_and_currency_are_read():
    mentions = {m.of: m.quantity for m in quantities_in_text("Weng earns $12 an hour and worked 50 minutes")}
    assert mentions["dollar"] == Quantity(12, Unit.of("dollar"))
    assert mentions["minute"] == Quantity(50, Unit.of("minute"))
    assert quantities_in_text("a discount of 20%")[0].quantity.unit == Unit.of("percent")


def test_a_conditional_is_recovered_as_a_relation_between_its_clauses():
    text = "if I press Send an announcement appears"
    got = read(text)
    assert len(got.meanings) >= 2  # the grammar returns the clauses unlinked
    link = link_in(text, got.meanings)
    assert isinstance(link, Link) and (link.kind, link.relation) == ("conditional", "if_then")
    assert link.antecedent.predicate == "press" and link.consequent.predicate == "appear"


def test_a_conditional_never_enters_the_world_as_its_consequent():
    """Believing "an announcement appears" because someone said "if ..." is the whole risk."""
    mind = Store()
    text = "if I press Send an announcement appears"
    link = link_in(text, read(text).meanings)
    tell_link(mind, link, source=SOURCE)
    assert mind.claims(predicate="is_a", object="conditional")
    assert not any(r.claim.predicate == "appear" and r.claim.scope is None for r in mind.claims())


def test_because_is_read_in_the_direction_it_was_said():
    text = "the field failed because it did not rain"
    link = link_in(text, read(text).meanings)
    assert (link.kind, link.relation) == ("causal", "causes")
    assert link.antecedent.predicate == "rain"  # the cause is the clause after "because"
    assert link.antecedent.negated and link.consequent.predicate == "fail"


def test_so_reverses_that_direction():
    text = "it did not rain so the field failed"
    link = link_in(text, read(text).meanings)
    assert link.relation == "causes" and link.antecedent.predicate == "rain"


def test_a_sentence_with_no_connective_is_refused_not_invented():
    text = "the field failed"
    got = link_in(text, read(text).meanings)
    assert isinstance(got, Unknown) and got.reason in ("no_connective", "one_reading")


def test_likelihood_adverbs_are_uncalibrated_not_probabilities():
    score = probability_in("it will probably rain")
    assert score.kind == "uncalibrated" and score.value == pytest.approx(0.75)
    assert probability_in("it might rain").value < score.value
    assert isinstance(probability_in("it will rain"), Unknown)


def test_mentions_become_claims_that_keep_their_units():
    mind = Store()
    mentions = quantities_in(read("the granary holds 40 bushels").meanings[0])
    (claim,) = tell_mentions(mind, mentions, source=SOURCE, subject=Ref("entity:granary"))
    assert claim.object == Quantity(40, Unit.of("bushel"))
    assert mind.claims(Ref("entity:granary"), "hold")[0].claim.object.unit == Unit.of("bushel")


def test_the_grammar_gaps_this_works_around_are_written_down():
    """Surface-string workarounds are recorded, so they stay visible as gaps."""
    assert len(needed_from_grammar) >= 5
    assert any("conditional" in gap for gap in needed_from_grammar)
    assert any("numeral" in gap for gap in needed_from_grammar)
