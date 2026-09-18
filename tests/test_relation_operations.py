"""Applying a relation to two values read out of evidence, and refusing when it cannot.

The refusals are the substance of this file: an operation that guesses when a value is missing or
when two values are of different dimensions is worse than one that says so, because a guess is
indistinguishable from an answer downstream.
"""

from __future__ import annotations

from tensorcode.outcomes import Unknown
from tensorcode.quantity import Quantity, Unit
from tensorcode.relation import (
    Relation,
    Resolved,
    Values,
    apply,
    complement_of,
    difference_in,
    quantities_offered,
    read,
    resolve,
    year_in,
)

TRAPERO = ("Pablo Trapero", "Pablo Trapero (born 4 October 1971) is an Argentine film director.")
FORD = ("Aleksander Ford", "Aleksander Ford (born 24 November 1908) was a Polish film director.")


class TestOrder:
    def test_returns_the_named_candidate_not_a_span(self):
        out = resolve("Who was born first, Pablo Trapero or Aleksander Ford?", (TRAPERO, FORD))
        assert isinstance(out, Resolved) and out.text == "Aleksander Ford"
        assert "1908" in " ".join(out.steps)  # the working is recorded, not just the verdict

    def test_later_is_the_other_one(self):
        out = resolve("Who was born later, Pablo Trapero or Aleksander Ford?", (TRAPERO, FORD))
        assert out.text == "Pablo Trapero"

    def test_the_cue_picks_which_year(self):
        sentences = ["Smith (born 1930) died in 1994 after a long career."]
        assert year_in(sentences, "born")[0] == 1930
        assert year_in(sentences, "died")[0] == 1994

    def test_a_missing_year_refuses_and_names_the_candidate(self):
        out = resolve("Who was born first, Pablo Trapero or Aleksander Ford?",
                      (TRAPERO, ("Aleksander Ford", "Aleksander Ford was a Polish director.")))
        assert isinstance(out, Unknown) and out.reason == "value_missing"
        assert "Aleksander Ford" in out.detail


class TestMagnitude:
    def test_compares_across_spellings_of_one_dimension(self):
        out = resolve("Which film is longer, Alien or Tron?",
                      (("Alien", "Alien runs 117 minutes."), ("Tron", "Tron is 1.4 hours long.")))
        assert out.text == "Alien"  # 117 min vs 84 min, through quantity's base units

    def test_no_comparable_pair_refuses(self):
        # one film's runtime against the other's budget is not a magnitude comparison
        out = resolve("Which film is longer, Alien or Tron?",
                      (("Alien", "Alien runs 117 minutes."), ("Tron", "Tron cost 28 million dollars.")))
        assert isinstance(out, Unknown) and out.reason == "value_missing"

    def test_different_dimensions_refuse_rather_than_compare_raw_numbers(self):
        # the refusal text comes from quantity.compare, which knows the two dimensions
        comparison = read("Which is longer, Alien or Tron?")
        values = Values(values={"Alien": Quantity(117.0, Unit({"minute": 1})),
                                "Tron": Quantity(28.0, Unit({"metre": 1}))},
                        sources={"Alien": "", "Tron": ""})
        out = apply(comparison, values)
        assert isinstance(out, Unknown) and out.reason == "dimension_mismatch"
        assert "time vs length" in out.detail

    def test_the_asked_noun_selects_the_quantity(self):
        # the year and the record sales are in the same sentence; only one is a count of members
        sentences = ["The band, formed in 1994, has four members and sold 2 million records."]
        matched, _ = quantities_offered(sentences, "members")
        assert matched[0][0] == Quantity(4.0, Unit({"member": 1}))

    def test_a_bare_year_is_not_a_magnitude(self):
        matched, offered = quantities_offered(["It was founded in 1994."], "members")
        assert not matched and not offered


class TestYesNo:
    def test_same_attribute_is_answered_from_the_attribute_not_the_sentence(self):
        # both sentences contain "director"; only the nationality answers a nationality question
        out = resolve("Were Pablo Trapero and Aleksander Ford of the same nationality?",
                      (TRAPERO, FORD))
        assert out.text == "no"

    def test_same_nationality_when_they_are(self):
        out = resolve("Were Scott Derrickson and Ed Wood of the same nationality?",
                      (("Scott Derrickson", "Scott Derrickson is an American director."),
                       ("Ed Wood", "Edward Davis Wood Jr. was an American filmmaker.")))
        assert out.text == "yes"

    def test_both_requires_the_predicate_of_each_candidate(self):
        pair = (("The New Pornographers", "The New Pornographers are a Canadian indie rock band."),
                ("Kings of Leon", "Kings of Leon are an American rock band from Nashville."))
        assert resolve("Are both The New Pornographers and Kings of Leon American rock bands?",
                       pair).text == "no"
        assert resolve("Are both The New Pornographers and Kings of Leon rock bands?",
                       pair).text == "yes"


class TestShared:
    def test_answers_with_the_head_noun(self):
        out = resolve("What profession do Mike Tyson and Muhammad Ali have in common?",
                      (("Mike Tyson", "Mike Tyson is an American former professional boxer."),
                       ("Muhammad Ali", "Muhammad Ali was an American professional boxer.")))
        # the phrase, not its head: on train the gold for these is "film director", not
        # "director", and "professional tennis player", not "player"
        assert out.text == "professional boxer"

    def test_nothing_in_common_refuses(self):
        out = resolve("What do Mike Tyson and Mount Fuji have in common?",
                      (("Mike Tyson", "Mike Tyson is an American boxer."),
                       ("Mount Fuji", "Mount Fuji is a stratovolcano.")))
        assert isinstance(out, Unknown)

    def test_complement_drops_the_subject(self):
        assert complement_of("Scott Derrickson is an American director").strip() == "an American director"


class TestTransfer:
    def test_a_difference_word_problem_uses_the_same_two_values(self):
        out = difference_in("How many more apples than oranges does she have?",
                            "She has 12 apples and 5 oranges.")
        assert isinstance(out, Resolved) and out.text == "7"

    def test_a_non_difference_problem_is_not_claimed(self):
        out = difference_in("How much does she pay in total?", "Ten apples cost $2 each.")
        assert isinstance(out, Unknown) and out.reason == "not_a_difference"
